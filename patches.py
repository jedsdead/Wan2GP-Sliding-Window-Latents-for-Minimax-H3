"""Carry MiniMax H3 latents across sliding windows instead of re-encoding pixels.

Wan2GP already pins a run of previous-window frames as history conditioning: at overlap
18 it takes 17 pixel frames, encodes them in a single VAE call with
keep_all_latents=True, and submits them as an "history" keyframe block
(models/minimax_h3/pipeline.py, _add_video_history).  That works, but the pixels
it encodes are themselves VAE output from the previous window, so every window
adds one decode -> encode round trip to the carried context.

This module replaces that re-encode with the previous window's *own* latents,
captured on their way into the decoder.

Seven bindings are patched, all reversible via uninstall():

  1. MiniMaxH3VideoVAE.decode      - capture the tail of the latent tensor plus a
                                     join-frame fingerprint.
  2. MiniMaxH3AudioVAE.decode      - same for audio, plus an envelope signature.
  3. MiniMaxH3Pipeline.generate    - window bookkeeping + usability checks.
  4. MiniMaxH3Pipeline._add_video_history
                                   - substitute cached latents for the encode.
  5. MiniMaxH3Pipeline._encode_audio
                                   - substitute cached audio latents for the
                                     continuation encode (first call only).
  6. transformer.build_packed_sequence
  7. transformer.build_ref2va_packed_sequence
                                   - correct the time coordinates (see below).

Phase alignment, and why the coordinate correction is small
-----------------------------------------------------------
H3 compresses time periodically, not as a single clip-initial special case.
pipeline.video_latent_frames is 2 + ((N-5)//17)*5 - seventeen pixel frames per
five latents - and _FRAME_PER_TOKEN = (1, 4, 4, 4, 4) in components/packing.py is
that same period.  Latent i covers _FRAME_PER_TOKEN[i % 5] pixel frames, so every
fifth latent covers one frame and the rest cover four, all the way through a
window.

packing._video_t_grid walks the pattern from position 0.  That means a carried
block is laid out correctly *if and only if* its first latent sits at an absolute
index divisible by 5.  Since a window always holds 2 + 5k latents, slicing the
last n gives a phase-aligned block exactly when n = 2 (mod 5).  n = 7 is the
smallest useful choice and carries 22 pixel frames, a little more context than
the 17 the native re-encode provides.

With the phase right, the internal spacing needs no correction at all.  What
remains is a pure translation: the native block stops one frame short of the join
(history_frames excludes it), while a carried block's last latent contains the
join frame, so reference_t_span places the whole block one pixel frame too early.
Patch (4) shifts everything anchored to target_origin up by that single frame.

    n=7, carried    truth   -21  -20  -16  -12   -8   -4   -3
                    layout  -22  -21  -17  -13   -9   -5   -4
                            spacing identical, translation -1

Set SWL_FIX_COORDS=0 to skip patches (6) and (7) and reproduce the uncorrected
layout for A/B purposes.

Job boundaries
--------------
One generate() call is one window, but consecutive generate() calls are not
necessarily consecutive windows of the same video.  Wan2GP's queue runs jobs
back to back in the same process, and a job that repeats a prompt restarts its
window loop for every sample, so "the previous generate call" is routinely the
last window of an unrelated video.

wgp.py passes window_no on every call (it lands in generate's **kwargs), and it
restarts at 1 for each new job and each repeat within a job.  That is the exact
boundary signal: patch (3) drops the cache whenever it sees window_no <= 1, and
the carry paths additionally require the cached window to be numbered exactly one
lower than the window asking for it.  A first window is therefore never served
from cache, whatever its pixels look like - including a continue-video job whose
source happens to be the previous job's output.

If window_no is missing the caller is not wgp.py's window loop and the boundary
cannot be established, so the plugin stays inert rather than guess.  Set
SWL_REQUIRE_WINDOW_NO=0 to fall back to fingerprint-only matching for a
front end that drives the pipeline itself.

Configuration
-------------
The environment variables below set the initial values only.  CONFIG is read at
each decision point rather than captured at install time, so the Gradio panel
changes behaviour live and anything saved from it wins over the environment.

  SWL_ENABLE        1/0    master switch                          (default 1)
  SWL_FIX_COORDS    1/0    apply the coordinate correction        (default 1)
  SWL_REQUIRE_WINDOW_NO
                     1/0    refuse to carry when the caller does
                            not report window_no                   (default 1)
  SWL_LATENTS       int    carried latent count; 0 uses 7.  Must
                            be 2 (mod 5): 7, 12, 17...             (default 0)
  SWL_AUDIO         1/0    also carry audio latents               (default 1)
  SWL_VERBOSE       1/0    log every engage / fall back           (default 1)
  SWL_MATCH_TOL     float  join-frame fingerprint tolerance       (default 0.03)
  SWL_AUDIO_TOL     float  audio envelope tolerance               (default 0.15)
  SWL_DIAGNOSE      1/0    also run the native encode and log how
                            the two latent sets differ.  Costs one
                            encode per window                      (default 0)
  SWL_MOMENT_MATCH  1/0    rescale carried latents onto the
                            encoded per-channel distribution.
                            Also costs one encode per window       (default 0)
"""

import os
import threading

import numpy as np
import torch


_FRAME_RESCALE = 5.0 / 3.0          # packing._FRAME_RESCALE
_LATENT_PERIOD = 5                  # latents per 17 pixel frames
_PHASE_RESIDUE = 2                  # a window holds 2 + 5k latents, so n must be 2 (mod 5)
_RESERVE_DEFICIT = 1                # frames of translation error at the join
_DEFAULT_LATENTS = 7                # smallest phase-aligned block worth carrying (22 frames)
_AUDIO_CHANNELS = 2                 # packing.MINIMAX_H3_AUDIO_CHANNELS
_VAE_SPATIAL_FACTOR = 16            # confirmed by pipeline's ceil(target_height / 16)
_FINGERPRINT_GRID = (16, 16)        # resolution-independent frame signature
_ALIGN_SEARCH_FRAMES = 12           # decoded tail frames fingerprinted for the search
_ALIGN_EXPECTED_OFFSET = -2         # where the join frame must sit; see _check_alignment
_ALIGN_DECISIVE = 0.6               # a rival offset must beat expected by this factor
_AUDIO_SAMPLE_RATE = 32000          # pipeline.AUDIO_SAMPLE_RATE
_AUDIO_SIGNATURE_BINS = 32
_AUDIO_SIGNATURE_SECONDS = 0.5
_MAX_CACHED_AUDIO_LATENTS = 64      # 18 frames at 40 latent fps is ~30
_CLIP_LENGTH = 17                   # H3 window quantum
_COLOUR_AXES = ("brightness", "contrast", "saturation", "cast")
_COLOUR_SPAN = 5                    # frames measured each side of a join
_COLOUR_POOL = 64                   # measurement is pooled to this width
_COLOUR_CONTINUATION_SPAN = 48      # a continuation can measure more; see README
_COLOUR_CHUNK = 4_000_000           # values per pixel-correction chunk
_MAX_CACHED_LATENTS = 17            # floor for the cached tail; covers n = 7, 12 and 17.
                                    # _cache_depth() raises this to match a larger
                                    # SWL_LATENTS so the cap can never silently
                                    # undercut the configured slice.


def _flag(name, default):
    return str(os.environ.get(name, default)).strip().lower() not in ("0", "false", "no", "")


class _Config:
    def __init__(self):
        self.enable = _flag("SWL_ENABLE", "1")
        self.fix_coords = _flag("SWL_FIX_COORDS", "1")
        self.require_window_no = _flag("SWL_REQUIRE_WINDOW_NO", "1")
        self.verbose = _flag("SWL_VERBOSE", "1")
        try:
            self.latents = max(0, int(os.environ.get("SWL_LATENTS", "0")))
        except ValueError:
            self.latents = 0
        try:
            self.match_tolerance = float(os.environ.get("SWL_MATCH_TOL", "0.03"))
        except ValueError:
            self.match_tolerance = 0.03
        self.audio = _flag("SWL_AUDIO", "1")
        self.diagnose = _flag("SWL_DIAGNOSE", "0")
        self.moment_match = _flag("SWL_MOMENT_MATCH", "0")
        try:
            self.audio_tolerance = float(os.environ.get("SWL_AUDIO_TOL", "0.15"))
        except ValueError:
            self.audio_tolerance = 0.15
        # Colour correction, applied to the carried latents.  Off by default:
        # a plugin update should not quietly start altering conditioning that
        # was already working.
        # Search a run of frames for the join rather than testing one offset.
        self.align_search = _flag("SWL_ALIGN_SEARCH", "1")
        self.colour = _flag("SWL_COLOUR", "0")
        self.colour_scope = (os.environ.get("SWL_COLOUR_SCOPE", "latents")
                             .strip().lower())
        if self.colour_scope not in ("latents", "both"):
            self.colour_scope = "latents"
        self.colour_match = (os.environ.get("SWL_COLOUR_MATCH", "previous")
                             .strip().lower())
        if self.colour_match not in ("previous", "first"):
            self.colour_match = "previous"
        # Offsets lean on the linear latent-to-RGB map lightly; gains lean on
        # it harder.  See colour.py.  Gains are opt in.
        axes = os.environ.get("SWL_COLOUR_AXES", "brightness,cast")
        self.colour_axes = tuple(a.strip() for a in axes.split(",")
                                 if a.strip() in _COLOUR_AXES)
        for name, key, default in (("colour_strength", "SWL_COLOUR_STRENGTH", 1.0),
                                   ("colour_scene", "SWL_COLOUR_SCENE", 0.06),
                                   ("colour_max", "SWL_COLOUR_MAX", 0.15)):
            try:
                setattr(self, name, float(os.environ.get(key, str(default))))
            except ValueError:
                setattr(self, name, default)


CONFIG = _Config()


def _cache_depth():
    """How many latents to keep from each window's tail.

    Read at capture time rather than fixed at import, because the panel changes
    CONFIG.latents live.  Keeping at least the configured slice is what makes the
    12 and 17 settings reachable: a fixed cap below the requested n turned every
    window into a "cached tail holds N latents" fall back, which was logged but
    easy to miss against the other fallback reasons.
    """
    return max(_MAX_CACHED_LATENTS, int(CONFIG.latents or _DEFAULT_LATENTS))


def _log(message):
    state = globals().get("STATE")
    if state is not None:
        state.last_message = message
    if CONFIG.verbose:
        print(f"[sliding-window-latents] {message}")


def _to_unit(tensor):
    """Map a frame to float32 in [0, 1] without assuming its input range.

    Decoded output is [-1, 1]; continuation video reaching _add_video_history may
    be uint8 or float.  Guessing wrong here would make every fingerprint
    comparison fail, so the range is sniffed rather than assumed.
    """
    if tensor.dtype == torch.uint8:
        return tensor.float().div_(255.0)
    values = tensor.float()
    low, high = float(values.min()), float(values.max())
    if low < -0.05:
        return values.add(1.0).div_(2.0)
    if high > 1.5:
        return values.div(255.0)
    return values


def _fingerprint(frame_chw):
    """A small resolution-independent signature of one frame."""
    gray = _to_unit(frame_chw.detach()).mean(dim=0, keepdim=True)
    pooled = torch.nn.functional.adaptive_avg_pool2d(gray.unsqueeze(0), _FINGERPRINT_GRID)
    return pooled[0, 0].to(torch.float32).cpu()


def _audio_signature(waveform):
    """A peak-normalised RMS envelope of the last half second, mono."""
    audio = waveform.detach().float()
    while audio.ndim > 2:
        audio = audio[0]
    if audio.ndim == 2:
        audio = audio.mean(dim=0)
    window = min(int(audio.shape[-1]), int(_AUDIO_SIGNATURE_SECONDS * _AUDIO_SAMPLE_RATE))
    audio = audio[-window:]
    if audio.shape[-1] < _AUDIO_SIGNATURE_BINS:
        return None
    trimmed = (audio.shape[-1] // _AUDIO_SIGNATURE_BINS) * _AUDIO_SIGNATURE_BINS
    envelope = audio[:trimmed].reshape(_AUDIO_SIGNATURE_BINS, -1).pow(2).mean(dim=-1).sqrt()
    peak = float(envelope.max())
    if peak < 1e-5:
        return torch.zeros_like(envelope).cpu()          # silence matches silence
    return (envelope / peak).cpu()


def _audio_matches(cached, incoming):
    if cached is None or incoming is None:
        return False, "no audio signature"
    if float(cached.abs().max()) < 1e-6 and float(incoming.abs().max()) < 1e-6:
        return True, None
    difference = float((cached - incoming).abs().mean())
    if difference > CONFIG.audio_tolerance:
        return False, f"audio mismatch (envelope difference {difference:.4f})"
    return True, None


class _State:
    """Cross-window state.  One generate() call == one window."""

    def __init__(self):
        self.lock = threading.Lock()
        self.generation = 0
        # wgp.py's window number for the call in progress; None when the caller
        # does not report one.  Restarts at 1 for every job and every repeat, so
        # it is what separates "next window" from "next queue entry".
        self.window_no = None
        # cached tail of the previous window
        self.latents = None
        self.signature = None
        self.signatures = None
        self.source_generation = -1
        self.source_window_no = None
        self.usable = False
        self.expected_latents = None
        self.total_latents = None
        # Set the first time the stock _add_video_history is seen returning a
        # value.  See _patched_add_video_history.
        self.history_returns_value = False
        # set per stage by the _add_video_history patch, read by build_packed_sequence
        self.active_latents = None
        self.plan = None
        # audio side
        self.audio_latents = None
        self.audio_signature = None
        self.audio_generation = -1
        self.audio_source_window_no = None
        self.audio_encode_calls = 0
        # colour: statistics survive a window, its frames do not
        self.colour_reference = None     # what the next window is matched to
        self.colour_first = None         # window 1's opening, for "first" mode
        self.colour_total = None         # running (gain, offset) in Y'CbCr
        self.colour_correction = None    # (A, b) built from the total
        self.colour_report = None
        self.colour_step = None          # (A, b) applied to this window's capture
        self.colour_seeded = False       # reference came from a continuation
        self.last_message = "idle"
        self.stats = {"engaged": 0, "fell_back": 0, "audio_engaged": 0,
                      "colour_applied": 0}

    def invalidate(self, reason):
        if self.latents is not None:
            _log(f"cache dropped: {reason}")
        self.latents = None
        self.signature = None
        self.source_window_no = None
        self.audio_latents = None
        self.audio_signature = None
        self.audio_source_window_no = None
        self.usable = False
        self.total_latents = None
        self.colour_reference = None
        self.colour_first = None
        self.colour_total = None
        self.colour_correction = None
        self.colour_report = None
        self.colour_step = None
        self.colour_seeded = False


STATE = _State()

VERSION = "1.0.0"

_PACKING = None                     # packing module, resolved by _preflight()

_ORIGINALS = {}
_ORIGINAL_OWNERS = {}


# --------------------------------------------------------------------------
# 1. latent capture
# --------------------------------------------------------------------------

def _colour_stats(decoded, start, stop):
    """Y'CbCr statistics of a slice of decoded frames.

    Pooled to a small grid first.  The statistics wanted are the frame's
    overall grade, so full resolution costs memory without changing the
    answer, and pooling makes the measurement independent of output size.
    """
    from . import colour
    frames = decoded[0, :, start:stop]
    if frames.shape[1] == 0:
        return None
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        frames.permute(1, 0, 2, 3).float(), (_COLOUR_POOL, _COLOUR_POOL))
    unit = _to_unit(pooled).permute(0, 2, 3, 1).cpu().numpy()
    return colour.ycbcr_stats(unit)


def _colour_seed(history_video):
    """Take the colour reference from the frames a continuation hands in.

    Every other reference is a window this plugin watched being generated.
    The first window of a Continue Video run has no such predecessor, so
    without this it is the one window that begins with nothing to match to -
    across the join between the source video and the new one, which is the
    most visible join there is.

    Those frames are the source video's own grade, which is exactly what the
    new window should come out looking like.
    """
    from . import colour
    if not CONFIG.colour or STATE.colour_reference is not None:
        return
    frames = history_video
    if frames is None or frames.shape[1] == 0:
        return
    span = min(_COLOUR_CONTINUATION_SPAN, frames.shape[1])
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        frames[:, -span:].permute(1, 0, 2, 3).float(),
        (_COLOUR_POOL, _COLOUR_POOL))
    unit = _to_unit(pooled).permute(0, 2, 3, 1).cpu().numpy()
    STATE.colour_reference = colour.ycbcr_stats(unit)
    STATE.colour_first = STATE.colour_reference
    STATE.colour_seeded = True
    _log(f"colour: reference taken from {span} frames of the video being "
         f"continued")


def _correct_pixels(decoded, gain, offset):
    """Apply the correction to the decoded window itself.

    Only reached in `both` scope. Frames are signed [-1, 1]; the correction
    is defined on unit RGB, so it is rebased here rather than in colour.py.

    Done a chunk at a time so the working set does not scale with output
    size, and in place where the tensor allows it. A tensor that refuses an
    in-place write - inference mode is the usual reason - falls back to a
    corrected copy, decided by a probe write before any real work so a window
    can never end up corrected at one end and not the other.
    """
    from . import colour
    Q, o = colour.rgb_affine(gain, offset)
    matrix = torch.as_tensor(Q, dtype=torch.float32, device=decoded.device)
    shift = torch.as_tensor(o, dtype=torch.float32, device=decoded.device)

    try:
        probe = decoded[:, :, :1]
        probe.mul_(1.0)
        in_place = True
    except Exception:
        in_place = False
    out = decoded if in_place else decoded.clone()

    total = out.shape[2]
    step = max(1, _COLOUR_CHUNK // max(1, out.shape[-1] * out.shape[-2]))
    for start in range(0, total, step):
        block = out[:, :, start:start + step].float()
        unit = block.add(1.0).mul_(0.5)
        moved = torch.einsum("rk,bkthw->brthw", matrix, unit) + shift.view(1, -1, 1, 1, 1)
        moved = moved.mul_(2.0).sub_(1.0).clamp_(-1.0, 1.0)
        out[:, :, start:start + step] = moved.to(out.dtype)
    return out


def _colour_noise(decoded):
    """How much the statistics move inside one window, where there is no join.

    Two interior groups the same size as the measurement span.  Whatever
    separates them is the wobble a join measurement has to beat.
    """
    total = decoded.shape[2]
    if total < 4 * _COLOUR_SPAN:
        return None
    mid = total // 2
    left = _colour_stats(decoded, mid - 2 * _COLOUR_SPAN, mid - _COLOUR_SPAN)
    right = _colour_stats(decoded, mid, mid + _COLOUR_SPAN)
    if left is None or right is None:
        return None
    return {"mean": right["mean"] - left["mean"],
            "std": right["std"] - left["std"]}


def _measure_colour(decoded):
    """Compare this window's opening against the reference, and set up (A, b).

    Called at decode, which is the only moment both the new window's frames
    and the previous window's statistics exist at once.
    """
    from . import colour
    head = _colour_stats(decoded, 0, _COLOUR_SPAN)
    tail = _colour_stats(decoded, max(decoded.shape[2] - _COLOUR_SPAN, 0),
                         decoded.shape[2])
    if head is None or tail is None:
        return decoded

    if STATE.colour_first is None:
        STATE.colour_first = head
    reference = (STATE.colour_first if CONFIG.colour_match == "first"
                 else STATE.colour_reference)

    report = None
    if reference is not None:
        gain, offset, report = colour.measure(
            reference, head, noise=_colour_noise(decoded),
            strength=CONFIG.colour_strength,
            scene_threshold=CONFIG.colour_scene,
            max_correction=CONFIG.colour_max,
            axes=CONFIG.colour_axes)
        if gain is None:
            _log(f"colour: {report['rejected']}, leaving this window alone")
        elif report["is_noop"]:
            _log("colour: measured drift is inside the noise floor, no correction")
        elif CONFIG.colour_scope == "both":
            # The window itself is rewritten, so the reference the next window
            # measures against is corrected too and the loop closes on its own
            # - no total to carry.  The same step goes onto the latents about
            # to be cached, which is what keeps conditioning and video from
            # drifting apart.
            decoded = _correct_pixels(decoded, gain, offset)
            STATE.colour_step = colour.latent_correction(gain, offset)
            tail = _colour_stats(decoded, max(decoded.shape[2] - _COLOUR_SPAN, 0),
                                 decoded.shape[2])
            _log(f"colour: corrected the window, gain "
                 f"{np.round(gain, 4).tolist()} offset "
                 f"{np.round(offset, 4).tolist()}")
        else:
            # Accumulated, not replaced.  The video is not rewritten, so the
            # reference keeps moving; a per-window correction would fire on
            # alternate windows and let the drift back in at half rate.
            STATE.colour_total = colour.compose(STATE.colour_total, (gain, offset))
            _log(f"colour: step gain {np.round(gain, 4).tolist()} "
                 f"offset {np.round(offset, 4).tolist()}  ->  total gain "
                 f"{np.round(STATE.colour_total[0], 4).tolist()} offset "
                 f"{np.round(STATE.colour_total[1], 4).tolist()}")

    if CONFIG.colour_scope == "both":
        STATE.colour_correction = None      # the capture is corrected instead
    else:
        STATE.colour_correction = (colour.latent_correction(*STATE.colour_total)
                                   if STATE.colour_total is not None else None)
    STATE.colour_report = report
    # The next window is matched to what this one actually looks like.
    STATE.colour_reference = tail
    return decoded


def _check_alignment(signatures, signature, incoming):
    """Locate the join frame in the cached decoded tail.

    The next window hands back `continuation[:, -overlap:-1]`, whose last frame
    is the decoded window's frame -2 - the join frame itself is excluded and
    pinned separately.  So the incoming frame must match offset -2 of the
    cached tail, and matching anywhere else means the cached latents do not end
    where the emitted video ends.

    Testing that one offset alone cannot detect this.  Neighbouring frames of a
    slow shot differ by far less than the tolerance, so a tail off by several
    frames passes, the carried block is then placed as though it ended at the
    join, and the model is handed footage from after the join as though it were
    before it - which it duly renders again.  Searching the run turns that
    silent misalignment into a number.

    Returns (ok, detail).  `detail` names the measured skew in frames when the
    join sits somewhere other than where it must.
    """
    if signatures is None or signatures.shape[0] < abs(_ALIGN_EXPECTED_OFFSET):
        if signature is None:
            return False, "no join fingerprint captured"
        difference = float((incoming - signature).abs().mean())
        if difference > CONFIG.match_tolerance:
            return False, (f"content mismatch (join difference {difference:.4f} > "
                           f"{CONFIG.match_tolerance:.4f})")
        return True, None

    errors = (signatures - incoming[None]).abs().mean(dim=(1, 2))
    depth = int(signatures.shape[0])
    expected_row = depth + _ALIGN_EXPECTED_OFFSET
    expected_error = float(errors[expected_row])
    best_row = int(errors.argmin())
    best_error = float(errors[best_row])
    skew = best_row - expected_row

    if not CONFIG.align_search:
        pass
    elif skew != 0 and best_error < expected_error * _ALIGN_DECISIVE:
        # Decisively better elsewhere: the tail really is displaced.  The join
        # frame sitting EARLIER in the cached tail than offset -2 (a negative
        # skew) means the cache holds frames from after the join, which is the
        # case that gets rendered a second time.  Note only +1 is reachable in
        # the other direction, since the cached fingerprints stop at offset -1.
        direction = ("cached latents run past the end of the emitted video"
                     if skew < 0 else "cached latents stop short of it")
        return False, (f"tail misaligned by {skew:+d} frame(s) - {direction} "
                       f"(join matches offset {best_row - depth:+d} at "
                       f"{best_error:.4f}, expected {_ALIGN_EXPECTED_OFFSET:+d} "
                       f"at {expected_error:.4f})")

    if expected_error > CONFIG.match_tolerance:
        return False, (f"content mismatch (join difference {expected_error:.4f} > "
                       f"{CONFIG.match_tolerance:.4f}); cache belongs to a "
                       f"different video")
    return True, None


def _patched_decode(self, latents):
    decoded = _ORIGINALS["decode"](self, latents)
    STATE.colour_step = None
    if CONFIG.colour:
        try:
            # NOT `_measure_colour(decoded) or decoded`: `or` calls bool() on
            # the tensor, which raises "Boolean value of Tensor with more than
            # one element is ambiguous", and the except below then swallowed it
            # and silently disabled colour on every window.
            corrected = _measure_colour(decoded)
            if corrected is not None:
                decoded = corrected
        except Exception as error:          # never break a render
            # Loud, and worded as a defect rather than a fallback.  The previous
            # wording read like a considered decline, which is how a hard error
            # went unnoticed on every window for a whole release.
            _log(f"BUG: colour correction raised and was skipped for this "
                 f"window - {type(error).__name__}: {error}. This is not a "
                 f"normal fallback; please report it.")
            STATE.colour_correction = STATE.colour_step = None
    try:
        tail = latents[:, :, -_cache_depth():].detach().to(torch.float32).cpu()
        if STATE.colour_step is not None:
            # `both` scope corrected the pixels; the cached latents have to
            # move with them or the next window is conditioned on a grade the
            # video no longer has.
            from . import colour
            tail = colour.apply_torch(tail, *STATE.colour_step)
        # Frame -2 of the decoded window is the last frame the next window will
        # hand back as history (continuation[:, -overlap:-1] excludes the join
        # frame itself).  Fingerprinting it lets the next window confirm these
        # latents belong to the video it is actually continuing.
        signature = _fingerprint(decoded[0, :, -2]) if decoded.shape[2] >= 2 else None
        # A run of tail fingerprints, newest last, so the next window can find
        # *where* the join frame sits instead of only asking whether it is where
        # it was assumed to be.
        depth = min(_ALIGN_SEARCH_FRAMES, int(decoded.shape[2]))
        signatures = (torch.stack([_fingerprint(decoded[0, :, i])
                                   for i in range(-depth, 0)]) if depth >= 2 else None)
        with STATE.lock:
            STATE.latents = tail
            STATE.signature = signature
            STATE.signatures = signatures
            STATE.source_generation = STATE.generation
            STATE.total_latents = int(latents.shape[2])
            plan = STATE.plan or {}
            STATE.source_window_no = plan.get("window_no")
            STATE.usable = bool(plan.get("usable"))
            STATE.expected_latents = plan.get("expected_latents")
    except Exception as error:                                   # never break a render
        _log(f"capture failed, continuing without cache: {error!r}")
        STATE.invalidate("capture error")
    return decoded


def _patched_audio_decode(self, latents):
    waveform = _ORIGINALS["audio_decode"](self, latents)
    try:
        tail = latents[..., -_MAX_CACHED_AUDIO_LATENTS:].detach().to(torch.float32).cpu()
        with STATE.lock:
            STATE.audio_latents = tail
            STATE.audio_signature = _audio_signature(waveform)
            STATE.audio_generation = STATE.generation
            STATE.audio_source_window_no = (STATE.plan or {}).get("window_no")
    except Exception as error:
        _log(f"audio capture failed, continuing without it: {error!r}")
        with STATE.lock:
            STATE.audio_latents = None
    return waveform


def _patched_encode_audio(self, waveform):
    """Substitute the previous window's audio latents for the continuation encode.

    Unlike the video side there is no coordinate work to do: _fill_audio_positions
    lays audio conditions out as origin + arange(length), a uniform integer axis
    with none of the (1, 4, 4, 4, 4) periodicity, so a tail slice needs no phase
    alignment and no translation.

    The real encode is always run.  It fixes the latent count that the pipeline's
    history/boundary split is computed from, and it is the fallback.  The audio
    VAE is cheap beside the video one, so paying for it and discarding the result
    buys a lot of safety for little time.
    """
    native = _ORIGINALS["encode_audio"](self, waveform)

    STATE.audio_encode_calls += 1
    if not (CONFIG.enable and CONFIG.audio):
        return native
    if STATE.audio_encode_calls != 1:
        # generate() encodes the continuation first; later calls are reference
        # clips or a supplied soundtrack, which must not be replaced.
        return native
    if not (STATE.plan or {}).get("continuation_audio"):
        # ...but "first call" only means the continuation when this window has a
        # continuation to encode.  Without one, call 1 is a reference clip or the
        # target audio condition, and substituting into either would be wrong.
        return native

    with STATE.lock:
        cached = STATE.audio_latents
        signature = STATE.audio_signature
        source = STATE.audio_generation
        source_window_no = STATE.audio_source_window_no
        generation = STATE.generation
        usable = STATE.usable

    if cached is None or source != generation - 1 or not usable:
        return native

    # The envelope match below cannot carry this on its own: silence matches
    # silence by design, so a silent job following a silent job would otherwise
    # inherit the previous video's audio latents at its first window.
    continues, reason = _continues_cached_window(source_window_no)
    if not continues:
        _log(f"re-encoding continuation audio ({reason})")
        return native

    matched, reason = _audio_matches(signature, _audio_signature(waveform))
    if not matched:
        _log(f"re-encoding continuation audio ({reason})")
        return native

    wanted = int(native.shape[-1])
    if wanted > cached.shape[-1]:
        _log(f"re-encoding continuation audio (cached tail holds "
             f"{cached.shape[-1]} latents, need {wanted})")
        return native
    if tuple(cached.shape[1:-1]) != tuple(native.shape[1:-1]):
        _log(f"re-encoding continuation audio (layout {tuple(cached.shape)} vs "
             f"{tuple(native.shape)})")
        return native

    STATE.stats["audio_engaged"] += 1
    _log(f"carried {wanted} audio latents from the previous window")
    return cached[..., -wanted:].to(dtype=native.dtype).clone()


# --------------------------------------------------------------------------
# 2. window bookkeeping
# --------------------------------------------------------------------------

def _window_number(values):
    """wgp.py's 1-based window counter, or None if the caller did not send one.

    The pipeline itself never reads this - it lands in generate's **kwargs and is
    ignored - but it is the only argument that distinguishes the second window of
    a video from the first window of the next queue entry.
    """
    raw = values.get("window_no")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _continues_cached_window(source_window_no):
    """Is the current window the immediate successor of the cached one?

    Returns (ok, reason).  The generation counter alone only proves the cache
    came from the previous generate() call, which a queued job's first window
    also satisfies.
    """
    current = STATE.window_no
    if current is None:
        if CONFIG.require_window_no:
            return False, ("caller did not report window_no, so a new job cannot be "
                           "told from a new window (SWL_REQUIRE_WINDOW_NO=0 to allow)")
        return True, None
    if source_window_no is None:
        if CONFIG.require_window_no:
            return False, "cached window has no window_no recorded"
        return True, None
    if current != source_window_no + 1:
        return False, (f"cache is from window {source_window_no} of its job, "
                       f"this is window {current} - different job")
    return True, None


def _plan_window(pipeline, args, kwargs):
    """Mirror the few lines of generate() that decide the target frame count.

    Returns a dict describing what this window will produce.  Everything here is
    re-validated against the captured tensor before it is ever used, so a drift
    against upstream degrades to a fall back rather than a wrong layout.
    """
    from models.minimax_h3.pipeline import _as_video, video_latent_frames  # noqa: F401
    from shared.utils.frame_scheduler import floor_frame_count, normalize_frame_count, normalize_overlap

    # Read straight from kwargs rather than binding against a signature.  Another
    # plugin may already have wrapped generate() with a (self, *args, **kwargs)
    # shim, and unless it set __wrapped__ inspect.signature would report that
    # shim's parameters instead of the pipeline's.  wgp.py passes everything by
    # keyword, so this is both simpler and more robust.
    values = {} if args else kwargs
    window_no = _window_number(values)

    def refuse(reason):
        return {"usable": False, "window_no": window_no, "reason": reason}

    if args:
        return refuse("generate() called positionally, cannot read arguments")

    if "2" in (values.get("audio_prompt_type") or ""):
        return refuse("audio-from-control-video rewrites frame_num")

    if window_no is None and CONFIG.require_window_no:
        return refuse("generate() did not report window_no, so job boundaries "
                      "cannot be established")

    prefix, overlap_error = normalize_overlap(int(values.get("prefix_frames_count") or 0), _CLIP_LENGTH, 1)
    if overlap_error:
        return refuse("invalid overlap")

    input_video = values.get("input_video")
    continuation = _as_video(input_video) if input_video is not None and prefix > 0 else None
    count = min(prefix, continuation.shape[1]) if continuation is not None and values.get("image_start") is None else 0
    if count and count < prefix:
        count = floor_frame_count(count, 1, _CLIP_LENGTH, 1)
    history_count = continuation[:, -count:-1].shape[1] if count > 1 else 0

    target_frames = int(values.get("frame_num") or 0) - history_count
    if target_frames <= 0:
        return refuse("no target frames")
    aligned = normalize_frame_count(target_frames, 5, _CLIP_LENGTH, 5)

    if aligned != target_frames:
        # The emitted video is truncated relative to the latent grid, so the last
        # cached latent no longer ends on the join frame.
        return refuse(f"target {target_frames} != aligned {aligned}")

    return {"usable": True,
            "window_no": window_no,
            # generate() only encodes continuation audio when it has both
            # continuation frames and a waveform (pipeline._waveform returns None
            # for a missing input_waveform).  Without both, the first
            # _encode_audio call of this window is something else entirely.
            "continuation_audio": bool(count) and values.get("input_waveform") is not None,
            "expected_latents": video_latent_frames(aligned)}


def _patched_generate(self, *args, **kwargs):
    with STATE.lock:
        STATE.generation += 1
    STATE.active_latents = None
    STATE.audio_encode_calls = 0
    STATE.plan = _plan_window(self, args, kwargs)
    STATE.window_no = STATE.plan.get("window_no")

    # A queue entry's first window is a different video from whatever ran last,
    # even though it is the next generate() call.  So is each repeat of a prompt,
    # which restarts wgp.py's window loop at 1.  Drop the cache here rather than
    # rely on the join fingerprint to notice: two jobs continuing the same source
    # clip, or a static shot regenerated with a new seed, can be similar enough
    # to pass a pixel comparison.
    if STATE.window_no is None:
        # With SWL_REQUIRE_WINDOW_NO=0 the caller has opted into fingerprint-only
        # matching, so leave the cache alone - dropping it here would make that
        # setting mean "never carry" rather than "carry on the old evidence".
        if CONFIG.require_window_no:
            with STATE.lock:
                STATE.invalidate("caller did not report window_no")
    elif STATE.window_no <= 1:
        with STATE.lock:
            STATE.invalidate(f"window {STATE.window_no} starts a new video")

    if not STATE.plan.get("usable"):
        _log(f"this window will not be cached: {STATE.plan.get('reason')}")
    return _ORIGINALS["generate"](self, *args, **kwargs)


# --------------------------------------------------------------------------
# 3. substitution
# --------------------------------------------------------------------------

def _take_cached(history_video):
    """Return latents to use as the history block, or None to fall back."""
    if not CONFIG.enable:
        return None, "disabled"

    with STATE.lock:
        latents = STATE.latents
        signature = STATE.signature
        signatures = STATE.signatures
        source = STATE.source_generation
        source_window_no = STATE.source_window_no
        usable = STATE.usable
        expected = STATE.expected_latents
        total = getattr(STATE, "total_latents", None)
        generation = STATE.generation

    if latents is None:
        return None, "no cached latents"
    if source != generation - 1:
        return None, f"cache is from window {source}, need {generation - 1}"
    continues, reason = _continues_cached_window(source_window_no)
    if not continues:
        return None, reason
    if not usable:
        return None, "previous window was not cacheable"
    if expected is not None and total is not None and total != expected:
        # Our reconstruction of the frame maths disagrees with what the pipeline
        # actually produced.  Refuse rather than guess.
        return None, f"latent count {total} != expected {expected}"

    height, width = history_video.shape[-2], history_video.shape[-1]
    expected_h = -(-height // _VAE_SPATIAL_FACTOR)
    expected_w = -(-width // _VAE_SPATIAL_FACTOR)
    if latents.shape[-2] != expected_h or latents.shape[-1] != expected_w:
        # Normal and expected during two-phase pass 1, which runs at a reduced
        # resolution the cached latents cannot serve.
        return None, (f"resolution mismatch (cached latent {latents.shape[-2]}x{latents.shape[-1]}, "
                      f"need {expected_h}x{expected_w} for {height}x{width})")

    wanted = CONFIG.latents or _DEFAULT_LATENTS

    # Do these latents actually depict the frames the pipeline is about to
    # re-encode?  The window-continuity check above rules out a different job;
    # this rules out the same job's window having produced something other than
    # what is being handed back, which the frame maths alone cannot see.  It is a
    # backstop now rather than the only defence, so a similar-looking pair of
    # clips no longer decides anything on its own.
    aligned, detail = _check_alignment(signatures, signature,
                                       _fingerprint(history_video[:, -1]))
    if not aligned:
        return None, detail

    if wanted % _LATENT_PERIOD != _PHASE_RESIDUE:
        # A window holds 2 + 5k latents, so only n = 2 (mod 5) starts the slice on
        # a phase-0 latent.  Any other n mislabels which latents cover one frame
        # and which cover four.
        return None, f"SWL_LATENTS={wanted} is not phase-aligned (need 7, 12, 17...)"
    if wanted > latents.shape[2]:
        return None, f"cached tail holds {latents.shape[2]} latents, need {wanted}"
    if total is not None:
        if wanted >= total:
            return None, "previous window is too short to carry from"
        if (total - wanted) % _LATENT_PERIOD != 0:
            return None, f"slice would start at latent {total - wanted}, off phase"

    return latents[:, :, -wanted:].clone(), None


def _patched_add_video_history(self, video, visual_latents, keyframes):
    """Append the history block, from cache where possible.

    Contract: upstream _add_video_history communicates through the two lists it
    is handed and returns nothing.  The carry path below therefore returns None,
    and the fallback path watches the original to confirm that stays true - if a
    Wan2GP update ever gives it a return value, carrying would start dropping it
    silently, so say so rather than let it pass.
    """
    STATE.active_latents = None
    if CONFIG.colour:
        try:
            _colour_seed(video)
        except Exception as error:
            _log(f"colour: could not read the continuation reference: {error!r}")
    latents, reason = _take_cached(video)
    if latents is None:
        STATE.stats["fell_back"] += 1
        _log(f"re-encoding history ({reason})")
        result = _ORIGINALS["add_video_history"](self, video, visual_latents, keyframes)
        if result is not None and not STATE.history_returns_value:
            STATE.history_returns_value = True
            _log(f"WARNING: stock _add_video_history returned {type(result).__name__}, "
                 f"but the carry path returns None. Upstream's contract has changed - "
                 f"the substitution needs to reproduce that value. Set SWL_ENABLE=0 "
                 f"until it does.")
        return result

    if STATE.history_returns_value:
        # Observed upstream returning something we cannot reproduce.  Fall back
        # rather than carry and drop it.
        STATE.stats["fell_back"] += 1
        _log("re-encoding history (stock _add_video_history returns a value this "
             "version cannot reproduce)")
        return _ORIGINALS["add_video_history"](self, video, visual_latents, keyframes)

    if CONFIG.colour and STATE.colour_correction is not None:
        try:
            from . import colour
            matrix, shift = STATE.colour_correction
            latents = colour.apply_torch(latents, matrix, shift)
            STATE.stats["colour_applied"] += 1
        except Exception as error:
            _log(f"colour correction failed, carrying uncorrected: {error!r}")

    if CONFIG.moment_match:
        latents = _moment_match(self, video, latents)

    visual_latents.append(latents)
    keyframes.append({"anchor": "history", "latent_frame_count": latents.shape[2]})
    STATE.active_latents = int(latents.shape[2])
    STATE.stats["engaged"] += 1
    _log(f"carried {latents.shape[2]} latents from the previous window "
         f"(coords {'corrected' if CONFIG.fix_coords else 'UNCORRECTED'})")

    if CONFIG.diagnose:
        _diagnose_statistics(self, video, latents)

    return None                      # matches stock; see the docstring above


def _moment_match(pipeline, history_video, carried):
    """Rescale carried latents per channel onto the encoded distribution.

    An affine per-channel transform, so temporal and spatial structure - the part
    that never went through decode/encode - is preserved; only the scale and
    offset the model expects to see are adopted.  Costs one native encode per
    window, which is the thing this plugin otherwise avoids, so this is a test
    instrument rather than a default.
    """
    try:
        native = pipeline._encode_video(history_video, keep_all_latents=True).to(torch.float32)
        values = carried.to(torch.float32)
        dims = [d for d in range(values.ndim) if d != 1]
        source_mean, source_std = values.mean(dim=dims, keepdim=True), values.std(dim=dims, keepdim=True)
        target_mean, target_std = native.mean(dim=dims, keepdim=True), native.std(dim=dims, keepdim=True)
        adjusted = (values - source_mean) / source_std.clamp_min(1e-5) * target_std + target_mean
        shift = float((target_mean - source_mean).abs().max())
        _log(f"moment matched carried latents (largest per-channel shift {shift:.4f})")
        return adjusted.to(dtype=carried.dtype)
    except Exception as error:
        _log(f"moment match failed, using carried latents unchanged: {error!r}")
        return carried


def _diagnose_statistics(pipeline, history_video, carried):
    """Compare carried latents against what the native re-encode would produce.

    Both live in the same normalised space, but they come from different places:
    encode_condition samples the VAE posterior, while carried latents are
    denoised model output.  A per-channel offset or scale difference between them
    would present as a colour cast at the first join.
    """
    try:
        native = pipeline._encode_video(history_video, keep_all_latents=True)
        a_full = carried.to(torch.float32)
        b = native.to(torch.float32)
        # The full carried block covers more frames than the encoded one (22 vs 17
        # at the default n=7), so comparing them wholesale mixes a distribution
        # difference with a content difference.  The last len(b) carried latents
        # cover the same 17 frames, offset by one.
        a = a_full[:, :, -b.shape[2]:] if a_full.shape[2] >= b.shape[2] else a_full
        dims = [d for d in range(a.ndim) if d != 1]          # keep the channel axis
        a_mean, a_std = a.mean(dim=dims), a.std(dim=dims)
        b_mean, b_std = b.mean(dim=dims), b.std(dim=dims)
        _log(f"diagnose: carried  mean {float(a.mean()):+.4f}  std {float(a.std()):.4f}  "
             f"({a.shape[2]} of {a_full.shape[2]} latents, length-matched)")
        _log(f"diagnose: encoded  mean {float(b.mean()):+.4f}  std {float(b.std()):.4f}  "
             f"({b.shape[2]} latents)")
        _log(f"diagnose: per-channel |mean diff| max {float((a_mean - b_mean).abs().max()):.4f} "
             f"avg {float((a_mean - b_mean).abs().mean()):.4f}")
        _log(f"diagnose: per-channel std ratio min {float((a_std / b_std.clamp_min(1e-6)).min()):.4f} "
             f"max {float((a_std / b_std.clamp_min(1e-6)).max()):.4f}")
        worst = int((a_mean - b_mean).abs().argmax())
        _log(f"diagnose: worst channel {worst}: carried {float(a_mean[worst]):+.4f}/"
             f"{float(a_std[worst]):.4f}  encoded {float(b_mean[worst]):+.4f}/{float(b_std[worst]):.4f}")
    except Exception as error:
        _log(f"diagnose failed: {error!r}")


# --------------------------------------------------------------------------
# 4. coordinate correction
# --------------------------------------------------------------------------

def _rows_per_frame(latent_height, latent_width, patch_size, target_spatial_context):
    """Ask packing for the real row count rather than assuming height*width/patch.

    _frame_grid applies target_spatial_context, so a tiled pass does not have
    (latent_height // patch_h) * (latent_width // patch_w) rows per frame.
    """
    _frame_grid = _PACKING._frame_grid
    _, patch_h, patch_w = patch_size
    grid, _ = _frame_grid(latent_height, latent_width, patch_h, patch_w, target_spatial_context)
    return int(grid.shape[0])


def _apply_origin_shift(sequence, text_len, keyframe_anchors, audio_condition_anchors,
                        rows_per_frame, video_time_scale):
    """Move everything anchored to target_origin up by one pixel frame.

    Shared by both sequence builders.  The row layout is
    [text][condition video][condition audio][target audio][target video] in each;
    Ref2VA additionally packs its reference rows inside the two condition spans.
    Those references are positioned from time_cursor, below target_origin, so
    they are left alone - as is the carried history block itself, whose internal
    spacing is already correct because the slice is phase-aligned.
    """
    _unpack_keyframe_anchor = _PACKING._unpack_keyframe_anchor

    carried = STATE.active_latents
    anchors = list(keyframe_anchors)
    if not anchors:
        return sequence
    name, count, _ = _unpack_keyframe_anchor(anchors[0])
    if name != "history" or int(count) != carried:
        _log("keyframe layout not as expected, leaving coordinates untouched")
        return sequence

    times = sequence.position_ids[:, 0]
    # NEGATIVE.  The block is one frame LATER than the layout reserves for it,
    # and a block's position is reported as (block_absolute - target_origin),
    # so target_origin must move EARLIER to close the gap.  Raising it instead
    # widens the skew from one frame to two.  tests/test_layout_contract.py
    # pins this down.
    delta = -_RESERVE_DEFICIT * _FRAME_RESCALE * float(video_time_scale)

    keyframe_frames = sum(_unpack_keyframe_anchor(entry)[1] for entry in anchors)
    block_stop = text_len + carried * rows_per_frame
    keyframe_stop = text_len + keyframe_frames * rows_per_frame
    times[block_stop:keyframe_stop] += delta        # "first" / "last" / "frame" keyframes

    cursor = keyframe_stop
    for entry in audio_condition_anchors:
        anchor, length = entry if isinstance(entry, tuple) else (entry, 1)
        stop = cursor + length * _AUDIO_CHANNELS
        if anchor != "history":
            times[cursor:stop] += delta
        cursor = stop

    target_start = (text_len + sequence.num_condition_video_rows
                    + sequence.num_condition_audio_rows)
    times[target_start:] += delta                   # target audio + target video
    return sequence


def _patched_build_packed_sequence(text_token_tags, num_latent_frames, latent_height, latent_width,
                                   num_audio_latents, patch_size, keyframe_anchors=(),
                                   video_time_scale=1.0, audio_condition_anchors=(),
                                   target_spatial_context=None, **kwargs):
    sequence = _ORIGINALS["build_packed_sequence"](
        text_token_tags, num_latent_frames, latent_height, latent_width, num_audio_latents,
        patch_size, keyframe_anchors, video_time_scale,
        audio_condition_anchors=audio_condition_anchors,
        target_spatial_context=target_spatial_context, **kwargs)

    if STATE.active_latents is None or not CONFIG.fix_coords:
        return sequence
    return _apply_origin_shift(
        sequence, int(text_token_tags.shape[0]), keyframe_anchors, audio_condition_anchors,
        _rows_per_frame(latent_height, latent_width, patch_size, target_spatial_context),
        video_time_scale)


def _patched_build_ref2va_packed_sequence(text_token_tags, references, num_latent_frames,
                                          latent_height, latent_width, num_audio_latents,
                                          patch_size, video_time_scale=1.0, keyframe_anchors=(),
                                          audio_condition_anchors=(),
                                          target_spatial_context=None, **kwargs):
    sequence = _ORIGINALS["build_ref2va_packed_sequence"](
        text_token_tags, references, num_latent_frames, latent_height, latent_width,
        num_audio_latents, patch_size, video_time_scale, keyframe_anchors=keyframe_anchors,
        audio_condition_anchors=audio_condition_anchors,
        target_spatial_context=target_spatial_context, **kwargs)

    if STATE.active_latents is None or not CONFIG.fix_coords:
        return sequence
    return _apply_origin_shift(
        sequence, int(text_token_tags.shape[0]), keyframe_anchors, audio_condition_anchors,
        _rows_per_frame(latent_height, latent_width, patch_size, target_spatial_context),
        video_time_scale)


# --------------------------------------------------------------------------
# install / uninstall
# --------------------------------------------------------------------------

_EXPECTED_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_EXPECTED_FRAME_RESCALE = 5.0 / 3.0

# Patch targets.  (key, module-or-class getter, attribute, required).  Optional
# targets are skipped when absent so an older or newer Wan2GP loses one
# capability instead of the whole plugin.
_TARGETS = (
    ("decode", "video_vae", "decode", True),
    ("generate", "pipeline", "generate", True),
    ("add_video_history", "pipeline", "_add_video_history", True),
    # transformer.py imports the packing helpers by name, so the binding that
    # matters lives in the transformer module namespace, not in packing.
    ("build_packed_sequence", "transformer", "build_packed_sequence", True),
    ("build_ref2va_packed_sequence", "transformer", "build_ref2va_packed_sequence", False),
    ("audio_decode", "audio_vae", "decode", False),
    ("encode_audio", "pipeline", "_encode_audio", False),
)

_REPLACEMENTS = {
    "decode": "_patched_decode",
    "generate": "_patched_generate",
    "add_video_history": "_patched_add_video_history",
    "build_packed_sequence": "_patched_build_packed_sequence",
    "build_ref2va_packed_sequence": "_patched_build_ref2va_packed_sequence",
    "audio_decode": "_patched_audio_decode",
    "encode_audio": "_patched_encode_audio",
}


def _import_packing():
    """The packing module, wherever this Wan2GP keeps it."""
    last = None
    for path in ("models.minimax_h3.components.packing", "models.minimax_h3.packing"):
        try:
            return __import__(path, fromlist=["packing"]), None
        except Exception as error:
            last = error
    return None, last


def _preflight():
    """Check every assumption before touching anything.

    The plugin rewrites seven bindings and reproduces a coordinate layout from
    Wan2GP's own constants.  If a future or older release moves a binding or
    changes the frame grid, patching regardless would not fail loudly - it would
    produce a plausible video with the history in the wrong place.  So each
    assumption is verified up front, and anything unverifiable makes the plugin
    stay inert and say why.

    Returns (owners, available, problems).
    """
    problems = []
    try:
        from models.minimax_h3 import transformer
        from models.minimax_h3.pipeline import MiniMaxH3Pipeline
        from models.minimax_h3.video_vae import MiniMaxH3VideoVAE
    except Exception as error:
        return None, None, [f"MiniMax H3 is not available ({error!r})"]

    owners = {"transformer": transformer, "pipeline": MiniMaxH3Pipeline,
              "video_vae": MiniMaxH3VideoVAE}
    try:
        from models.minimax_h3.audio_vae import MiniMaxH3AudioVAE
        owners["audio_vae"] = MiniMaxH3AudioVAE
    except Exception:
        pass

    # The frame grid the coordinate maths and the phase rule are built on.
    global _PACKING
    packing, error = _import_packing()
    _PACKING = packing
    if packing is None:
        problems.append(f"cannot import the H3 packing module ({error!r})")
    else:
        for helper in ("_unpack_keyframe_anchor", "_frame_grid", "_video_t_grid",
                       "_reference_t_span"):
            if not callable(getattr(packing, helper, None)):
                problems.append(f"packing.{helper} is missing")
        grid = tuple(getattr(packing, "_FRAME_PER_TOKEN", ()))
        if grid != _EXPECTED_FRAME_PER_TOKEN:
            problems.append(f"_FRAME_PER_TOKEN is {grid}, expected "
                            f"{_EXPECTED_FRAME_PER_TOKEN}; the phase rule and the "
                            f"coordinate correction would both be wrong")
        rescale = float(getattr(packing, "_FRAME_RESCALE", 0.0))
        if abs(rescale - _EXPECTED_FRAME_RESCALE) > 1e-9:
            problems.append(f"_FRAME_RESCALE is {rescale}, expected "
                            f"{_EXPECTED_FRAME_RESCALE}")
        channels = getattr(packing, "MINIMAX_H3_AUDIO_CHANNELS", None)
        if channels != 2:
            problems.append(f"MINIMAX_H3_AUDIO_CHANNELS is {channels}, expected 2")

    if not callable(getattr(MiniMaxH3Pipeline, "video_latent_frames", None)):
        problems.append("MiniMaxH3Pipeline.video_latent_frames is missing; the "
                        "latent count cannot be checked against expectation")

    available = {}
    for key, owner_name, attribute, required in _TARGETS:
        owner = owners.get(owner_name)
        if owner is not None and getattr(owner, attribute, None) is not None:
            available[key] = (owner, attribute)
        elif required:
            problems.append(f"{owner_name}.{attribute} is missing")
    return owners, available, problems


def install():
    if _ORIGINALS:
        return True

    owners, available, problems = _preflight()
    if problems:
        _log("staying inert, nothing patched:")
        for problem in problems:
            _log(f"  - {problem}")
        _log("  this build of Wan2GP is not one this plugin can patch safely")
        return False

    if "audio_decode" not in available or "encode_audio" not in available:
        CONFIG.audio = False
        _log("audio bindings absent, carrying video latents only")
    if "build_ref2va_packed_sequence" not in available:
        _log("Ref2VA builder absent, plain sliding windows only")

    # Apply as a unit: any failure rolls the whole thing back rather than
    # leaving some bindings patched and others not.
    applied = []
    try:
        for key, (owner, attribute) in available.items():
            _ORIGINALS[key] = getattr(owner, attribute)
            setattr(owner, attribute, globals()[_REPLACEMENTS[key]])
            applied.append((owner, attribute, key))
    except Exception as error:
        for owner, attribute, key in reversed(applied):
            setattr(owner, attribute, _ORIGINALS.pop(key))
        _ORIGINALS.clear()
        _log(f"install failed and was rolled back, nothing patched: {error!r}")
        return False

    _ORIGINAL_OWNERS.update({key: (owner, attribute)
                             for key, (owner, attribute) in available.items()})
    _log(f"installed v{VERSION} (enable={CONFIG.enable}, fix_coords={CONFIG.fix_coords}, "
         f"latents={CONFIG.latents or _DEFAULT_LATENTS}, audio={CONFIG.audio}, "
         f"colour={CONFIG.colour}, diagnose={CONFIG.diagnose}, "
         f"moment_match={CONFIG.moment_match})")
    return True


def uninstall():
    if not _ORIGINALS:
        return
    for key, original in list(_ORIGINALS.items()):
        owner, attribute = _ORIGINAL_OWNERS.get(key, (None, None))
        if owner is None:
            continue
        try:
            setattr(owner, attribute, original)
        except Exception as error:
            _log(f"could not restore {attribute}: {error!r}")
    _ORIGINALS.clear()
    _ORIGINAL_OWNERS.clear()
    STATE.invalidate("uninstalled")
    _log("removed")
