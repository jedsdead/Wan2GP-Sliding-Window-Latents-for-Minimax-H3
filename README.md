# Sliding Window Latents

A Wan2GP (WanGP) extension plugin that carries MiniMax H3 latents across
sliding windows instead of re-encoding the previous window's decoded pixels,
removing one VAE decode/encode round trip per window.

Version 1.0.0 · MIT · built and verified against Wan2GP `362c346` (7 Sep 2026)

Video and audio latent carry are on by default. Colour consistency is
experimental and off by default. Do not run this alongside the Sliding Window
Anchor plugin — see below.

## Requirements

Nothing beyond Wan2GP itself: `torch` and `numpy` are already dependencies.
The tools in `tools/` additionally need `ffmpeg` and `ffprobe` on `PATH`, which
Wan2GP also requires. The tests need only `numpy` — no torch, no GPU:

```
python tests/test_layout_contract.py
python tests/test_alignment.py
python tests/test_colour_latents.py
python tests/test_no_tensor_truthiness.py
```

## Credit

Independently written against Wan2GP's own internals, but the idea came from
[ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)
by NikoDemon80, which does the equivalent job inside ComfyUI. No code is shared
between the two projects — they patch different applications with different
internals, and the only lines they have in common are `import numpy as np` and
similar boilerplate. Two ideas in particular are owed to it: keeping the carried
context in the latent domain rather than round-tripping it through pixels, and
measuring the join rather than trusting it by ear.

## What it changes

Wan2GP already pins a run of previous-window frames as history conditioning. At
overlap 18 `MiniMaxH3Pipeline.generate` takes 17 pixel frames, encodes them in
one VAE call with `keep_all_latents=True`, and submits them as an `"history"`
keyframe block. Nothing about that mechanism needs adding.

What it does mean is that every window feeds the next one *decoded* pixels, which
are then re-encoded — one extra decode/encode round trip per window, applied to
the context the whole join depends on. This plugin skips it: the previous
window's latents are captured on their way into the decoder and handed straight
back as the history block.

The latent spaces line up without conversion. `encode_condition` returns
`self._normalize(latents)` and `decode` opens by undoing exactly that, so the
tensor at the decode boundary is already in the space `_add_video_history`
produces.

## Install

```
cp -r wan2gp-sliding-window-latents /path/to/Wan2GP/plugins/
```

Then enable it in the **Plugins** tab, save, and restart Wan2GP.

The folder name becomes the package name — `PluginManager.load_plugins_from_directory`
does `importlib.import_module(f"{plugin_dir_name}.plugin")` — so rename it only
if you also rename the entry in the enabled-plugins list.

```
wan2gp-sliding-window-latents/
├── plugin.py           entry point: Gradio panel, settings persistence
├── patches.py          the seven patched bindings; all the real work
├── plugin_info.json    metadata; overrides what plugin.py sets on itself
├── __init__.py         package marker (empty)
├── README.md
├── colour.py           latent-space colour correction
├── LICENSE
├── CHANGELOG.md
├── tests/
│   ├── test_layout_contract.py   grid arithmetic; no Wan2GP, no torch, no GPU
│   ├── test_alignment.py         join-alignment search
│   ├── test_colour_latents.py    colour maths and an eight-window simulation
│   └── test_no_tensor_truthiness.py  AST lint; tensors in boolean contexts
└── tools/
    ├── measure_tone.py           per-frame tone statistics
    └── measure_joins.py          join steps, chain drift, join continuity
```

`state/settings.json` is written next to `plugin.py` the first time the panel is
used, and takes precedence over the environment variables below.

## Settings

Environment variables, read once when the plugin loads.

| Variable | Default | Meaning |
|---|---|---|
| `SWL_ENABLE` | `1` | master switch |
| `SWL_FIX_COORDS` | `1` | apply the coordinate correction below |
| `SWL_REQUIRE_WINDOW_NO` | `1` | refuse to carry when the caller does not report `window_no` |
| `SWL_LATENTS` | `0` | carried latent count; `0` uses 7. Must be ≡ 2 (mod 5): 7, 12, 17… |
| `SWL_VERBOSE` | `1` | log every engage and every fall back |
| `SWL_MATCH_TOL` | `0.03` | how closely the cached join frame must match the incoming history |
| `SWL_ALIGN_SEARCH` | `1` | search the decoded tail for the join frame instead of testing one offset |
| `SWL_AUDIO` | `1` | also carry audio latents |
| `SWL_AUDIO_TOL` | `0.15` | how closely the cached audio envelope must match |
| `SWL_DIAGNOSE` | `0` | also run the native encode and log how the two latent sets differ |
| `SWL_COLOUR` | `0` | correct grade drift on the carried latents |
| `SWL_COLOUR_AXES` | `brightness,cast` | which of `brightness`, `contrast`, `saturation`, `cast` to correct |
| `SWL_COLOUR_SCOPE` | `latents` | `latents` corrects only the conditioning; `both` also rewrites the window |
| `SWL_COLOUR_MATCH` | `previous` | `previous` window or `first` window as the reference |
| `SWL_COLOUR_STRENGTH` | `1.0` | how much of the measured difference to remove |
| `SWL_COLOUR_SCENE` | `0.06` | past this, a difference is read as a cut and the window is skipped |
| `SWL_COLOUR_MAX` | `0.15` | ceiling on the accumulated correction |

Watch the console. The plugin says which path it took on every window, and it
falls back to stock behaviour rather than guessing whenever anything looks off.

## Audio

H3 does not hold one joint video+audio latent. There are two autoencoders and two
tensors — `self.vae` and `self.audio_vae`, feeding separate `visual_latents` and
`audio_latents` lists. What is joint is the *token sequence*: both are packed into
one attention sequence over a shared time axis, which is why the same
continuation trick works on each.

Stock Wan2GP re-encodes the overlap waveform every window
(`continuation_audio = self._encode_audio(continuation_waveform)`), then splits it
into a `"history"` chunk and a `"first"` chunk placed at the target's own origin.
The plugin replaces that encode with the previous window's own audio latents.

The audio side needs no coordinate work. `_fill_audio_condition_positions` lays
conditions out as `origin + arange(length)` — a uniform integer axis with none of
the `(1, 4, 4, 4, 4)` periodicity of the video grid — so a tail slice needs
neither phase alignment nor translation.

The real encode still runs on every window. It fixes the latent count the
pipeline's history/boundary split is derived from, and it is the fallback if any
check fails. That wastes one audio VAE encode per window, which is cheap next to
the video one, and it means a bad substitution degrades to stock behaviour rather
than a shape error mid-render.

Identifying *which* encode to replace takes two conditions, not one.
`_encode_audio` is called for reference clips (`_add_audio_reference`) and for
the target audio condition as well as for the continuation, so the plugin
substitutes only into the first call of a window **and** only when the window has
a continuation encode to make — `pipeline.generate` runs that one inside
`if continuation_count:` and skips it when `input_waveform` is None, in which case
call one is something else entirely and must be left alone.

Matching uses a peak-normalised RMS envelope of the last half second, 32 bins,
compared against the previous window's decoded audio tail. Silence matches
silence explicitly rather than dividing by a near-zero peak — which is precisely
why the envelope is a supporting check and the window-number continuity in *Job
boundaries* is the one that decides.

## Colour consistency

Off by default. `colour.py`.

Because every window is conditioned on the end of the one before it, the
model's reconstruction error is inherited and compounds: blacks lift,
contrast softens, colour desaturates, and a long generation slowly washes
out. Latent carry removes the VAE round trip's share of that error but not
the model's own re-synthesis error, so the drift shrinks rather than
disappears.

Correcting decoded pixels is not compatible with carrying latents. The
correction happens after `decode`; the latents are captured during it. A
pixel correction would never reach the conditioning, and the loop that makes
pixel correction converge in stock Wan2GP - `pre_video_guide` is sliced out
of the *corrected* sample and re-encoded - is exactly the loop latent carry
removes.

So the correction is applied to the carried latents instead.

### How

Wan2GP ships a linear latent-to-RGB map for H3 previews
(`shared/RGB_factors.py`). Signed RGB is `latent @ M + bias`, with M 24x3,
full rank and well conditioned (singular values 0.847, 0.381, 0.126). Since
the map is linear, an affine colour correction in RGB collapses to an affine
in latent space:

```
L' = (I + P^T D M^T) L + P^T [D (bias + 1) + 2 o]        D = Q - I,  P = pinv(M)
```

One 24x24 matrix and a 24-vector, built once per window and applied with a
single einsum. The same collapse a pixel implementation makes from Y'CbCr
into a 3x3, carried one step further down. Cost does not scale with
resolution the way a VAE pass does.

Measurement is in Rec. 709 Y'CbCr so the axes stay separable - luma mean is
brightness, its spread is contrast, chroma means are cast, their shared
spread is saturation - with one scale for both chroma channels, since
scaling them separately shifts hue while claiming to change only saturation.

### Corrections accumulate

The running total is carried forward and composed, not replaced. This is not
a refinement; without it the feature half works.

Correcting the latents does not rewrite the video, so the reference a window
is matched against - the previous window's tail - is itself uncorrected and
keeps moving. A per-window correction cancels one window's drift, measures
nothing the next, and lets the following one through. In the eight-window
simulation in `tests/test_colour_latents.py` that stair-step gives back
roughly half the drift:

```
uncorrected      0.5564 0.5616 0.5668 0.5719 0.5771 0.5822 0.5873 0.5923
per window       0.5564 0.5616 0.5616 0.5668 0.5668 0.5719 0.5719 0.5771
accumulated      0.5564 0.5616 0.5616 0.5616 0.5616 0.5616 0.5616 0.5616
```

The residual first step is the one-window lag: nothing can be measured until
a second window exists. The total is clamped rather than the step, since it
is the total that reaches the model.

### Is it doing anything?

Watch the console. Each window logs either the correction it applied, or
`measured drift is inside the noise floor, no correction`. If it reports the
noise floor on window after window, latent carry has already removed enough of
the drift that there is nothing left to correct, and the feature is not earning
its place on that material - turn it off.

A line beginning `BUG:` is not a fallback and should not be read as one.

### Guards

Same shape as the measurement needs, and each one discards rather than caps.

- **Scene change** — past `SWL_COLOUR_SCENE` a difference is read as a cut,
  not drift, and the window is left alone. Capping and applying it anyway
  would drag a new scene bodily toward the grade of the old one.
- **Noise floor** — the same statistics are measured between two interior
  groups of the *same* window, where there is no join and the true answer is
  no change. Anything at the join not clearly larger than that wobble is held
  back. This matters because each window inherits the last, so chasing noise
  does not average out over a long generation; it accumulates into exactly
  the drift the correction exists to prevent.
- **Axis toggles** — brightness and cast are offsets and rest on the linear
  map lightly. Contrast and saturation are gains, lean on the approximation
  harder, and are off by default.

### Scope

**`latents`** (default) corrects only the conditioning. It stops drift
compounding into later windows without altering frames already written, and
it never rewrites anything Wan2GP saves.

**`both`** also corrects the window itself, and applies the same step to the
latents on their way into the cache so the two cannot drift apart. That
sync is the whole reason this is possible: correcting pixels alone would
leave the conditioning describing a grade the video no longer has, which is
why pixel-domain correction and latent carry are otherwise incompatible.
With the window rewritten, the reference the next window measures against is
corrected too, so the loop closes by itself and no running total is carried.

`both` is what fixes a join you can already see. `latents` only stops the
next one getting worse.

### Continuing a video

A Continue Video run hands its first window a strip of frames from the end
of the video being continued. Those frames are the source video's own grade,
which is exactly what the new window should come out looking like - so they
become the colour reference for that window.

Without this, the first window of a continuation is the one window that
begins with nothing to match against, across the join between source and
new video, which is the most visible join there is. It is also the join
where a shift is hardest to hide: a cut disguises it, but continuing the
same shot does not.

This is the case where scope matters most, and it is free of the usual
trade-off. The plugin drops its latent cache at `window_no <= 1`, so a
continuation's first window re-encodes rather than carrying - which means
there are no cached latents to fall out of step with a pixel correction
there. `both` scope on that window costs nothing.

The reference span is wider here (48 frames rather than 5). Both sides of a
continuation join sit in the same call, so more frames can be measured
without the previous window's having been discarded.

In the simulation in `tests/test_colour_latents.py`, a first window that
steps 0.0044 in luma away from the source lands within 9e-16 of it once
seeded, and stays there:

```
source video luma: 0.5564
no reference:      0.5608 0.5652 0.5695 0.5738
seeded reference:  0.5564 0.5564 0.5564 0.5564
```

One caveat on diagnosis. A shift at a continuation join is not always
reconstruction drift. If the source video was written with a different
level convention - limited range against full range, blacks at 16 rather
than 0 - the step is a systematic levels mismatch and no amount of drift
correction is the right fix. Drift is a percent or two and grows window on
window; a range mismatch is larger, appears entirely at the first join, and
does not compound. `tools/measure_joins.py` separates the two.

### What it does not do

In `latents` scope it does not touch the saved video, so nothing there
repairs frames already written. `both` scope does.

The linear map is a preview approximation of the real decoder, not the
decoder. `tests/test_colour_latents.py` proves the algebra is exact to 4e-16
against that map; it does not and cannot prove the map matches the decoder.
Measure with `tools/measure_joins.py` before trusting the gain axes.

## The coordinate correction

This is the part that isn't a straight substitution, and it's the part most
likely to need revisiting.

H3 compresses time periodically. `pipeline.video_latent_frames` is
`2 + ((N-5)//17)*5` — seventeen pixel frames per five latents — and
`_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)` in `components/packing.py` is that same
period. Latent *i* covers `_FRAME_PER_TOKEN[i % 5]` frames, so every fifth latent
covers one frame and the rest cover four, throughout the window.

`_video_t_grid` walks that pattern from position 0, which means a carried block
is laid out correctly **if and only if its first latent sits at an absolute index
divisible by 5**. A window always holds `2 + 5k` latents, so slicing the last *n*
is phase-aligned exactly when `n ≡ 2 (mod 5)`. The plugin defaults to `n = 7`
(22 pixel frames, slightly more context than the 17 the native re-encode gives);
`12` and `17` also work. Anything else is refused rather than silently mislabelled.

With the phase right, the internal spacing needs no correction. What remains is a
pure translation: the native block stops one frame short of the join
(`history_frames` excludes it) while a carried block's last latent contains the
join frame, so `reference_t_span` places the whole block one pixel frame early.

```
n=7   layout  -22  -21  -17  -13   -9   -5   -4
      content -21  -20  -16  -12   -8   -4   -3
              spacing identical, translation -1 frame (42 ms at 24fps)
```

The fix moves everything anchored to `target_origin` **earlier** by that single
frame and leaves the block untouched. The direction matters: a block's position
is reported as `block_absolute - target_origin`, so raising `target_origin`
widens the skew to two frames instead of closing it. `tests/test_layout_contract.py`
pins this down and runs in a second without Wan2GP, torch or a GPU:

```
python tests/test_layout_contract.py
```

Note that once corrected, the carried block's final latent reaches target frame
0, where `_add_image_condition` has already pinned the join frame with a `"first"`
anchor. Both describe the same instant — one sliced from the latent, one
re-encoded from the decoded pixel — so they reinforce rather than contradict. It
cannot be sliced away in any case: the last latent of a window always ends on
that window's final frame, and latents are indivisible.

Caveat: `transformer.py` imports the packing helpers **by name**, so the patch
rebinds `models.minimax_h3.transformer.build_packed_sequence`. That's the
fragile patch of the seven — it's the one to check first after a Wan2GP update.

## Repeated content at a join

If the end of one window reappears inside the next, the cause is almost always
that the cached latents do not end where the emitted video ends.

The next window hands back `continuation[:, -overlap:-1]`, so its last frame is
the previous window's decoded frame **-2** — the join frame itself is excluded
and pinned separately by a `"first"` anchor. If the cached latents extend past
that point, the carried block still gets placed as though it ended at the join,
so the model receives footage from *after* the join as though it came before,
and renders it a second time. That is a repeated shot rather than a stutter,
which is what distinguishes it from a coordinate skew: the coordinate
correction is worth one frame, and one frame cannot repeat a shot.

Testing a single offset cannot detect this. Neighbouring frames of a slow shot
differ by far less than `SWL_MATCH_TOL` — in `tests/test_alignment.py` a
four-frame overrun measures 0.004 against a 0.03 tolerance — so the old check
passed and said nothing. The plugin now fingerprints a run of tail frames at
capture and searches for the join at use, reporting the skew in frames:

```
[sliding-window-latents] tail misaligned by -4 frame(s) - cached latents run
past the end of the emitted video (join matches offset -6 at 0.0021, expected
-2 at 0.0040)
```

A negative skew means the cache holds frames from after the join, which is the
direction that repeats. Only `+1` is reachable upward, since the cached
fingerprints stop at offset -1.

When the tail is flat enough that no offset wins by a decisive margin, the
search abstains and the tolerance decides as before, so a static shot is not
refused on noise. `SWL_ALIGN_SEARCH=0` restores the old single-offset
behaviour.

Two things to check if you see this reported, since both displace the tail
without the plugin being able to see it from inside the pipeline:

- **Discard Last Frames of a Window** set above 0. `wgp.py` trims `sample`
  before taking `pre_video_guide`, so the emitted end moves but the latents
  handed to `decode` do not.
- Any post-decode processing that shortens the window.

## Job boundaries

One `generate()` call is one window, but two consecutive `generate()` calls are
not necessarily two consecutive windows of the same video. Wan2GP runs queued
jobs back to back in the same process, and a job set to produce several samples
restarts its window loop for each one, so "the previous generate call" is
routinely the last window of an unrelated video.

`wgp.py` passes `window_no` on every call. The pipeline itself never reads it —
it lands in `generate`'s `**kwargs` — but it restarts at 1 for each new job *and*
each repeat within a job, which is exactly the boundary that matters. The plugin
drops its cache whenever it sees `window_no <= 1`, and both carry paths require
the cached window to be numbered one lower than the window asking for it.

The join fingerprint is not enough on its own here, which is why this is a
separate check rather than a tolerance tweak. Two queued jobs continuing the same
source clip, or the same static shot regenerated with a new seed, can sit well
inside `SWL_MATCH_TOL`. The audio side is worse: silence matches silence by
design, so a silent job following a silent job would otherwise inherit the
previous video's audio latents at its first window.

Consequence worth knowing: running **Continue Video** on the job that just
finished re-encodes at that first window rather than carrying. That is correct —
the file on disk has been through trimming, colour correction and possibly
interpolation since those latents were produced.

If `window_no` is absent the caller is not `wgp.py`'s window loop and the
boundary cannot be established, so the plugin stays inert and says so. Set
`SWL_REQUIRE_WINDOW_NO=0` to fall back to fingerprint-only matching for a front
end that drives the pipeline itself.

## When it declines to engage

All of these fall back to stock re-encoding and say so:

- **Two-phase pass 1** — `prepare_keyframes` runs at reduced resolution there;
  cached latents only exist at final resolution. Pass 2 engages normally. Tiled
  pass 2 falls back on the same check.
- **Audio-from-control-video** — rewrites `frame_num` after the point the plugin
  reconstructs it.
- **`target_frames != aligned_target_frames`** — the emitted video is truncated
  relative to the latent grid, so the last cached latent no longer ends on the
  join frame.
- **Latent count disagrees with expectation** — the plugin asks
  `pipeline.video_latent_frames` what the window should have produced and refuses
  if the captured tensor disagrees. This guard already caught one real bug: the
  first draft assumed `1 + (N-1)//4` latents and was corrected by a
  `latent count 52 != expected 44` fallback rather than by producing a wrong
  layout.
- **First window of a job** — `window_no <= 1`. See *Job boundaries*.
- **Window numbers not consecutive** — the cached window is not numbered exactly
  one lower than the window asking for it, or the caller reported no `window_no`
  at all and `SWL_REQUIRE_WINDOW_NO` is on.
- **Tail misaligned** — the join frame is not where it must be in the cached
  decoded tail. See *Repeated content at a join* below. The log reports the
  measured skew in frames.
- **Content mismatch** — the cached latents don't depict the frames being
  replaced. The check compares a 16x16 pooled greyscale signature of the join
  frame, so it is independent of resolution and of whether the tensors are uint8
  or float. It also catches the case where `wgp.py` trimmed the decoded sample
  (`discard_last_frames`, `automatic_trim_last_frames`) between capture and use.
- **Slice off phase** — `SWL_LATENTS` not ≡ 2 (mod 5), or a previous window too
  short to give a phase-aligned slice.

## A/B protocol

Three runs, same seed, same prompt, overlap 18:

1. `SWL_ENABLE=0` — stock Wan2GP.
2. `SWL_ENABLE=1 SWL_FIX_COORDS=0` — latent carry, upstream layout. Isolates
   the round trip and shows what the 1-frame skew costs.
3. `SWL_ENABLE=1 SWL_FIX_COORDS=1` — both.

Then:

```
python tools/measure_joins.py run1.mp4 --baseline run3.mp4 --window 362 --overlap 18
python tools/measure_tone.py run3.mp4 --window 362 --overlap 18
```

Read the **chain** slopes first. Wan2GP re-emits each window's carried frames
verbatim, so the frames either side of a join are the same content with exactly
one generation pass in between. If the picture-detail and audio HF slopes flatten
between run 1 and run 3, the round trip was the mechanism and this plugin is the
fix. If the joins are already flat in run 1 and the drift is spread evenly
through each window, the cause is elsewhere — the sampler, the prompt, or the
model — and latent carry won't touch it.

Then read **continuity**. Run 3 should sit closer to 1.00 than run 2; if it
doesn't, the one-frame premise behind the coordinate correction is wrong and the
frame-0 overlap noted above is the next thing to look at.

One confound to keep in mind when reading runs 2 and 3: `encode_condition`
samples noise with a fixed seed, adds `std * noise`, and rounds through float16
before normalising. Carried latents skip all of that, so the history block is
strictly cleaner than the native one. The comparison isn't isolating the round
trip alone.

## Ref2VA

Supported from v0.6. `build_ref2va_packed_sequence` uses the same anchor
semantics as the plain builder — `"history"` immediately precedes
`target_origin` — but takes its origin from a `time_cursor` positioned after the
reference blocks rather than straight after the text. Both builders share the
row layout `[text][condition video][condition audio][target audio][target
video]`, with Ref2VA packing its reference rows inside the two condition spans.
Those references are positioned below `target_origin` and are left untouched, so
the same one-frame shift applies to both.

Row counts now come from packing's own `_frame_grid` rather than
`(latent_height // patch_h) * (latent_width // patch_w)`, which was wrong
whenever `target_spatial_context` was set — a latent bug in the plain builder
path too, not just Ref2VA.

## Do not run this with Sliding Window Anchor

The Sliding Window Anchor plugin injects the previous window's final frame as a
`"frame"` keyframe at `frame_index = history_count`, which `pipeline.py` turns
into `frame_index = 0` - the same time coordinate as `target_origin`. Wan2GP
already pins that frame there itself with a `"first"` anchor, and the carried
history block reaches it too. Two or three condition blocks all asserting one
instant makes the model hold on it, which is the frozen frame at the join.

Changing the carried latent count does not avoid it. The final latent of a
window always ends on that window's last frame, so the block reaches target
frame 0 at every setting:

```
  n   frames   last latent covers   reaches frame 0
  7       22               -3..+0              True
 12       39               -3..+0              True
 17       56               -3..+0              True
```

`n` changes how far back the block reaches, never where it ends.

The anchor's frame injection has been redundant since Wan2GP added its own
continuation pin (commit `5c8b4ac`, 8 August 2026, which introduced both
`_add_video_history` and the `continuation[:, -1:]` pin together). Its colour
matching is the part that is not redundant - but that is pixel-domain and
cannot be combined with latent carry, which is why this plugin has its own
latent-domain version.

## Compatibility

Built and verified against Wan2GP `362c346`. It patches seven bindings and
reproduces a coordinate layout from Wan2GP's own constants, so rather than
claim it works everywhere, `install()` checks first and refuses when it cannot
be sure:

- every required binding must exist: `MiniMaxH3VideoVAE.decode`,
  `MiniMaxH3Pipeline.generate`, `MiniMaxH3Pipeline._add_video_history`,
  `transformer.build_packed_sequence`, `MiniMaxH3Pipeline.video_latent_frames`
- the packing helpers `_unpack_keyframe_anchor`, `_frame_grid`, `_video_t_grid`
  and `_reference_t_span` must be present, at either
  `models.minimax_h3.components.packing` or `models.minimax_h3.packing`
- `_FRAME_PER_TOKEN` must be `(1, 4, 4, 4, 4)`, `_FRAME_RESCALE` must be 5/3 and
  `MINIMAX_H3_AUDIO_CHANNELS` must be 2 - the phase rule and the coordinate
  correction are derived from all three

If any of those fail, nothing is patched and the console lists what was wrong.
Two bindings are optional and degrade rather than abort: without the audio VAE
and `_encode_audio` it carries video latents only, and without
`build_ref2va_packed_sequence` it supports plain sliding windows only.

Patching is applied as a unit. If any binding fails to replace, the ones
already replaced are restored before returning.

This cannot guarantee correct output on a version it has never seen. It does
guarantee that such a version gets stock behaviour and a clear message instead
of a plausible-looking video with the history in the wrong place.

## Reverting

Disable in the Plugins tab and restart. `patches.uninstall()` restores all seven
originals if you'd rather do it from a console.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Licence

MIT. See [LICENSE](LICENSE).
