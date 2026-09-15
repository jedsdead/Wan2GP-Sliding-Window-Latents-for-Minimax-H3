# Changelog

## 1.2
- Added `SWL_VIDEO`, with a matching panel control. `SWL_VIDEO=0` with
  `SWL_AUDIO=1` carries audio only, leaving Wan2GP's pixel re-encode in place
  for the video history. The two carries were always independent - video
  substitution happens in `_add_video_history`, audio in `_encode_audio` - but
  there was no way to ask for one without the other. Audio-only needs no
  coordinate correction at all, since audio conditions are laid on a uniform
  integer axis, so it is the plugin's least invasive configuration.
  `SWL_AUDIO_CONTEXT` is unavailable in this mode and says so: the compensating
  layout shift lives in the video carry.
- A deliberate audio-only run no longer counts as a fall back, so that counter
  keeps meaning "wanted to carry and could not".
- Added a guard for trimmed windows. `wgp.py` sets
  `image_end_frame_position = current_video_length - tail_trim_frames - 1`, so
  an end image pinned short of the last generated frame reveals a tail trim -
  the one place a trim is visible from inside `generate()`. Such a window
  generates more frames than it emits, so its cached tail would end past the
  video the next window hands back. `_check_alignment` would usually catch that
  as a negative skew, but a window pinned to an end image tends to settle onto a
  held composition, and on a flat tail the search abstains by design and leaves
  the decision to the tolerance, which a static shot passes.
- Added a guard for anything conditioned on target frame 0. Wan2GP pins the join
  there with a `"first"` anchor and the corrected carried block reaches it too;
  a third block asserting that instant makes the model hold on it and the join
  gains a frame that does not move. `pipeline.py:802` subtracts `history_count`
  from every injected position, so a raw position of exactly `history_count`
  lands on frame 0 - which is where Sliding Window Anchor puts its anchor
  (`_history_count` returns 17 at overlap 18 and `_inject` appends it raw). It
  blocks the carry for that window, not the cache: the window's own output is
  still sound. Keyed to the collision rather than to a plugin name.
- Wan2GP's own `L` frame injection does **not** reach frame 0 and needs no
  guard. On FL2VA and Ref2VA `extract_guide_from_window_start` is False, so a
  frame's relative position resolves to `abs_pos - window_start_frame` and the
  smallest value reachable is `reuse_frames`, giving a minimum `frame_index` of
  1. An `L` frame also sits one index below the next window's slice, so it is
  not handed forward at all. Latents carry normally alongside `L` frames;
  `tests/test_carry_modes.py` pins the arithmetic.
- The end-image position is now classified rather than subtracted blindly. A
  plugin can set `image_end_frame_position` to anything, and reading a position
  on the opening frame as a tail trim would have produced a large, confident and
  wrong number.
- An end image that is *not* short of the last frame changes nothing and is not
  refused: it arrives as a `"frame"` anchor positioned relative to
  `target_origin`, while the correction moves the carried block relative to the
  same origin, so the two cannot interact. Documented under *End images*.
- Added `tests/test_carry_modes.py`, which loads `patches.py` against stubbed
  Wan2GP modules and exercises both paths.

## 1.1
- Fixed: the coordinate correction displaced the carried audio history by one
  frame - 41.7 ms at 24 fps - and pushed its last latent past `target_origin`.
  It moved everything anchored to `target_origin` earlier and exempted the audio
  history, so the audio's relationship to the target changed even though the
  carried audio tail ends at the join exactly where a native encode's does. The
  correction now moves the carried video block one frame later instead, which is
  identical in relative terms and leaves every other distance - audio, text,
  keyframes - untouched.
- Added `SWL_AUDIO_CONTEXT`, extra seconds of audio history beyond Wan2GP's
  overlap-derived 0.75s window, with a matching panel control. Off by default.
  Gated on the video carry and `fix_coords`, since the compensating layout shift
  lives there.
- The install line now reports `audio_context`, and every window logs which
  carry path it took, so a setting that is not applying can be seen rather than
  guessed at.
- Added `tests/test_audio_context.py`, which checks the layout geometry
  numerically at 0.5, 1, 2 and 4 seconds.

## 1.0.2
- Fixed: the plugin stayed inert on every build. `_preflight` looked for
  `video_latent_frames` on `MiniMaxH3Pipeline`, but it is a module-level
  function in `pipeline.py`, so the check could never pass and 1.0.0 and 1.0.1
  declared every Wan2GP unpatchable - including the one they were written
  against. Preflight now verifies pipeline helpers by performing the same
  import the runtime path performs, so the two cannot drift apart.
- `tests/test_install_table.py` now checks that every name imported from
  `models.minimax_h3.*` anywhere in the plugin is also imported by
  `_preflight`, and that no module-level name is probed on a class. It fails
  on the 1.0.0 code.

## 1.0.1
- Colour consistency now checks that a window is uniform before correcting it.
  The correction is measured at the window's opening and applied to its closing
  latents, so a cut part way through put the measurement and its target on
  opposite sides of it and the carried block was corrected toward a grade its
  own scene never had. The window's head is now compared against its own tail
  and the correction is withheld beyond `SWL_COLOUR_SCENE`.

## 1.0.0
- First release.
- Install now preflights every patch target and the packing constants, and
  applies all seven bindings as a unit with rollback on failure. A Wan2GP
  that moves a binding or changes the frame grid makes the plugin stay
  inert with a reason rather than patch something it cannot reason about.
- Optional bindings degrade instead of aborting: no audio VAE means video
  latents only, no Ref2VA builder means plain sliding windows only.
- The packing module is resolved once at preflight and reused, so the
  coordinate correction no longer hard-codes one import path while
  preflight accepts two.
- Video and audio latent carry are both on by default.
- Colour consistency is labelled experimental in the panel.
- Documented that this must not be run alongside Sliding Window Anchor.

### 0.7.2
- Colour correction never ran. `_patched_decode` used
  `_measure_colour(decoded) or decoded`, and `or` calls `bool()` on a tensor,
  which raises; the surrounding `except` then caught it and logged a line that
  read like a deliberate fallback. Fixed, and the handler now words an
  unexpected failure as a defect.
- Added `tests/test_no_tensor_truthiness.py`, an AST lint forbidding tensor
  values in boolean contexts across the plugin.

### 0.7.1
- The join check tested one offset with a tolerance, which a slow shot passes
  even when the cached tail is displaced by several frames; the carried block
  was then placed as though it ended at the join and the overrun was rendered
  again. It now searches the decoded tail for the join and reports the skew.
- Added `tests/test_alignment.py`.

### 0.7.0
- Colour consistency, applied to the carried latents rather than to decoded
  pixels, so it is compatible with carrying. Off by default.
- `both` scope corrects the window as well, keeping the cached latents in
  step with it, so a visible join can be fixed rather than only contained.
- A Continue Video run takes its colour reference from the frames handed in,
  so the join between source and new video is no longer the one never
  checked.

### 0.6.3
- Coordinate correction shifted the target-anchored rows in the wrong direction,
  widening the skew from one frame to two rather than closing it. With
  `SWL_FIX_COORDS=1` the default was worse aligned than `SWL_FIX_COORDS=0`.
- Added `tests/test_layout_contract.py`, which fails on the old sign.
- Added `tools/measure_joins.py`.
- Renamed to Sliding Window Latents; environment prefix `H3LC_` is now `SWL_`,
  and the plugin id is now `SlidingWindowLatents`. Update the enabled-plugins
  entry and the folder name to match.
