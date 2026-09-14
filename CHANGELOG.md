# Changelog

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
