# Changelog

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
