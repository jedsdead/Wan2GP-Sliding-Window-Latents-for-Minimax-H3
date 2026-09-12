#!/usr/bin/env python3
"""Does the colour path actually engage on real tensors?

Needs torch (CPU only - no Wan2GP, no GPU).

    python tests/test_colour_runtime.py

test_colour_latents.py proves the maths and test_colour_anchor.py proves the
anchoring. Neither touches _measure_colour, which is the function a render
actually calls, on tensors shaped the way a decode actually produces them.

That gap is not academic. In 0.7.2 the whole feature was dead in exactly this
layer - `_measure_colour(decoded) or decoded` called bool() on a tensor, raised,
and the surrounding except logged it as a considered fallback - while the maths
tests passed green the entire time.
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PLUGIN))
from importlib import import_module  # noqa: E402

patches = import_module(f"{os.path.basename(PLUGIN)}.patches")

FAILURES = []


def check(label, condition, detail=""):
    print(f"  [{'pass' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def window(frames=40, size=48, brightness=0.0, tint=0.0, seed=0):
    """A decoded window: (1, 3, T, H, W), signed [-1, 1], like the VAE emits."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.rand(1, 3, frames, size, size, generator=generator) * 0.35 + 0.30
    base = base + brightness
    base[:, 0] += tint                       # push red
    return (base.clamp(0.0, 1.0) * 2.0 - 1.0)


patches.CONFIG.colour = True
patches.CONFIG.colour_scope = "latents"
patches.CONFIG.colour_match = "first"
patches.CONFIG.colour_axes = ("brightness", "cast")
patches.CONFIG.verbose = False

print("the colour path on decode-shaped tensors")

# Window 1: nothing to match against yet, but it must set a reference.
patches.STATE.invalidate("test")
patches.STATE.colour_correction = None
out = patches._measure_colour(window(seed=1))
check("window 1 returns a tensor, not None", isinstance(out, torch.Tensor))
check("window 1 records a reference for the next window",
      patches.STATE.colour_reference is not None)
check("window 1 has nothing to correct yet",
      patches.STATE.colour_correction is None)

# Window 2: visibly lifted and warmed - the drift the feature exists to catch.
out2 = patches._measure_colour(window(brightness=0.05, tint=0.04, seed=2))
check("window 2 returns a tensor", isinstance(out2, torch.Tensor))
check("window 2 produced a correction", patches.STATE.colour_correction is not None,
      "" if patches.STATE.colour_correction is not None else "correction is None")

if patches.STATE.colour_correction is not None:
    A, b = patches.STATE.colour_correction
    check("the correction is a 24x24 matrix and a 24-vector",
          tuple(A.shape) == (24, 24) and tuple(b.shape) == (24,),
          f"A{tuple(A.shape)} b{tuple(b.shape)}")
    check("the correction is finite",
          bool(torch.isfinite(torch.as_tensor(A)).all()) and
          bool(torch.isfinite(torch.as_tensor(b)).all()))
    check("the correction is not the identity",
          float(torch.as_tensor(b).abs().max()) > 1e-6,
          f"max |b| {float(torch.as_tensor(b).abs().max()):.5f}")

    # And it must survive being applied to a latent block.
    latents = torch.randn(1, 24, 7, 4, 4)
    colour_mod = import_module(f"{os.path.basename(PLUGIN)}.colour")
    applied = colour_mod.apply_torch(latents, A, b)
    check("applying it to carried latents preserves shape and dtype",
          applied.shape == latents.shape and applied.dtype == latents.dtype)
    check("applying it actually changes the latents",
          not torch.allclose(applied, latents),
          f"max change {float((applied - latents).abs().max()):.4f}")

# A hard cut must be discarded rather than dragged toward the old grade.
patches.STATE.invalidate("test")
patches._measure_colour(window(seed=3))
patches._measure_colour(window(brightness=0.45, seed=4))
check("a scene-sized jump is refused, not corrected",
      patches.STATE.colour_correction is None,
      "correction applied across what should read as a cut")

# The 0.7.2 regression: a tensor in a boolean context anywhere on this path.
patches.STATE.invalidate("test")
try:
    patches._measure_colour(window(seed=5))
    patches._measure_colour(window(brightness=0.05, seed=6))
    raised = None
except Exception as error:                      # noqa: BLE001
    raised = error
check("no exception escapes the measurement path", raised is None,
      f"{type(raised).__name__}: {raised}" if raised else "")

print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("all checks passed")
