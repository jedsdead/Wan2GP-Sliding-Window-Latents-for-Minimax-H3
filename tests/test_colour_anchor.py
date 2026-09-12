#!/usr/bin/env python3
"""Which reference a window is matched against, when windows drift internally.

Numpy only - no Wan2GP, no torch, no GPU.

    python tests/test_colour_anchor.py

test_colour_latents.py models a window as a single grade: its head and its
tail are the same measurement.  Under that model every anchoring policy
coincides, which is why `previous` looks sufficient there.

A real window is not uniform.  The re-synthesis error accrues *across* it, so
the tail - the part that gets carried - is further from the reference than the
head, which is the part that gets measured.  `within_window_change` exists
precisely because head and tail differ.

This file puts that gap into the simulation and pins what it does to each
policy.  HEAD_FRAC is how much of the window's drift has accrued by the time
the head is measured; 1.0 reproduces the uniform model of the other file.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import colour  # noqa: E402

FAILURES = []
rng = np.random.default_rng(11)


def check(label, condition, detail=""):
    print(f"  [{'pass' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


M, BIAS, SOURCE = colour.load_factors()


def sample_latents(n=6000):
    latents = rng.normal(0.0, 1.0, size=(n, M.shape[0]))
    rgb = latents @ M + BIAS
    return latents[np.all((rgb > -0.9) & (rgb < 0.9), axis=1)][:2000]


def unit_rgb(latents):
    return np.clip((latents @ M + BIAS + 1.0) * 0.5, 0.0, 1.0)


def stats(latents):
    return colour.ycbcr_stats(unit_rgb(latents))


GAIN, LIFT = (0.995, 0.995, 0.995), 0.008


def drifts(head_frac):
    """The window's error at its head and at its tail."""
    tail = colour.latent_correction(GAIN, [LIFT, 0.0, 0.0], M, BIAS)
    head = colour.latent_correction(tuple(g ** head_frac for g in GAIN),
                                    [LIFT * head_frac, 0.0, 0.0], M, BIAS)
    return head, tail


def run(start, head_drift, tail_drift, windows=8, anchor="previous"):
    """Trace the carried grade over a run, matching each window to `anchor`.

    Mirrors _measure_colour: measure the window's head against the reference,
    compose onto the running total, apply the total to the tail latents that
    become the next window's conditioning, and take the next reference from
    this window's own tail.
    """
    tail = start
    first = opening = stats(tail)
    reference, total = opening, None
    trace = [opening["mean"][0]]

    for _ in range(2, windows + 1):
        conditioning = (colour.apply_numpy(tail, *colour.latent_correction(*total, M, BIAS))
                        if total is not None else tail)
        head_latents = colour.apply_numpy(conditioning, *head_drift)
        tail_latents = colour.apply_numpy(conditioning, *tail_drift)
        head, tail_stats = stats(head_latents), stats(tail_latents)
        trace.append(tail_stats["mean"][0])

        gain, offset, _ = colour.measure(
            first if anchor == "first" else reference, head, strength=1.0)
        if gain is not None:
            total = colour.compose(total, (gain, offset))
        reference, tail = tail_stats, tail_latents
    return np.array(trace)


def uncorrected(start, tail_drift, windows=8):
    latents, trace = start, [stats(start)["mean"][0]]
    for _ in range(2, windows + 1):
        latents = colour.apply_numpy(latents, *tail_drift)
        trace.append(stats(latents)["mean"][0])
    return np.array(trace)


print(f"anchoring under within-window drift  (latent-to-RGB map: {SOURCE})")
seed = sample_latents()
residuals = {}

for head_frac in (1.0, 0.75, 0.45, 0.2):
    head_drift, tail_drift = drifts(head_frac)
    loose = uncorrected(seed, tail_drift)
    base = abs(loose[-1] - loose[0])
    row = {}
    for anchor in ("previous", "first"):
        trace = run(seed, head_drift, tail_drift, anchor=anchor)
        row[anchor] = abs(trace[-1] - trace[0]) / base
    residuals[head_frac] = row
    print(f"    head_frac {head_frac:.2f}:  previous {row['previous'] * 100:5.1f}%   "
          f"first {row['first'] * 100:5.1f}%   of uncorrected drift")

# 1. The uniform case is the one the other test file models, and there the two
#    policies must agree - otherwise this simulation contradicts that one.
check("with a uniform window both anchors agree",
      abs(residuals[1.0]["previous"] - residuals[1.0]["first"]) < 0.01,
      f"previous {residuals[1.0]['previous'] * 100:.1f}% vs "
      f"first {residuals[1.0]['first'] * 100:.1f}%")

# 2. `first` composes the residual gap onto the total every window, which is an
#    integral term: it converges on the anchor wherever in the window the
#    measurement happens to land.
floor = residuals[1.0]["first"]
check("`first` reaches the same floor however the drift is distributed",
      all(abs(row["first"] - floor) < 0.01 for row in residuals.values()),
      "  ".join(f"{frac:.2f}:{row['first'] * 100:.1f}%"
                for frac, row in residuals.items()))

# 3. `previous` chases a reference that is itself drifting, so the less of the
#    window's error has accrued by the head, the more of it survives.
check("`previous` degrades as more drift accrues after the head",
      (residuals[0.2]["previous"] > residuals[0.45]["previous"]
       > residuals[0.75]["previous"] > residuals[1.0]["previous"]),
      "  ".join(f"{frac:.2f}:{row['previous'] * 100:.1f}%"
                for frac, row in residuals.items()))

# 4. The size of it.  This is the reason the default matters.
check("`previous` leaves several times more drift than `first` on a "
      "realistically non-uniform window",
      residuals[0.45]["previous"] > 2 * residuals[0.45]["first"],
      f"previous {residuals[0.45]['previous'] * 100:.1f}% vs "
      f"first {residuals[0.45]['first'] * 100:.1f}%")

print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("all checks passed")
