"""M2's real-lens sanity checks (CLAUDE.md), run against the pinned model.

Not part of the ``subvocal`` package -- an ad hoc script per CLAUDE.md's own
guidance ("only ad hoc sanity-check scripts should load the real model").
Downloads ``Qwen/Qwen3.5-4B`` (~9.3GB) on first run; see the README's
"Resource requirements" section. Takes several minutes.

Checks, per CLAUDE.md:
  1. occupancy is near zero in the first third of layers and plateaus near
     25 in the middle band
  2. fve never exceeds 0.10
  3. all five boundary signals land within 10% of each other

Prompts are long enough, and evaluated positions restricted, so every
position probed is one ``jlens.fitting.valid_position_mask`` (source_layers
excludes the first 16 positions of any prompt -- they act as attention
sinks with atypical residual statistics the lens was never fit on)
considers valid. See FINDINGS.md for a recorded run's output and what it
means; this script only reproduces the numbers, not the interpretation.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/m2_sanity_check.py
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
from jlens.fitting import valid_position_mask

from subvocal import metrics
from subvocal.lens import FittedLens, QWEN3_5_4B

PROMPTS = [
    "In the years following the collapse of the old trading routes, the merchants of the "
    "northern cities began to rebuild their fortunes through a new network of river ports, "
    "each one competing for the same scarce cargo of grain, timber, and dye that once flowed "
    "freely across the border. Local governors, sensing an opportunity, imposed tolls on every "
    "barge that passed beneath their walls, and within a decade the price of bread had tripled "
    "in towns that once considered themselves prosperous.",
    "The research team spent three years cataloguing the beetles they found in the limestone "
    "cave system, carefully photographing each specimen before returning it to the exact crevice "
    "where it had been discovered. Many of the species had never been recorded above ground, "
    "and several appeared to have evolved a waxy coating on their exoskeletons that scientists "
    "believe helps them retain moisture in the unusually dry lower chambers of the cave.",
    "When the central bank announced the surprise rate increase, traders on the exchange floor "
    "scrambled to unwind positions they had built over the previous quarter, and by the closing "
    "bell the benchmark index had recorded its steepest single-day decline in nearly a decade. "
    "Analysts spent the following days debating whether the move reflected genuine concern about "
    "inflation or simply an attempt to restore credibility after months of conflicting signals.",
    "She picked up the violin her grandmother had left her and began to play a slow, mournful "
    "melody in the empty concert hall, the sound carrying up into the rafters where dust still "
    "hung in the afternoon light. It had been years since she last performed in public, and her "
    "fingers, stiff from disuse, fumbled over the passages she once knew by heart, yet something "
    "in the room seemed to steady her hand as the piece went on.",
]


def valid_positions(lens: FittedLens, prompt: str) -> list[int]:
    n = len(lens.encode(prompt))
    mask = valid_position_mask(n)
    return [i for i, v in enumerate(mask.tolist()) if v]


def main() -> None:
    t0 = time.time()
    print("Loading FittedLens.from_pretrained(QWEN3_5_4B)...", flush=True)
    lens = FittedLens.from_pretrained(QWEN3_5_4B)
    print(f"Loaded in {time.time() - t0:.1f}s. n_layers={lens.n_layers} d_model={lens.d_model}", flush=True)

    layers = metrics.subsample_layers(lens.n_layers)
    depths = [metrics.reindex_to_depth(l, lens.n_layers) for l in layers]
    print(f"Subsampled layers: {layers}", flush=True)
    print(f"Depths: {[round(d) for d in depths]}", flush=True)

    all_occ = []
    all_fve = []
    for prompt in PROMPTS:
        positions = valid_positions(lens, prompt)
        t1 = time.time()
        occ = metrics.occupancy_grid(lens, prompt, positions=positions, layers=layers, k_max=40)
        fve = metrics.fve_per_layer(
            lens, prompt, positions=positions, layers=layers, occupancy=occ, k_max=40
        )
        all_occ.append(occ)
        all_fve.append(fve)
        print(
            f"  prompt={prompt[:40]!r} n_valid_pos={len(positions)} ({time.time() - t1:.1f}s) "
            f"occ mean/layer={np.round(occ.mean(axis=0), 1).tolist()}",
            flush=True,
        )
        lens.clear_cache()

    occ_all = np.concatenate(all_occ, axis=0)
    fve_all = np.stack(all_fve, axis=0)
    mean_occ = occ_all.mean(axis=0)
    mean_fve = fve_all.mean(axis=0)
    max_fve = fve_all.max()

    n = len(layers)
    first_third = mean_occ[: n // 3]
    middle_band = mean_occ[n // 3 : 2 * n // 3]

    print("\n=== Occupancy sanity ===", flush=True)
    print(f"mean occupancy per layer: {np.round(mean_occ, 2).tolist()}", flush=True)
    print(f"first-third mean: {first_third.mean():.2f} (expect near 0)", flush=True)
    print(f"middle-band mean: {middle_band.mean():.2f} (expect plateau near 25)", flush=True)

    print("\n=== FVE sanity ===", flush=True)
    print(f"mean fve per layer: {np.round(mean_fve, 4).tolist()}", flush=True)
    print(f"max fve across all layers/prompts: {max_fve:.4f} (must be <= 0.10)", flush=True)

    print("\n=== Boundary signals ===", flush=True)
    t2 = time.time()
    # topk_accuracy/kurtosis/autocorrelation_signal iterate every position of
    # each prompt internally (no positions override in the public API), so
    # they still see the attention-sink positions the occupancy/fve pass
    # above filtered out. Reported as-is -- see FINDINGS.md for why this
    # matters less for these signals than it does for occupancy.
    acc = metrics.topk_accuracy_signal(lens, PROMPTS, layers=layers)
    kurt = metrics.kurtosis_signal(lens, PROMPTS, layers=layers)
    autocorr = metrics.autocorrelation_signal(lens, PROMPTS, layers=layers)
    eff_dim = metrics.effective_dim_signal(lens, layers=layers)
    cka = metrics.cka_signal(lens, layers=layers)
    depths_arr = np.array(depths)
    sig = {
        "topk_accuracy": metrics.boundary_from_curve(acc, depths_arr),
        "kurtosis": metrics.boundary_from_curve(kurt, depths_arr),
        "autocorrelation": metrics.boundary_from_curve(autocorr, depths_arr),
        "effective_dim": metrics.boundary_from_curve(eff_dim, depths_arr),
        "cka": metrics.boundary_from_curve(cka, depths_arr),
    }
    print(f"computed in {time.time() - t2:.1f}s", flush=True)
    for name, depth in sig.items():
        print(f"  {name}: depth {depth:.1f}", flush=True)
    spread = max(sig.values()) - min(sig.values())
    print(f"spread: {spread:.1f} (flag if > 10)", flush=True)

    print(f"\nTotal wall time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
