"""How faithful is jlens's own fitted J_l transport, by depth?

Not part of the ``subvocal`` package -- an ad hoc script per CLAUDE.md's own
guidance. Downloads ``Qwen/Qwen3.5-4B`` (~9.3GB) on first run; see the
README's "Resource requirements" section.

readout() (and everything metrics.py builds on it -- occupancy, FVE,
loading, boundaries) decodes ``unembed(J_l @ h_l)``. This measures how well
``J_l @ h_l`` actually resembles the model's real final-layer residual
``h_final`` from the same forward pass, via cosine similarity and R²
(fraction of variance explained), per fitted layer, pooled over positions
and prompts. A follow-up to the M2 sanity-check failures (see
scripts/m2_sanity_check.py and FINDINGS.md): if this linear approximation is
itself low-fidelity, weak/noisy downstream metrics could be explained by the
lens's own fit, not by the model's workspace genuinely being narrow.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/transport_fidelity_check.py
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from subvocal.lens import FittedLens, QWEN3_5_4B

PROMPTS = [
    "In the years following the collapse of the old trading routes, the "
    "merchants of the northern cities began to rebuild their fortunes "
    "through a new network of river ports, each one competing for the same "
    "scarce cargo of grain, timber, and dye that once flowed freely across "
    "the border.",
    "The research team spent three years cataloguing the beetles they found "
    "in the limestone cave system, carefully photographing each specimen "
    "before returning it to the exact crevice where it had been discovered.",
    "When the central bank announced the surprise rate increase, traders on "
    "the exchange floor scrambled to unwind positions they had built over "
    "the previous quarter, and by the closing bell the benchmark index had "
    "recorded its steepest single-day decline in nearly a decade.",
]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.where(denom > 0, dot / np.where(denom > 0, denom, 1.0), 0.0)


def r_squared(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    residual_energy = np.sum((actual - pred) ** 2, axis=-1)
    total_energy = np.sum(actual**2, axis=-1)
    return np.where(
        total_energy > 0,
        1.0 - residual_energy / np.where(total_energy > 0, total_energy, 1.0),
        0.0,
    )


def main() -> None:
    t0 = time.time()
    lens = FittedLens.from_pretrained(QWEN3_5_4B)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    final_layer = lens.n_layers - 1
    fitted_layers = lens.fitted_layers
    print(f"fitted layers: {fitted_layers}, final layer: {final_layer}", flush=True)

    for layer in fitted_layers:
        cos_all, r2_all = [], []
        for prompt in PROMPTS:
            n_pos = len(lens.encode(prompt))
            h_l = np.stack([lens.residual(prompt, p, layer) for p in range(n_pos)])
            h_final = np.stack([lens.residual(prompt, p, final_layer) for p in range(n_pos)])

            h_l_t = torch.from_numpy(h_l).to(lens.device)
            with torch.no_grad():
                predicted = lens.jacobian_lens.transport(h_l_t, layer).float().cpu().numpy()

            cos_all.append(cosine_sim(predicted, h_final))
            r2_all.append(r_squared(predicted, h_final))
        cos_all = np.concatenate(cos_all)
        r2_all = np.concatenate(r2_all)
        print(
            f"layer {layer:>2}: cosine(J_l@h_l, h_final) mean={cos_all.mean():.4f} "
            f"median={np.median(cos_all):.4f} min={cos_all.min():.4f} | "
            f"R^2 mean={r2_all.mean():.4f} median={np.median(r2_all):.4f}",
            flush=True,
        )
        lens.clear_cache()


if __name__ == "__main__":
    main()
