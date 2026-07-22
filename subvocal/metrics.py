"""Metrics that populate a :class:`~subvocal.profile.Profile`.

Two layers: pure array math (no lens/prompt dependency, unit-testable in
isolation) underneath lens-aware orchestration that produces the grids
``Profile`` expects.

Several formulas here are best-effort interpretations, not paper-verified
formulas — the public writeup of "Verbalizable Representations Form a
Global Workspace" describes Gradient Pursuit, the FVE random-control
procedure, and the five boundary signals qualitatively and defers exact
formulas to an appendix that wasn't available. Every such judgment call is
called out in the relevant docstring. The sanity checks CLAUDE.md specifies
(occupancy shape, FVE <= 0.10, boundary agreement), run against a real
fitted lens once one exists, are the actual test of whether these
interpretations hold up — this module does not assert them against
:class:`~subvocal.lens.StubLens`, since fake random dictionaries have no
reason to satisfy real-lens statistical properties.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from subvocal.profile import Boundaries, Profile

#: Default top-k accuracy window; the paper says "top-k" without pinning a
#: value, so this is a judgment call.
DEFAULT_TOPK_ACCURACY_K = 5


class MetricsLens(Protocol):
    """The duck-typed lens surface every function here is written against.
    :class:`~subvocal.lens.StubLens` already satisfies this; a future real
    lens wrapper only needs to match it."""

    n_layers: int
    vocab: list[str]

    def encode(self, prompt: str) -> Sequence[object]: ...
    def residual(self, prompt: str, position: int, layer: int) -> np.ndarray: ...
    def concept_direction(self, concept: str, layer: int) -> np.ndarray: ...
    def readout(self, prompt: str, position: int, layer: int) -> np.ndarray: ...
    def topk(
        self, prompt: str, position: int, layer: int, k: int = 25
    ) -> list[tuple[str, float]]: ...


# --------------------------------------------------------------------------
# Layer subsampling
# --------------------------------------------------------------------------


def subsample_layers(n_layers: int, n: int = 25) -> list[int]:
    """Evenly spaced layer indices, per the paper's 25-layer convention.

    Returns all layers if ``n_layers <= n``.
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    if n_layers <= n:
        return list(range(n_layers))
    idx = np.round(np.linspace(0, n_layers - 1, n)).astype(int)
    return sorted({int(i) for i in idx})


def reindex_to_depth(layer: int, n_layers: int) -> float:
    """Map a layer index to the paper's ``[0, 100]`` depth scale."""
    if n_layers <= 1:
        return 0.0
    return float(layer) / float(n_layers - 1) * 100.0


# --------------------------------------------------------------------------
# Concept dictionaries
# --------------------------------------------------------------------------


def concept_dictionary(
    lens: MetricsLens, layer: int, concepts: Sequence[str] | None = None
) -> tuple[list[str], np.ndarray]:
    """J-lens vectors at ``layer`` for ``concepts`` (default: full vocab).

    Returns ``(concepts, D)`` where ``D`` has shape ``(n_concepts, d_model)``.
    """
    names = list(concepts) if concepts is not None else list(lens.vocab)
    D = np.stack([lens.concept_direction(c, layer) for c in names]).astype(np.float32)
    return names, D


def random_dictionary(d_model: int, n: int, *, seed: int) -> np.ndarray:
    """``n`` isotropic random unit vectors in ``d_model`` dims: shape ``(n, d_model)``.

    The size-matched control used by :func:`occupancy_from_residuals` and
    :func:`fve_from_residuals`. Shared utility — the random-direction
    ablation control in a later milestone's ``debug.py`` should reuse this
    rather than reinventing it.
    """
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, d_model)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def cosine_loading(h: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Cosine similarity between ``h`` and ``v`` along the last axis.

    The atomic primitive behind CLAUDE.md's ``loading(concept, h)``. Zero
    vectors yield 0 rather than raising.
    """
    h = np.asarray(h, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    dot = np.sum(h * v, axis=-1)
    denom = np.linalg.norm(h, axis=-1) * np.linalg.norm(v, axis=-1)
    return np.where(denom > 0, dot / np.where(denom > 0, denom, 1.0), 0.0).astype(
        np.float32
    )


def loading_grid(
    lens: MetricsLens,
    prompt: str,
    concept: str,
    positions: Sequence[int] | None = None,
    layers: Sequence[int] | None = None,
) -> np.ndarray:
    """Unaveraged loading trace: shape ``(n_pos, n_layer)``. Feeds
    ``Profile.loading(concept)`` directly."""
    positions = (
        list(positions) if positions is not None else list(range(len(lens.encode(prompt))))
    )
    layers = list(layers) if layers is not None else subsample_layers(lens.n_layers)
    grid = np.zeros((len(positions), len(layers)), dtype=np.float32)
    for li, layer in enumerate(layers):
        v = lens.concept_direction(concept, layer)
        for pi, pos in enumerate(positions):
            grid[pi, li] = cosine_loading(lens.residual(prompt, pos, layer), v)
    return grid


def loading_trace(
    lens: MetricsLens,
    prompt: str,
    concept: str,
    positions: Sequence[int],
    layers: Sequence[int] | None = None,
) -> np.ndarray:
    """Loading trace over layers, averaged over ``positions``: shape
    ``(n_layer,)``. What CLAUDE.md's M2 bullet describes; used by ``probe()``
    in a later milestone."""
    positions = list(positions)
    if not positions:
        raise ValueError("positions must be non-empty")
    return loading_grid(lens, prompt, concept, positions=positions, layers=layers).mean(
        axis=0
    )


# --------------------------------------------------------------------------
# Gradient Pursuit / occupancy
# --------------------------------------------------------------------------


def gradient_pursuit(
    h: np.ndarray, D: np.ndarray, k: int | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse nonnegative reconstruction of ``h`` from atoms of ``D``.

    Standard Gradient Pursuit (greedy atom selection by correlation, then a
    single gradient-direction step rather than a full least-squares re-solve
    like OMP — this is what makes it cheap enough to batch). Nonnegativity
    is enforced by only ever selecting positive-correlation atoms and
    clipping coefficients to >= 0 after each step.

    Args:
        h: Residuals, shape ``(B, d_model)`` (always 2D; pass ``h[None]``
            for a single vector).
        D: Dictionary atoms, shape ``(n_atoms, d_model)``.
        k: Sparsity budget. Either a scalar (applied to every row) or an
            array of shape ``(B,)`` for a different budget per row — used by
            :func:`occupancy_from_residuals` to binary-search per position
            in one batched call instead of one call per candidate K.

    Returns:
        ``(coefficients, active)``: coefficients shape ``(B, n_atoms)``
        (nonnegative, sparse), active shape ``(B, n_atoms)`` (bool mask of
        selected atoms).
    """
    h = np.asarray(h, dtype=np.float32)
    if h.ndim != 2:
        raise ValueError(f"h must be 2D (B, d_model), got shape {h.shape}")
    B, d = h.shape
    n_atoms = D.shape[0]

    k_arr = np.full(B, k, dtype=np.int64) if np.isscalar(k) else np.asarray(k, dtype=np.int64)
    if k_arr.shape != (B,):
        raise ValueError(f"k array shape {k_arr.shape} != ({B},)")
    if np.any(k_arr < 0):
        raise ValueError("k must be >= 0")
    n_steps = int(min(k_arr.max(), n_atoms)) if B else 0

    r = h.copy()
    c = np.zeros((B, n_atoms), dtype=np.float32)
    active = np.zeros((B, n_atoms), dtype=bool)
    Dt = D.T.astype(np.float32)
    rows = np.arange(B)

    for step in range(n_steps):
        still_running = k_arr > step
        if not still_running.any():
            break
        corr = r @ Dt
        corr = np.where(active, -np.inf, corr)
        best = np.argmax(corr, axis=1)
        best_corr = corr[rows, best]
        gainful = still_running & (best_corr > 0)
        active[rows[gainful], best[gainful]] = True

        g = np.where(active, r @ Dt, 0.0)
        G = g @ D
        denom = np.sum(G * G, axis=1)
        alpha = np.where(denom > 0, np.sum(r * G, axis=1) / np.where(denom > 0, denom, 1.0), 0.0)
        alpha = np.where(still_running, alpha, 0.0)
        c = np.clip(c + alpha[:, None] * g, a_min=0.0, a_max=None)
        r = h - c @ D

    return c, active


def variance_explained(h: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """Fraction of ``h``'s energy captured by ``recon``, along the last axis."""
    h = np.asarray(h, dtype=np.float32)
    recon = np.asarray(recon, dtype=np.float32)
    residual_energy = np.sum((h - recon) ** 2, axis=-1)
    total_energy = np.sum(h**2, axis=-1)
    return np.where(
        total_energy > 0, 1.0 - residual_energy / np.where(total_energy > 0, total_energy, 1.0), 0.0
    ).astype(np.float32)


def occupancy_from_residuals(
    h_batch: np.ndarray, D: np.ndarray, control_D: np.ndarray, k_max: int = 40
) -> np.ndarray:
    """Per-position occupancy: the largest K (up to ``k_max``) at which the
    K-th atom still improves reconstruction more than it would for a
    same-size random control.

    Adaptive: binary search over K in ``[1, k_max]`` per position — O(log
    k_max) Gradient Pursuit evaluations, not an exhaustive 1..k_max sweep.
    Assumes "real marginal gain beats control marginal gain" is monotone in
    K (true while inside the workspace band, false once saturated); this is
    a standard elbow-finding heuristic, not guaranteed by the algorithm.

    Args:
        h_batch: Residuals, shape ``(B, d_model)``.
        D: Real J-lens dictionary, shape ``(n_atoms, d_model)``.
        control_D: Size-matched random-direction control, shape
            ``(n_atoms, d_model)`` -- matched to ``D``'s atom count, not to
            ``k_max``. A control merely ``>= k_max``-sized but far smaller
            than ``D`` is a much less overcomplete dictionary, which lets a
            wildly overcomplete real ``D`` (e.g. a full ~150k-token
            vocabulary against ``d_model`` in the low thousands) "beat" it at
            every K almost regardless of real conceptual content -- occupancy
            never converges below ``k_max`` and excess FVE is inflated by the
            size gap rather than genuine signal.
        k_max: Upper bound on K to search.

    Returns:
        Per-position occupancy, shape ``(B,)``, float32 (integer-valued).
    """
    h_batch = np.asarray(h_batch, dtype=np.float32)
    if h_batch.ndim != 2:
        raise ValueError(f"h_batch must be 2D (B, d_model), got shape {h_batch.shape}")
    B = h_batch.shape[0]
    k_max = int(min(k_max, D.shape[0], control_D.shape[0]))
    if k_max < 1:
        raise ValueError("k_max must allow at least one atom")

    def beats_control(k: np.ndarray) -> np.ndarray:
        k_prev = np.maximum(k - 1, 0)
        real_c, _ = gradient_pursuit(h_batch, D, k)
        real_c_prev, _ = gradient_pursuit(h_batch, D, k_prev)
        ctrl_c, _ = gradient_pursuit(h_batch, control_D, k)
        ctrl_c_prev, _ = gradient_pursuit(h_batch, control_D, k_prev)
        real_gain = variance_explained(h_batch, real_c @ D) - variance_explained(
            h_batch, real_c_prev @ D
        )
        ctrl_gain = variance_explained(
            h_batch, ctrl_c @ control_D
        ) - variance_explained(h_batch, ctrl_c_prev @ control_D)
        return real_gain > ctrl_gain

    lo = np.ones(B, dtype=np.int64)
    hi = np.full(B, k_max, dtype=np.int64)
    beats_at_max = beats_control(hi)
    active = ~beats_at_max
    final_k = np.where(beats_at_max, k_max, 1).astype(np.int64)

    while active.any():
        mid = np.maximum((lo + hi) // 2, 1)
        beats_mid = beats_control(mid)
        raise_lo = active & beats_mid & (mid > lo)
        lo = np.where(raise_lo, mid, lo)
        lower_hi = active & ~beats_mid & (mid < hi)
        hi = np.where(lower_hi, mid, hi)
        settled = active & ~(raise_lo | lower_hi)
        final_k[settled] = lo[settled]
        active = active & ~settled

    return final_k.astype(np.float32)


def occupancy_grid(
    lens: MetricsLens,
    prompt: str,
    positions: Sequence[int] | None = None,
    layers: Sequence[int] | None = None,
    k_max: int = 40,
    seed: int = 0,
) -> np.ndarray:
    """Occupancy over every position x subsampled layer: shape ``(n_pos,
    n_layer)``. Feeds ``Profile.occupancy()``."""
    positions = (
        list(positions) if positions is not None else list(range(len(lens.encode(prompt))))
    )
    layers = list(layers) if layers is not None else subsample_layers(lens.n_layers)
    grid = np.zeros((len(positions), len(layers)), dtype=np.float32)
    for li, layer in enumerate(layers):
        _, D = concept_dictionary(lens, layer)
        control_D = random_dictionary(D.shape[1], D.shape[0], seed=seed + layer)
        h_batch = np.stack([lens.residual(prompt, pos, layer) for pos in positions])
        grid[:, li] = occupancy_from_residuals(h_batch, D, control_D, k_max=k_max)
    return grid


# --------------------------------------------------------------------------
# fve
# --------------------------------------------------------------------------


def fve_from_residuals(
    h_batch: np.ndarray, D: np.ndarray, control_D: np.ndarray, k: int | np.ndarray
) -> np.ndarray:
    """Per-position variance explained by ``k`` atoms of ``D``, in excess of
    a same-size random control: shape ``(B,)``."""
    h_batch = np.asarray(h_batch, dtype=np.float32)
    real_c, _ = gradient_pursuit(h_batch, D, k)
    ctrl_c, _ = gradient_pursuit(h_batch, control_D, k)
    return variance_explained(h_batch, real_c @ D) - variance_explained(
        h_batch, ctrl_c @ control_D
    )


def fve_per_layer(
    lens: MetricsLens,
    prompt: str,
    positions: Sequence[int] | None = None,
    layers: Sequence[int] | None = None,
    occupancy: np.ndarray | None = None,
    k_max: int = 40,
    seed: int = 0,
) -> np.ndarray:
    """Mean excess variance explained per layer: shape ``(n_layer,)``. Feeds
    ``Profile.fve()``. Reuses each position's own occupancy K unless
    ``occupancy`` is supplied, to avoid recomputing it."""
    positions = (
        list(positions) if positions is not None else list(range(len(lens.encode(prompt))))
    )
    layers = list(layers) if layers is not None else subsample_layers(lens.n_layers)
    if occupancy is None:
        occupancy = occupancy_grid(
            lens, prompt, positions=positions, layers=layers, k_max=k_max, seed=seed
        )
    occupancy = np.asarray(occupancy)
    expected = (len(positions), len(layers))
    if occupancy.shape != expected:
        raise ValueError(f"occupancy shape {occupancy.shape} != {expected}")

    fve_vals = np.zeros(len(layers), dtype=np.float32)
    for li, layer in enumerate(layers):
        _, D = concept_dictionary(lens, layer)
        control_D = random_dictionary(D.shape[1], D.shape[0], seed=seed + layer)
        h_batch = np.stack([lens.residual(prompt, pos, layer) for pos in positions])
        k_row = np.clip(occupancy[:, li].astype(np.int64), 1, k_max)
        fve_vals[li] = float(np.mean(fve_from_residuals(h_batch, D, control_D, k_row)))
    return fve_vals


# --------------------------------------------------------------------------
# Boundary signals
# --------------------------------------------------------------------------


def _model_top1_proxy(lens: MetricsLens, prompt: str, position: int) -> str:
    """The lens's own final-layer readout stands in for "the model's actual
    next-token prediction" — exact for a real lens (the final layer's own
    unembed output *is* the model's logits), and the natural analogue for
    ``StubLens``."""
    return lens.topk(prompt, position, lens.n_layers - 1, k=1)[0][0]


def topk_accuracy_signal(
    lens: MetricsLens,
    prompts: Sequence[str],
    k: int = DEFAULT_TOPK_ACCURACY_K,
    layers: Sequence[int] | None = None,
) -> np.ndarray:
    """Fraction of positions where the model's actual top-1 token appears in
    the lens's top-``k`` at each layer: shape ``(n_layer,)``."""
    layers = list(layers) if layers is not None else subsample_layers(lens.n_layers)
    hits = np.zeros(len(layers), dtype=np.float64)
    total = 0
    for prompt in prompts:
        for pos in range(len(lens.encode(prompt))):
            actual = _model_top1_proxy(lens, prompt, pos)
            total += 1
            for li, layer in enumerate(layers):
                top = {tok for tok, _ in lens.topk(prompt, pos, layer, k=k)}
                if actual in top:
                    hits[li] += 1
    if total == 0:
        raise ValueError("prompts produced no positions")
    return (hits / total).astype(np.float32)


def _excess_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    var = x.var()
    if var == 0:
        return 0.0
    m4 = np.mean((x - x.mean()) ** 4)
    return float(m4 / (var**2) - 3.0)


def kurtosis_signal(
    lens: MetricsLens, prompts: Sequence[str], layers: Sequence[int] | None = None
) -> np.ndarray:
    """Excess (Fisher) kurtosis of the pooled readout logit distribution per
    layer: shape ``(n_layer,)``."""
    layers = list(layers) if layers is not None else subsample_layers(lens.n_layers)
    values = np.zeros(len(layers), dtype=np.float32)
    for li, layer in enumerate(layers):
        pooled = np.concatenate(
            [
                lens.readout(prompt, pos, layer)
                for prompt in prompts
                for pos in range(len(lens.encode(prompt)))
            ]
        )
        values[li] = _excess_kurtosis(pooled)
    return values


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    return x - np.log(np.sum(np.exp(x)))


def _top1_logprob(lens: MetricsLens, prompt: str, position: int, layer: int) -> float:
    return float(_log_softmax(lens.readout(prompt, position, layer)).max())


def _lag_autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag:
        return 0.0
    x = x - x.mean()
    denom = np.sum(x**2)
    if denom == 0:
        return 0.0
    return float(np.sum(x[:-lag] * x[lag:]) / denom)


def autocorrelation_signal(
    lens: MetricsLens,
    prompts: Sequence[str],
    layers: Sequence[int] | None = None,
    lag: int = 1,
    seed: int = 0,
) -> np.ndarray:
    """Lag-``lag`` autocorrelation of the top-1 lens token's log-probability
    across positions, minus the same statistic on a position-shuffled null:
    shape ``(n_layer,)``."""
    layers = list(layers) if layers is not None else subsample_layers(lens.n_layers)
    rng = np.random.default_rng(seed)
    values = np.zeros(len(layers), dtype=np.float32)
    for li, layer in enumerate(layers):
        real_vals, null_vals = [], []
        for prompt in prompts:
            n_pos = len(lens.encode(prompt))
            if n_pos <= lag:
                continue
            logprobs = np.array(
                [_top1_logprob(lens, prompt, pos, layer) for pos in range(n_pos)]
            )
            real_vals.append(_lag_autocorr(logprobs, lag))
            shuffled = logprobs.copy()
            rng.shuffle(shuffled)
            null_vals.append(_lag_autocorr(shuffled, lag))
        values[li] = float(np.mean(real_vals) - np.mean(null_vals)) if real_vals else 0.0
    return values


def effective_dim_signal(
    lens: MetricsLens, layers: Sequence[int] | None = None, variance_share: float = 0.9
) -> np.ndarray:
    """Fraction of ``d_model`` needed to capture ``variance_share`` of the
    variance across the layer's J-lens vectors: shape ``(n_layer,)``."""
    layers = list(layers) if layers is not None else subsample_layers(lens.n_layers)
    values = np.zeros(len(layers), dtype=np.float32)
    for li, layer in enumerate(layers):
        _, D = concept_dictionary(lens, layer)
        singular_values = np.linalg.svd(D, compute_uv=False)
        energy = singular_values**2
        total = energy.sum()
        if total == 0:
            values[li] = 0.0
            continue
        cumulative = np.cumsum(energy) / total
        n_components = int(np.searchsorted(cumulative, variance_share) + 1)
        values[li] = n_components / D.shape[1]
    return values


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between ``(n_samples, n_features)`` matrices sharing
    ``n_samples`` (Kornblith et al. 2019's simplified linear-kernel form)."""
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X, Y must share n_samples: {X.shape[0]} != {Y.shape[0]}")
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(Yc.T @ Xc, ord="fro") ** 2
    denom = np.linalg.norm(Xc.T @ Xc, ord="fro") * np.linalg.norm(Yc.T @ Yc, ord="fro")
    return float(hsic / denom) if denom > 0 else 0.0


def cka_signal(
    lens: MetricsLens, layers: Sequence[int] | None = None, concepts: Sequence[str] | None = None
) -> np.ndarray:
    """CKA between each layer's J-lens dictionary and the next subsampled
    layer's: shape ``(n_layer,)``. The paper points at layer-pair CKA
    (Figure 27) without specifying the pairing used for a boundary call;
    adjacent-layer similarity ("where does the geometry stabilize") is the
    most natural read for a boundary signal."""
    layers = list(layers) if layers is not None else subsample_layers(lens.n_layers)
    dictionaries = [concept_dictionary(lens, layer, concepts=concepts)[1] for layer in layers]
    values = np.zeros(len(layers), dtype=np.float32)
    for i in range(len(layers)):
        j = min(i + 1, len(layers) - 1)
        values[i] = 1.0 if i == j else linear_cka(dictionaries[i], dictionaries[j])
    return values


def boundary_from_curve(values: np.ndarray, depths: np.ndarray, edge_n: int = 3) -> float:
    """Depth at which ``values`` first crosses the midpoint between its
    early-layer baseline and late-layer plateau (linearly interpolated
    between bracketing sampled layers).

    Direction-agnostic by construction: baseline/plateau come from the data
    rather than assuming which of the five signals rise vs. fall with depth
    (the paper doesn't specify), so this one rule applies uniformly to all
    of them.
    """
    values = np.asarray(values, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    if values.shape != depths.shape:
        raise ValueError("values and depths must be the same shape")
    if len(values) < 2:
        raise ValueError("need at least 2 layers to locate a boundary")

    order = np.argsort(depths)
    values, depths = values[order], depths[order]
    edge_n = max(1, min(edge_n, len(values) // 2))
    baseline = values[:edge_n].mean()
    plateau = values[-edge_n:].mean()
    mid = (baseline + plateau) / 2.0
    rising = plateau >= baseline

    for i in range(len(values) - 1):
        v0, v1 = values[i], values[i + 1]
        crossed = (v0 <= mid <= v1) if rising else (v0 >= mid >= v1)
        if crossed and v1 != v0:
            t = (mid - v0) / (v1 - v0)
            return float(depths[i] + t * (depths[i + 1] - depths[i]))

    closest = int(np.argmin(np.abs(values - mid)))
    return float(depths[closest])


def compute_boundaries(
    lens: MetricsLens, prompts: Sequence[str], layers: Sequence[int] | None = None
) -> Boundaries:
    """Run all five boundary signals and locate each one's crossing depth."""
    layers = list(layers) if layers is not None else subsample_layers(lens.n_layers)
    depths = np.array([reindex_to_depth(l, lens.n_layers) for l in layers])

    acc = topk_accuracy_signal(lens, prompts, layers=layers)
    kurt = kurtosis_signal(lens, prompts, layers=layers)
    autocorr = autocorrelation_signal(lens, prompts, layers=layers)
    eff_dim = effective_dim_signal(lens, layers=layers)
    cka = cka_signal(lens, layers=layers)

    return Boundaries.from_signals(
        topk_accuracy=boundary_from_curve(acc, depths),
        kurtosis=boundary_from_curve(kurt, depths),
        autocorrelation=boundary_from_curve(autocorr, depths),
        effective_dim=boundary_from_curve(eff_dim, depths),
        cka=boundary_from_curve(cka, depths),
    )


# --------------------------------------------------------------------------
# Profile builder
# --------------------------------------------------------------------------


def build_profile(
    lens: MetricsLens,
    prompt: str,
    *,
    concepts: Sequence[str],
    positions: Sequence[int] | None = None,
    layers: Sequence[int] | None = None,
    k_max: int = 40,
    topk_k: int = 25,
    boundary_prompts: Sequence[str] | None = None,
    seed: int = 0,
) -> Profile:
    """Compute everything a :class:`Profile` needs for one prompt.

    Args:
        lens: Lens to read out from.
        prompt: The prompt to profile.
        concepts: Concepts to compute a loading trace for.
        positions: Positions to include. Defaults to every position in
            ``prompt``.
        layers: Layers to include. Defaults to :func:`subsample_layers`.
        k_max: Sparsity search bound for occupancy/FVE.
        topk_k: Tokens stored per (position, layer).
        boundary_prompts: Corpus to pool for the five boundary signals.
            Defaults to ``[prompt]`` — a single prompt is a thin basis for
            e.g. autocorrelation, so pass a real corpus when boundaries()
            matters.
        seed: Random-control seed.
    """
    positions = (
        list(positions) if positions is not None else list(range(len(lens.encode(prompt))))
    )
    layers = list(layers) if layers is not None else subsample_layers(lens.n_layers)
    topk_k = min(topk_k, len(lens.vocab))

    occ = occupancy_grid(lens, prompt, positions=positions, layers=layers, k_max=k_max, seed=seed)
    fve_vals = fve_per_layer(
        lens, prompt, positions=positions, layers=layers, occupancy=occ, k_max=k_max, seed=seed
    )
    boundaries = compute_boundaries(lens, boundary_prompts or [prompt], layers=layers)

    topk_tokens = np.empty((len(positions), len(layers), topk_k), dtype=object)
    topk_scores = np.zeros((len(positions), len(layers), topk_k), dtype=np.float32)
    for pi, pos in enumerate(positions):
        for li, layer in enumerate(layers):
            for ki, (tok, score) in enumerate(lens.topk(prompt, pos, layer, k=topk_k)):
                topk_tokens[pi, li, ki] = tok
                topk_scores[pi, li, ki] = score

    loadings = {
        c: loading_grid(lens, prompt, c, positions=positions, layers=layers) for c in concepts
    }

    return Profile(
        positions=positions,
        layers=layers,
        occupancy=occ,
        fve=fve_vals,
        boundaries=boundaries,
        topk_tokens=topk_tokens,
        topk_scores=topk_scores,
        loadings=loadings,
    )
