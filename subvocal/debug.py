"""Prompt debugging on top of ``metrics.py``: :func:`probe` and :func:`contrast`,
CLAUDE.md's first two M3 modes. ``propose()`` is a stretch goal and is not
implemented here.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from subvocal.metrics import (
    DEFAULT_DICTIONARY_SIZE,
    MetricsLens,
    concept_dictionary,
    cosine_loading,
    loading_grid,
    reindex_to_depth,
    subsample_layers,
)

#: Generic, mutually unrelated prompts -- the default null-distribution corpus
#: :func:`contrast` uses to estimate how much a concept's loading naturally
#: varies. A judgment call (CLAUDE.md specifies the normalization but not a
#: corpus); pass your own domain's prompts via ``baseline_prompts`` for
#: anything beyond a quick check, since "unrelated" is doing real statistical
#: work here.
DEFAULT_BASELINE_PROMPTS = [
    "The committee will reconvene on Thursday to finalize the annual budget.",
    "A light rain fell over the harbor as the ferry pulled away from the dock.",
    "Researchers isolated the compound after several failed extraction attempts.",
    "The children built a small fort out of cardboard boxes in the backyard.",
]


def _warn_if_multi_token(lens: MetricsLens, concept: str) -> None:
    """CLAUDE.md: "Warn at the API boundary whenever a user-supplied concept
    tokenizes to more than one token. The lens only sees single tokens and
    this is a real limitation, not an edge case." A warning, not a raise --
    the caller may still want the (degraded) result, e.g. ``concept_direction``
    on a :class:`~subvocal.lens.StubLens` doesn't actually require single
    tokens; :class:`~subvocal.lens.FittedLens` will separately raise when it
    can't resolve a single token id.
    """
    n_tokens = len(lens.encode(concept))
    if n_tokens != 1:
        warnings.warn(
            f"concept {concept!r} tokenizes to {n_tokens} tokens under this "
            "lens's tokenizer; the lens only reads out single tokens, so "
            "this reflects at most the first token's meaning, not the "
            "phrase as a whole.",
            stacklevel=3,
        )


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ProbeResult:
    """Where a concept peaks across depth for one prompt.

    ``eq=False``: holds ``ndarray`` fields, and numpy's elementwise ``==``
    breaks dataclass-generated equality (ambiguous truth value) -- falls back
    to identity comparison instead.

    Attributes:
        concept: The probed concept.
        layers: Layer indices scanned (caller's own index values).
        depths: Same layers, reindexed to the paper's 0-100 depth scale.
        trace: Loading averaged over the probed positions, shape ``(n_layer,)``.
        peak_layer: The layer with the highest loading.
        peak_depth: That layer's depth.
        peak_loading: The loading value there.
    """

    concept: str
    layers: list[int]
    depths: np.ndarray
    trace: np.ndarray
    peak_layer: int
    peak_depth: float
    peak_loading: float


def probe(
    lens: MetricsLens,
    prompt: str,
    concept: str,
    *,
    positions: Sequence[int] | None = None,
    layers: Sequence[int] | None = None,
) -> ProbeResult:
    """Where does ``concept`` load across ``prompt``'s depth?

    CLAUDE.md's first debug mode: the caller supplies a concept they expect
    to show up at some intermediate point, and this returns the loading
    trace plus its peak so they can check that expectation against the lens.

    Args:
        lens: Lens to read out from.
        prompt: The prompt to probe.
        concept: The expected intermediate concept.
        positions: Positions to average the trace over. Defaults to every
            position in ``prompt``.
        layers: Layers to scan. Defaults to
            :func:`~subvocal.metrics.subsample_layers`.

    Returns:
        A :class:`ProbeResult`.
    """
    _warn_if_multi_token(lens, concept)
    positions = (
        list(positions) if positions is not None else list(range(len(lens.encode(prompt))))
    )
    layers = list(layers) if layers is not None else subsample_layers(lens.n_layers)
    grid = loading_grid(lens, prompt, concept, positions=positions, layers=layers)
    trace = grid.mean(axis=0)
    depths = np.array([reindex_to_depth(l, lens.n_layers) for l in layers])
    peak_idx = int(np.argmax(trace))
    return ProbeResult(
        concept=concept,
        layers=layers,
        depths=depths,
        trace=trace,
        peak_layer=layers[peak_idx],
        peak_depth=float(depths[peak_idx]),
        peak_loading=float(trace[peak_idx]),
    )


# --------------------------------------------------------------------------
# contrast
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ConceptContrast:
    """One concept's working-vs-failing comparison. ``eq=False``: see
    :class:`ProbeResult`.

    Attributes:
        concept: The concept.
        score: The largest normalized (working - failing) loading delta
            within the scanned ``band``, across all positions and band
            layers. Positive means more present in ``working``.
        best_position: Position (caller's index) where ``score`` occurs.
        best_layer: Layer (caller's index) where ``score`` occurs.
        trace: Normalized delta over every scanned position/layer -- not
            just the band -- shape ``(n_pos, n_layer)``, for inspecting past
            the single best cell.
    """

    concept: str
    score: float
    best_position: int
    best_layer: int
    trace: np.ndarray


@dataclass(frozen=True)
class ContrastResult:
    """Ranked concept diffs between a working and a failing prompt.

    Attributes:
        working_prompt: The prompt that produced the desired behavior.
        failing_prompt: The prompt that didn't -- tokenizes to the same
            length as ``working_prompt`` (see :func:`contrast`).
        layers: Layers scanned.
        band: ``(lo, hi)`` layer-index bounds (inclusive)
            :attr:`ConceptContrast.score` was maximized over.
        hits: Every scanned concept, ranked by :attr:`ConceptContrast.score`
            descending -- strongly present in ``working``, weak or absent in
            ``failing`` first.
    """

    working_prompt: str
    failing_prompt: str
    layers: list[int]
    band: tuple[int, int]
    hits: list[ConceptContrast]

    def ranked(self, k: int = 10) -> list[ConceptContrast]:
        """Top ``k`` concepts by :attr:`ConceptContrast.score`."""
        return self.hits[:k]


def contrast(
    lens: MetricsLens,
    working_prompt: str,
    failing_prompt: str,
    *,
    concepts: Sequence[str] | None = None,
    layers: Sequence[int] | None = None,
    band: tuple[int, int] | None = None,
    baseline_prompts: Sequence[str] | None = None,
    max_atoms: int = DEFAULT_DICTIONARY_SIZE,
    seed: int = 0,
    std_floor: float = 0.05,
) -> ContrastResult:
    """Diff J-space content between a working and a failing prompt.

    CLAUDE.md's main M3 feature: position by position, layer by layer, which
    concepts are strongly present in ``working_prompt`` and weak or absent in
    ``failing_prompt``.

    The two prompts must be token-aligned -- interpreted here as *equal
    token count*, so position ``i`` means the same slot in both. CLAUDE.md
    says "token-aligned" without a precise definition; equal length is the
    weakest condition that still makes position-by-position diffing
    meaningful, and still allows genuinely different wording as long as it
    doesn't change the token count.

    Loading deltas are normalized per concept per layer by that concept's
    natural spread across ``baseline_prompts`` (std, floored at
    ``std_floor``) before ranking -- CLAUDE.md: "Normalize the loading delta
    against a baseline of how much that concept varies across unrelated
    prompt pairs, or you will surface junk." A concept that's naturally
    noisy needs a bigger raw delta to count as a real hit than one that's
    normally stable.

    Computed layer-major (one dictionary build per layer, reused across every
    concept and all three prompt sets) rather than the more obvious
    concept-major loop, since :class:`~subvocal.lens.FittedLens` only caches
    one layer's direction matrix at a time -- concept-major would rebuild it
    once per concept per layer.

    Args:
        lens: Lens to read out from.
        working_prompt: The prompt that produces the desired behavior.
        failing_prompt: The prompt that doesn't.
        concepts: Concepts to scan. Defaults to the same size-``max_atoms``
            vocab sample :func:`~subvocal.metrics.concept_dictionary` draws
            for occupancy -- pass an explicit list once you have a
            hypothesis to check specific concepts faster. Explicit concepts
            are checked for the single-token warning (see CLAUDE.md); the
            default sample is drawn from the vocab and is always single-token.
        layers: Layers to scan. Defaults to
            :func:`~subvocal.metrics.subsample_layers`.
        band: ``(lo, hi)`` layer-index bounds (inclusive) to rank
            :attr:`ConceptContrast.score` within -- the paper's "workspace
            band" convention. Defaults to all of ``layers``; pass the bounds
            from a prior :func:`~subvocal.metrics.compute_boundaries` call to
            restrict ranking to where the workspace actually is, rather than
            recomputing five boundary signals on every call.
        baseline_prompts: Corpus for the normalization above. Defaults to
            :data:`DEFAULT_BASELINE_PROMPTS`.
        max_atoms: Concept-dictionary size when ``concepts`` isn't given.
        seed: Concept-dictionary sampling seed.
        std_floor: Minimum per-concept-per-layer baseline std, so a concept
            with near-zero natural variance doesn't produce a spurious huge
            normalized score.

    Returns:
        A :class:`ContrastResult` ranking every scanned concept.

    Raises:
        ValueError: The two prompts don't tokenize to the same length, or
            ``band`` doesn't overlap any scanned layer.
    """
    working_toks = list(lens.encode(working_prompt))
    failing_toks = list(lens.encode(failing_prompt))
    if len(working_toks) != len(failing_toks):
        raise ValueError(
            "contrast() requires token-aligned prompts (equal token count) "
            "so position i means the same slot in both; got "
            f"{len(working_toks)} tokens ({working_toks}) vs "
            f"{len(failing_toks)} tokens ({failing_toks})."
        )
    positions = list(range(len(working_toks)))

    layers = list(layers) if layers is not None else subsample_layers(lens.n_layers)
    band = band if band is not None else (min(layers), max(layers))
    band_mask = np.array([band[0] <= l <= band[1] for l in layers])
    if not band_mask.any():
        raise ValueError(f"band={band} does not overlap any scanned layer in {layers}")

    if concepts is not None:
        concepts = list(concepts)
        for c in concepts:
            _warn_if_multi_token(lens, c)
    else:
        concepts, _ = concept_dictionary(lens, layers[0], max_atoms=max_atoms, seed=seed)

    baseline_prompts = (
        list(baseline_prompts) if baseline_prompts is not None else DEFAULT_BASELINE_PROMPTS
    )
    baseline_positions = [list(range(len(lens.encode(p)))) for p in baseline_prompts]

    n_concepts, n_pos, n_layer = len(concepts), len(positions), len(layers)
    normalized = np.zeros((n_concepts, n_pos, n_layer), dtype=np.float32)

    for li, layer in enumerate(layers):
        _, D = concept_dictionary(lens, layer, concepts=concepts)  # (n_concepts, d_model)

        working_h = np.stack([lens.residual(working_prompt, p, layer) for p in positions])
        failing_h = np.stack([lens.residual(failing_prompt, p, layer) for p in positions])
        working_score = cosine_loading(working_h[None, :, :], D[:, None, :])  # (n_concepts, n_pos)
        failing_score = cosine_loading(failing_h[None, :, :], D[:, None, :])

        baseline_h = np.concatenate(
            [
                np.stack([lens.residual(p, pos, layer) for pos in pos_list])
                for p, pos_list in zip(baseline_prompts, baseline_positions)
            ],
            axis=0,
        )  # (total_baseline_pos, d_model)
        baseline_score = cosine_loading(baseline_h[None, :, :], D[:, None, :])
        baseline_std = np.maximum(baseline_score.std(axis=1), std_floor)  # (n_concepts,)

        normalized[:, :, li] = (working_score - failing_score) / baseline_std[:, None]

    band_layers = np.array(layers)[band_mask]
    band_values = normalized[:, :, band_mask]  # (n_concepts, n_pos, n_band_layer)

    hits = []
    for ci, concept in enumerate(concepts):
        pi, bi = np.unravel_index(int(np.argmax(band_values[ci])), band_values[ci].shape)
        hits.append(
            ConceptContrast(
                concept=concept,
                score=float(band_values[ci, pi, bi]),
                best_position=positions[pi],
                best_layer=int(band_layers[bi]),
                trace=normalized[ci],
            )
        )
    hits.sort(key=lambda h: h.score, reverse=True)

    return ContrastResult(
        working_prompt=working_prompt,
        failing_prompt=failing_prompt,
        layers=layers,
        band=band,
        hits=hits,
    )
