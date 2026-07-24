"""Prompt debugging built on ``metrics.py``: M3's :func:`probe`,
:func:`contrast`, and :func:`propose` (the stretch goal), plus M4's
:func:`verify_ablate` and :func:`verify_steer`.

M3's two functions are typed against :class:`~subvocal.metrics.MetricsLens`
and run against either lens. M4's two are typed against
:class:`InterventionLens` instead: ablation/steering needs a real
forward-pass re-run past the intervened layer, which is fundamentally
something a real model provides and a fake one can't -- see
:meth:`~subvocal.lens.FittedLens.ablate`'s docstring. Only
:class:`~subvocal.lens.FittedLens` implements it.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from subvocal.metrics import (
    DEFAULT_DICTIONARY_SIZE,
    MetricsLens,
    concept_dictionary,
    cosine_loading,
    loading_grid,
    random_dictionary,
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

    Prefers the lens's own ``token_id()`` (when it has one, i.e.
    :class:`~subvocal.lens.FittedLens`) over ``len(lens.encode(concept))``:
    ``encode()`` tokenizes ``concept`` as if it were a whole prompt, which
    for a tokenizer configured to prepend a BOS token would always come back
    as "2 tokens" regardless of ``concept`` -- a false positive that doesn't
    match what ``concept_direction()`` actually enforces
    (``tokenizer.encode(concept, add_special_tokens=False)``). Falls back to
    ``encode()`` for lenses without ``token_id()`` (e.g. :class:`~subvocal.lens.StubLens`,
    which has no real single-token constraint to check against anyway).
    """
    token_id = getattr(lens, "token_id", None)
    if callable(token_id):
        try:
            token_id(concept)
            is_single = True
        except ValueError:
            is_single = False
    else:
        is_single = len(lens.encode(concept)) == 1
    if not is_single:
        warnings.warn(
            f"concept {concept!r} does not resolve to a single token under "
            "this lens's tokenizer; the lens only reads out single tokens, "
            "so this reflects at most the first token's meaning, not the "
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
# Shared by contrast() and propose(): layers/band/concepts resolution and
# baseline-corpus scoring.
# --------------------------------------------------------------------------


def _resolve_layers_and_band(
    lens: MetricsLens, layers: Sequence[int] | None, band: tuple[int, int] | None
) -> tuple[list[int], tuple[int, int], np.ndarray]:
    layers = list(layers) if layers is not None else subsample_layers(lens.n_layers)
    band = band if band is not None else (min(layers), max(layers))
    band_mask = np.array([band[0] <= l <= band[1] for l in layers])
    if not band_mask.any():
        raise ValueError(f"band={band} does not overlap any scanned layer in {layers}")
    return layers, band, band_mask


def _resolve_concepts(
    lens: MetricsLens,
    concepts: Sequence[str] | None,
    layers: Sequence[int],
    max_atoms: int,
    seed: int,
) -> list[str]:
    if concepts is not None:
        concepts = list(concepts)
        for c in concepts:
            _warn_if_multi_token(lens, c)
        return concepts
    names, _ = concept_dictionary(lens, layers[0], max_atoms=max_atoms, seed=seed)
    return names


def _resolve_baseline(
    lens: MetricsLens, baseline_prompts: Sequence[str] | None
) -> tuple[list[str], list[list[int]]]:
    prompts = (
        list(baseline_prompts) if baseline_prompts is not None else DEFAULT_BASELINE_PROMPTS
    )
    positions = [list(range(len(lens.encode(p)))) for p in prompts]
    return prompts, positions


def _pooled_baseline_scores(
    lens: MetricsLens,
    D: np.ndarray,
    layer: int,
    baseline_prompts: Sequence[str],
    baseline_positions: Sequence[Sequence[int]],
) -> np.ndarray:
    """Cosine loading of every atom in ``D`` against every ``(baseline
    prompt, position)`` at ``layer``: shape ``(n_atoms, total_baseline_pos)``.
    Shared by :func:`contrast` (which only needs the spread, per concept per
    layer) and :func:`propose` (which needs the center too, to tell "unusually
    high for this prompt" apart from "just generically high everywhere")."""
    baseline_h = np.concatenate(
        [
            np.stack([lens.residual(p, pos, layer) for pos in pos_list])
            for p, pos_list in zip(baseline_prompts, baseline_positions)
        ],
        axis=0,
    )  # (total_baseline_pos, d_model)
    return cosine_loading(baseline_h[None, :, :], D[:, None, :])


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

    layers, band, band_mask = _resolve_layers_and_band(lens, layers, band)
    concepts = _resolve_concepts(lens, concepts, layers, max_atoms, seed)
    baseline_prompts, baseline_positions = _resolve_baseline(lens, baseline_prompts)

    n_concepts, n_pos, n_layer = len(concepts), len(positions), len(layers)
    normalized = np.zeros((n_concepts, n_pos, n_layer), dtype=np.float32)

    for li, layer in enumerate(layers):
        _, D = concept_dictionary(lens, layer, concepts=concepts)  # (n_concepts, d_model)

        working_h = np.stack([lens.residual(working_prompt, p, layer) for p in positions])
        failing_h = np.stack([lens.residual(failing_prompt, p, layer) for p in positions])
        working_score = cosine_loading(working_h[None, :, :], D[:, None, :])  # (n_concepts, n_pos)
        failing_score = cosine_loading(failing_h[None, :, :], D[:, None, :])

        baseline_score = _pooled_baseline_scores(
            lens, D, layer, baseline_prompts, baseline_positions
        )
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


# --------------------------------------------------------------------------
# propose
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ConceptProposal:
    """One concept's standing-out-ness in a single prompt. ``eq=False``: see
    :class:`ProbeResult`.

    Attributes:
        concept: The concept.
        score: The largest z-score (this prompt's loading, minus the
            baseline corpus's mean loading for this concept, over the
            baseline's std) within the scanned ``band``, across all
            positions and band layers. Positive means loaded higher than
            this concept typically is; this is not a working-vs-failing
            delta the way :attr:`ConceptContrast.score` is, so a negative
            score is a meaningful ("unusually low here") result too, not
            just "not a hit."
        best_position: Position (caller's index) where ``score`` occurs.
        best_layer: Layer (caller's index) where ``score`` occurs.
        trace: The z-score over every scanned position/layer -- not just the
            band -- shape ``(n_pos, n_layer)``.
    """

    concept: str
    score: float
    best_position: int
    best_layer: int
    trace: np.ndarray


@dataclass(frozen=True)
class ProposeResult:
    """Concepts ranked by how much they stand out in one prompt, with no
    user-supplied hypothesis (:func:`probe`) or second prompt to diff against
    (:func:`contrast`).

    Attributes:
        prompt: The prompt scanned.
        layers: Layers scanned.
        band: ``(lo, hi)`` layer-index bounds (inclusive)
            :attr:`ConceptProposal.score` was maximized over.
        hits: Every scanned concept, ranked by :attr:`ConceptProposal.score`
            descending -- most unusually-present first.
    """

    prompt: str
    layers: list[int]
    band: tuple[int, int]
    hits: list[ConceptProposal]

    def ranked(self, k: int = 10) -> list[ConceptProposal]:
        """Top ``k`` concepts by :attr:`ConceptProposal.score`."""
        return self.hits[:k]


def propose(
    lens: MetricsLens,
    prompt: str,
    *,
    concepts: Sequence[str] | None = None,
    layers: Sequence[int] | None = None,
    band: tuple[int, int] | None = None,
    baseline_prompts: Sequence[str] | None = None,
    max_atoms: int = DEFAULT_DICTIONARY_SIZE,
    seed: int = 0,
    std_floor: float = 0.05,
) -> ProposeResult:
    """Automatically surface which concepts stand out in ``prompt``'s
    J-space -- CLAUDE.md's stretch-goal third M3 mode.

    :func:`probe` needs a concept the caller already suspects; :func:`contrast`
    needs a second, token-aligned prompt to diff against. ``propose`` needs
    neither: it scores each concept's loading in ``prompt`` as a z-score
    against that same concept's loading across ``baseline_prompts`` (mean and
    std, floored at ``std_floor``), the same normalization :func:`contrast`
    uses for its delta. This is what keeps a concept that's just generically
    high-loading on *any* prompt (e.g. one whose vocab row happens to have a
    large norm) from drowning out concepts that are unusually present in
    this specific prompt.

    Computed layer-major, same reasoning as :func:`contrast`.

    Args:
        lens: Lens to read out from.
        prompt: The prompt to scan.
        concepts: Concepts to scan. Defaults to the same size-``max_atoms``
            vocab sample :func:`~subvocal.metrics.concept_dictionary` draws
            for occupancy.
        layers: Layers to scan. Defaults to
            :func:`~subvocal.metrics.subsample_layers`.
        band: ``(lo, hi)`` layer-index bounds (inclusive) to rank
            :attr:`ConceptProposal.score` within. Defaults to all of
            ``layers``.
        baseline_prompts: Corpus for the z-score. Defaults to
            :data:`DEFAULT_BASELINE_PROMPTS`.
        max_atoms: Concept-dictionary size when ``concepts`` isn't given.
        seed: Concept-dictionary sampling seed.
        std_floor: Minimum per-concept-per-layer baseline std.

    Returns:
        A :class:`ProposeResult` ranking every scanned concept.

    Raises:
        ValueError: ``band`` doesn't overlap any scanned layer.
    """
    positions = list(range(len(lens.encode(prompt))))

    layers, band, band_mask = _resolve_layers_and_band(lens, layers, band)
    concepts = _resolve_concepts(lens, concepts, layers, max_atoms, seed)
    baseline_prompts, baseline_positions = _resolve_baseline(lens, baseline_prompts)

    n_concepts, n_pos, n_layer = len(concepts), len(positions), len(layers)
    z_scores = np.zeros((n_concepts, n_pos, n_layer), dtype=np.float32)

    for li, layer in enumerate(layers):
        _, D = concept_dictionary(lens, layer, concepts=concepts)  # (n_concepts, d_model)

        h = np.stack([lens.residual(prompt, p, layer) for p in positions])
        score = cosine_loading(h[None, :, :], D[:, None, :])  # (n_concepts, n_pos)

        baseline_score = _pooled_baseline_scores(
            lens, D, layer, baseline_prompts, baseline_positions
        )
        baseline_mean = baseline_score.mean(axis=1)  # (n_concepts,)
        baseline_std = np.maximum(baseline_score.std(axis=1), std_floor)

        z_scores[:, :, li] = (score - baseline_mean[:, None]) / baseline_std[:, None]

    band_layers = np.array(layers)[band_mask]
    band_values = z_scores[:, :, band_mask]  # (n_concepts, n_pos, n_band_layer)

    hits = []
    for ci, concept in enumerate(concepts):
        pi, bi = np.unravel_index(int(np.argmax(band_values[ci])), band_values[ci].shape)
        hits.append(
            ConceptProposal(
                concept=concept,
                score=float(band_values[ci, pi, bi]),
                best_position=positions[pi],
                best_layer=int(band_layers[bi]),
                trace=z_scores[ci],
            )
        )
    hits.sort(key=lambda h: h.score, reverse=True)

    return ProposeResult(
        prompt=prompt,
        layers=layers,
        band=band,
        hits=hits,
    )


# --------------------------------------------------------------------------
# verify_ablate / verify_steer (M4)
# --------------------------------------------------------------------------


class InterventionLens(MetricsLens, Protocol):
    """What :func:`verify_ablate`/:func:`verify_steer` need beyond
    :class:`~subvocal.metrics.MetricsLens`: real forward-pass intervention.
    ``d_model`` is needed to size the random-direction control.
    """

    d_model: int

    def token_id(self, token: str) -> int: ...
    def ablate(self, prompt: str, layer_directions: dict[int, np.ndarray]) -> np.ndarray: ...
    def steer(
        self, prompt: str, layer_directions: dict[int, np.ndarray], alpha: float
    ) -> np.ndarray: ...


def _final_position(lens: InterventionLens, prompt: str) -> int:
    return len(lens.encode(prompt)) - 1


def _ablation_layers(lens: InterventionLens, layers: Sequence[int] | None) -> list[int]:
    """Layers to intervene at, per CLAUDE.md's "workspace band" convention --
    with the model's actual final layer always excluded, since intervening
    there is mechanically just editing the logits directly (no residual
    stream left for a later layer to process), not a hidden-state
    intervention at all."""
    layers = list(layers) if layers is not None else subsample_layers(lens.n_layers)
    return [l for l in layers if l != lens.n_layers - 1]


def _rank_of(logits: np.ndarray, token_id: int) -> int:
    """0-indexed rank of ``token_id`` in ``logits`` (0 = top prediction)."""
    return int((logits > logits[token_id]).sum())


@dataclass(frozen=True)
class AblationOutcome:
    """One side (concept or random-direction control) of a
    :func:`verify_ablate` check.

    Attributes:
        top1_before: The model's top prediction before ablation.
        top1_after: Its top prediction after.
        answer_changed: Whether ``top1_before != top1_after``.
        top1_logit_before: ``top1_before``'s logit before ablation.
        top1_logit_after: ``top1_before``'s logit after ablation (how much
            the *original* answer was suppressed, even if it's still top-1).
    """

    top1_before: str
    top1_after: str
    answer_changed: bool
    top1_logit_before: float
    top1_logit_after: float


@dataclass(frozen=True)
class VerifyAblateResult:
    """CLAUDE.md's M4 necessity check, concept vs. required random-direction
    control, always reported side by side.

    Attributes:
        prompt: The prompt checked.
        concept: The concept ablated.
        layers: Layers ablated at (final layer always excluded; see
            :func:`verify_ablate`).
        skipped: True if ablation wasn't run at all.
        skip_reason: Why, when ``skipped``.
        concept_outcome: Effect of ablating ``concept``'s direction. ``None``
            when ``skipped``.
        control_outcome: Effect of ablating a matched random direction at the
            same layers. ``None`` when ``skipped``.
    """

    prompt: str
    concept: str
    layers: list[int]
    skipped: bool
    skip_reason: str | None
    concept_outcome: AblationOutcome | None
    control_outcome: AblationOutcome | None


def verify_ablate(
    lens: InterventionLens,
    prompt: str,
    concept: str,
    *,
    layers: Sequence[int] | None = None,
    top_k_skip: int = 10,
    seed: int = 0,
) -> VerifyAblateResult:
    """Project ``concept``'s direction out of the residual stream across
    ``layers`` and confirm the model's answer actually changes.

    CLAUDE.md's M4 necessity check. Per the paper's convention, skipped
    entirely when ``concept`` is already in the clean forward pass's top-
    ``top_k_skip`` prediction -- ablating there would just remove the token
    from its own imminent output (the "report"), which isn't a test of
    whether the model used the concept in internal reasoning. Always runs
    and reports a random-direction control matched for norm (unit vector,
    like the real direction) and layer band, per CLAUDE.md: "Without it any
    effect could be generic perturbation damage."

    Args:
        lens: An :class:`InterventionLens` (only :class:`~subvocal.lens.FittedLens`
            qualifies).
        prompt: The prompt to check.
        concept: The concept to ablate.
        layers: Layers to ablate at. Defaults to
            :func:`~subvocal.metrics.subsample_layers`, final layer excluded.
        top_k_skip: Skip threshold (CLAUDE.md says "top-10" specifically;
            exposed as a parameter since that's a judgment call, not a
            paper-pinned constant).
        seed: Random-control direction seed.

    Returns:
        A :class:`VerifyAblateResult`.
    """
    _warn_if_multi_token(lens, concept)
    layers = _ablation_layers(lens, layers)
    position = _final_position(lens, prompt)
    final_layer = lens.n_layers - 1

    clean_logits = lens.readout(prompt, position, final_layer)
    clean_top = lens.topk(prompt, position, final_layer, k=top_k_skip)
    if concept in {tok for tok, _ in clean_top}:
        return VerifyAblateResult(
            prompt=prompt,
            concept=concept,
            layers=layers,
            skipped=True,
            skip_reason=(
                f"{concept!r} is already in the clean forward pass's top-"
                f"{top_k_skip} prediction ({[t for t, _ in clean_top]}); "
                "ablating it would only remove it from its own imminent "
                "output, not test internal use."
            ),
            concept_outcome=None,
            control_outcome=None,
        )

    top1_id = lens.token_id(clean_top[0][0])
    top1_before = clean_top[0][0]
    top1_logit_before = float(clean_logits[top1_id])

    def outcome(after_logits: np.ndarray) -> AblationOutcome:
        top1_after_id = int(np.argmax(after_logits))
        top1_after = lens.vocab[top1_after_id]
        return AblationOutcome(
            top1_before=top1_before,
            top1_after=top1_after,
            answer_changed=top1_after != top1_before,
            top1_logit_before=top1_logit_before,
            top1_logit_after=float(after_logits[top1_id]),
        )

    concept_dirs = {l: lens.concept_direction(concept, l) for l in layers}
    concept_after = lens.ablate(prompt, concept_dirs)[position]

    control_vecs = random_dictionary(lens.d_model, len(layers), seed=seed)
    control_dirs = dict(zip(layers, control_vecs))
    control_after = lens.ablate(prompt, control_dirs)[position]

    return VerifyAblateResult(
        prompt=prompt,
        concept=concept,
        layers=layers,
        skipped=False,
        skip_reason=None,
        concept_outcome=outcome(concept_after),
        control_outcome=outcome(control_after),
    )


@dataclass(frozen=True)
class SteerOutcome:
    """One side (concept or random-direction control) of a
    :func:`verify_steer` check.

    Attributes:
        rank_before: ``concept``'s 0-indexed rank in the clean prediction
            (0 = top-1).
        rank_after: Its rank after steering.
        entered_topk: Whether ``rank_after < k`` -- CLAUDE.md's "check
            recovery": did steering bring the concept into the visible
            output at all.
        logit_before: ``concept``'s logit before steering.
        logit_after: Its logit after.
    """

    rank_before: int
    rank_after: int
    entered_topk: bool
    logit_before: float
    logit_after: float


@dataclass(frozen=True)
class VerifySteerResult:
    """CLAUDE.md's M4 sufficiency check, concept vs. required random-direction
    control, always reported side by side.

    Attributes:
        prompt: The prompt checked.
        concept: The concept steered in.
        alpha: Steering magnitude used.
        layers: Layers steered at (final layer always excluded; see
            :func:`verify_ablate`).
        concept_outcome: Effect of steering ``concept``'s direction in.
        control_outcome: Effect of steering a matched random direction in at
            the same layers and ``alpha``.
    """

    prompt: str
    concept: str
    alpha: float
    layers: list[int]
    concept_outcome: SteerOutcome
    control_outcome: SteerOutcome


def verify_steer(
    lens: InterventionLens,
    prompt: str,
    concept: str,
    alpha: float,
    *,
    layers: Sequence[int] | None = None,
    k: int = 25,
    seed: int = 0,
) -> VerifySteerResult:
    """Add ``alpha`` times ``concept``'s direction into the residual stream
    across ``layers`` and check whether it recovers into the visible output.

    CLAUDE.md's M4 sufficiency check, the counterpart to :func:`verify_ablate`'s
    necessity check. Always runs and reports a random-direction control
    matched for norm and layer band, same as :func:`verify_ablate`.

    Args:
        lens: An :class:`InterventionLens` (only :class:`~subvocal.lens.FittedLens`
            qualifies).
        prompt: The prompt to check.
        concept: The concept to steer in.
        alpha: Steering magnitude (added as ``alpha`` times a unit vector at
            each layer).
        layers: Layers to steer at. Defaults to
            :func:`~subvocal.metrics.subsample_layers`, final layer excluded.
        k: Top-k threshold "recovery" (:attr:`SteerOutcome.entered_topk`) is
            judged against.
        seed: Random-control direction seed.

    Returns:
        A :class:`VerifySteerResult`.
    """
    _warn_if_multi_token(lens, concept)
    layers = _ablation_layers(lens, layers)
    position = _final_position(lens, prompt)
    final_layer = lens.n_layers - 1
    concept_id = lens.token_id(concept)

    clean_logits = lens.readout(prompt, position, final_layer)
    rank_before = _rank_of(clean_logits, concept_id)
    logit_before = float(clean_logits[concept_id])

    def outcome(after_logits: np.ndarray) -> SteerOutcome:
        rank_after = _rank_of(after_logits, concept_id)
        return SteerOutcome(
            rank_before=rank_before,
            rank_after=rank_after,
            entered_topk=rank_after < k,
            logit_before=logit_before,
            logit_after=float(after_logits[concept_id]),
        )

    concept_dirs = {l: lens.concept_direction(concept, l) for l in layers}
    concept_after = lens.steer(prompt, concept_dirs, alpha)[position]

    control_vecs = random_dictionary(lens.d_model, len(layers), seed=seed)
    control_dirs = dict(zip(layers, control_vecs))
    control_after = lens.steer(prompt, control_dirs, alpha)[position]

    return VerifySteerResult(
        prompt=prompt,
        concept=concept,
        alpha=alpha,
        layers=layers,
        concept_outcome=outcome(concept_after),
        control_outcome=outcome(control_after),
    )
