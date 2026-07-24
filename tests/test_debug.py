"""Tests for debug.py's probe(), contrast(), propose(), verify_ablate(), and
verify_steer().

verify_ablate/verify_steer need a real forward-pass re-run past the
intervened layer, so unlike probe/contrast they're tested against the
tiny_fitted_lens fixture (conftest.py) rather than StubLens or _ToyLens --
see InterventionLens's docstring in debug.py. The tiny model's weights are
randomly initialized, so these tests check structural correctness (shapes,
types, skip logic, that an intervention actually perturbs the forward pass)
rather than asserting any particular causal direction of effect, which isn't
meaningful for an untrained model.
"""

from __future__ import annotations

import numpy as np
import pytest

from subvocal import debug
from subvocal.lens import StubLens


class _ToyLens:
    """A tiny, fully controllable :class:`~subvocal.metrics.MetricsLens` --
    unlike :class:`~subvocal.lens.StubLens`, whose residuals and concept
    directions are independent seeded-random vectors (fine for shape/bounds
    tests, useless for checking that :func:`~subvocal.debug.contrast`
    actually ranks the concept that differs on top).
    """

    n_layers = 2
    vocab = ["target", "other", "noise1", "noise2"]
    d_model = 4

    _DIRS = {
        "target": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "other": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "noise1": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "noise2": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }

    def __init__(self) -> None:
        self._residuals: dict[tuple[str, int, int], np.ndarray] = {}

    def set_residual(self, prompt: str, position: int, layer: int, vec) -> None:
        self._residuals[(prompt, position, layer)] = np.asarray(vec, dtype=np.float32)

    def encode(self, prompt: str) -> list[str]:
        return prompt.split()

    def residual(self, prompt: str, position: int, layer: int) -> np.ndarray:
        return self._residuals.get((prompt, position, layer), np.zeros(self.d_model, dtype=np.float32))

    def concept_direction(self, concept: str, layer: int) -> np.ndarray:
        return self._DIRS[concept]

    def readout(self, prompt: str, position: int, layer: int) -> np.ndarray:
        h = self.residual(prompt, position, layer)
        D = np.stack([self._DIRS[c] for c in self.vocab])
        return D @ h

    def topk(self, prompt: str, position: int, layer: int, k: int = 25):
        scores = self.readout(prompt, position, layer)
        order = np.argsort(-scores)[:k]
        return [(self.vocab[i], float(scores[i])) for i in order]


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------


class TestProbe:
    def test_shape_and_peak_in_range(self):
        lens = StubLens(n_layers=8, d_model=16, vocab_size=8)
        result = debug.probe(lens, "a b c d", "tok0", layers=[0, 3, 7])
        assert result.trace.shape == (3,)
        assert result.depths.shape == (3,)
        assert result.peak_layer in [0, 3, 7]
        assert result.peak_loading == pytest.approx(float(result.trace.max()))

    def test_default_positions_and_layers(self):
        lens = StubLens(n_layers=8, d_model=16, vocab_size=8)
        result = debug.probe(lens, "a b c", "tok0")
        assert len(result.layers) == 8
        assert result.trace.shape == (8,)

    def test_warns_on_multi_token_concept(self):
        lens = StubLens(n_layers=4, d_model=8, vocab_size=8)
        with pytest.warns(UserWarning, match="single token"):
            debug.probe(lens, "a b c", "multi word concept", layers=[0])

    def test_no_warning_on_single_token_concept(self, recwarn):
        lens = StubLens(n_layers=4, d_model=8, vocab_size=8)
        debug.probe(lens, "a b c", "tok0", layers=[0])
        assert len(recwarn) == 0

    def test_peak_matches_known_maximum(self):
        lens = _ToyLens()
        lens.set_residual("a b", 0, 0, [0.0, 1.0, 0, 0])  # orthogonal to "target"
        lens.set_residual("a b", 0, 1, [1.0, 0, 0, 0])  # clean peak at layer 1
        lens.set_residual("a b", 1, 0, [0, 0, 0, 0])
        lens.set_residual("a b", 1, 1, [0, 0, 0, 0])
        result = debug.probe(lens, "a b", "target", positions=[0], layers=[0, 1])
        assert result.peak_layer == 1
        assert result.peak_loading == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------
# contrast
# --------------------------------------------------------------------------


class TestContrast:
    def test_rejects_length_mismatch(self):
        lens = StubLens(n_layers=4, d_model=8, vocab_size=8)
        with pytest.raises(ValueError, match="token-aligned"):
            debug.contrast(lens, "a b c", "a b", layers=[0])

    def test_rejects_band_outside_layers(self):
        lens = StubLens(n_layers=8, d_model=8, vocab_size=4)
        with pytest.raises(ValueError, match="band"):
            debug.contrast(lens, "a b", "c d", layers=[0, 3, 7], band=(100, 200))

    def test_default_concepts_uses_whole_small_vocab(self):
        lens = StubLens(n_layers=4, d_model=8, vocab_size=6)
        result = debug.contrast(lens, "a b", "c d", layers=[0, 3])
        assert {h.concept for h in result.hits} == set(lens.vocab)

    def test_hits_sorted_descending(self):
        lens = StubLens(n_layers=4, d_model=8, vocab_size=6)
        result = debug.contrast(lens, "a b", "c d", layers=[0, 3])
        scores = [h.score for h in result.hits]
        assert scores == sorted(scores, reverse=True)

    def test_ranked_returns_top_k(self):
        lens = StubLens(n_layers=4, d_model=8, vocab_size=6)
        result = debug.contrast(lens, "a b", "c d", layers=[0, 3])
        top2 = result.ranked(2)
        assert [h.concept for h in top2] == [h.concept for h in result.hits[:2]]

    def test_warns_on_multi_token_explicit_concept(self):
        lens = StubLens(n_layers=4, d_model=8, vocab_size=8)
        with pytest.warns(UserWarning, match="single token"):
            debug.contrast(
                lens, "a b", "c d", concepts=["multi word concept"], layers=[0]
            )

    def test_ranks_the_concept_that_actually_differs(self):
        lens = _ToyLens()
        working, failing = "a b c", "d e f"
        for pos in range(3):
            lens.set_residual(working, pos, 1, [1.0, 0.0, 0.0, 0.0])  # aligned with "target"
            lens.set_residual(failing, pos, 1, [0.0, 0.0, 0.0, 0.0])
            lens.set_residual(working, pos, 0, [0.0, 0.0, 0.0, 0.0])
            lens.set_residual(failing, pos, 0, [0.0, 0.0, 0.0, 0.0])

        baseline = "x y z"
        for pos in range(3):
            lens.set_residual(baseline, pos, 1, [0.1, 0.0, 0.0, 0.0])
            lens.set_residual(baseline, pos, 0, [0.0, 0.0, 0.0, 0.0])

        result = debug.contrast(
            lens,
            working,
            failing,
            concepts=["target", "other", "noise1", "noise2"],
            layers=[0, 1],
            baseline_prompts=[baseline],
        )
        assert result.hits[0].concept == "target"
        assert result.hits[0].best_layer == 1
        assert result.hits[0].score > result.hits[1].score
        assert result.hits[0].score > 0

    def test_band_restricts_ranking(self):
        # "target" only differs at layer 1; excluding layer 1 from the band
        # should make it lose the top spot.
        lens = _ToyLens()
        working, failing = "a b", "c d"
        for pos in range(2):
            lens.set_residual(working, pos, 1, [1.0, 0.0, 0.0, 0.0])
            lens.set_residual(failing, pos, 1, [0.0, 0.0, 0.0, 0.0])
            lens.set_residual(working, pos, 0, [0.0, 1.0, 0.0, 0.0])  # "other" differs at layer 0
            lens.set_residual(failing, pos, 0, [0.0, 0.0, 0.0, 0.0])

        baseline = "x y"
        for pos in range(2):
            lens.set_residual(baseline, pos, 0, [0.0, 0.05, 0.0, 0.0])
            lens.set_residual(baseline, pos, 1, [0.05, 0.0, 0.0, 0.0])

        full = debug.contrast(
            lens,
            working,
            failing,
            concepts=["target", "other", "noise1", "noise2"],
            layers=[0, 1],
            baseline_prompts=[baseline],
        )
        assert full.hits[0].concept == "target"

        band_zero_only = debug.contrast(
            lens,
            working,
            failing,
            concepts=["target", "other", "noise1", "noise2"],
            layers=[0, 1],
            band=(0, 0),
            baseline_prompts=[baseline],
        )
        assert band_zero_only.hits[0].concept == "other"


# --------------------------------------------------------------------------
# propose
# --------------------------------------------------------------------------


class TestPropose:
    def test_rejects_band_outside_layers(self):
        lens = StubLens(n_layers=8, d_model=8, vocab_size=4)
        with pytest.raises(ValueError, match="band"):
            debug.propose(lens, "a b c", layers=[0, 3, 7], band=(100, 200))

    def test_default_concepts_uses_whole_small_vocab(self):
        lens = StubLens(n_layers=4, d_model=8, vocab_size=6)
        result = debug.propose(lens, "a b", layers=[0, 3])
        assert {h.concept for h in result.hits} == set(lens.vocab)

    def test_hits_sorted_descending(self):
        lens = StubLens(n_layers=4, d_model=8, vocab_size=6)
        result = debug.propose(lens, "a b", layers=[0, 3])
        scores = [h.score for h in result.hits]
        assert scores == sorted(scores, reverse=True)

    def test_ranked_returns_top_k(self):
        lens = StubLens(n_layers=4, d_model=8, vocab_size=6)
        result = debug.propose(lens, "a b", layers=[0, 3])
        top2 = result.ranked(2)
        assert [h.concept for h in top2] == [h.concept for h in result.hits[:2]]

    def test_warns_on_multi_token_explicit_concept(self):
        lens = StubLens(n_layers=4, d_model=8, vocab_size=8)
        with pytest.warns(UserWarning, match="single token"):
            debug.propose(lens, "a b", concepts=["multi word concept"], layers=[0])

    def test_surfaces_the_concept_unusually_present_here(self):
        # "target" is at its baseline-typical value nowhere special *except*
        # this prompt, where it's exactly aligned -- propose() should still
        # rank it first even though nothing is being diffed against a second
        # prompt.
        lens = _ToyLens()
        prompt = "p"
        lens.set_residual(prompt, 0, 1, [1.0, 0.0, 0.0, 0.0])

        baselines = ["b0", "b1", "b2"]
        baseline_vecs = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        for b, vec in zip(baselines, baseline_vecs):
            lens.set_residual(b, 0, 1, vec)

        result = debug.propose(
            lens,
            prompt,
            concepts=["target", "other", "noise1", "noise2"],
            layers=[0, 1],
            baseline_prompts=baselines,
        )
        assert result.hits[0].concept == "target"
        assert result.hits[0].best_layer == 1
        assert result.hits[0].score > 0
        # "noise2" never varies across baseline or this prompt (always
        # orthogonal) -- std_floor should keep its z-score at exactly 0,
        # not blow up.
        noise2 = next(h for h in result.hits if h.concept == "noise2")
        assert noise2.score == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------
# verify_ablate / verify_steer
# --------------------------------------------------------------------------

_PROMPT = "The quick brown fox jumps over the lazy dog near the river bank today."


class TestVerifyAblate:
    def test_excludes_final_layer_by_default(self, tiny_fitted_lens):
        result = debug.verify_ablate(tiny_fitted_lens, _PROMPT, " fox")
        assert tiny_fitted_lens.n_layers - 1 not in result.layers
        assert result.layers == [0, 1]

    def test_skips_when_concept_in_clean_topk(self, tiny_fitted_lens):
        final_layer = tiny_fitted_lens.n_layers - 1
        position = len(tiny_fitted_lens.encode(_PROMPT)) - 1
        top1 = tiny_fitted_lens.topk(_PROMPT, position, final_layer, k=1)[0][0]

        result = debug.verify_ablate(tiny_fitted_lens, _PROMPT, top1, top_k_skip=10)
        assert result.skipped is True
        assert result.skip_reason is not None
        assert result.concept_outcome is None
        assert result.control_outcome is None

    def test_runs_and_reports_control_side_by_side(self, tiny_fitted_lens):
        final_layer = tiny_fitted_lens.n_layers - 1
        position = len(tiny_fitted_lens.encode(_PROMPT)) - 1
        # Pick a concept guaranteed *not* to be skipped: the lowest-ranked
        # single token in the clean readout.
        low_rank_token = tiny_fitted_lens.topk(
            _PROMPT, position, final_layer, k=len(tiny_fitted_lens.vocab)
        )[-1][0]

        result = debug.verify_ablate(
            tiny_fitted_lens, _PROMPT, low_rank_token, top_k_skip=10
        )
        assert result.skipped is False
        assert isinstance(result.concept_outcome, debug.AblationOutcome)
        assert isinstance(result.control_outcome, debug.AblationOutcome)
        # A real ablation perturbs the forward pass; the original top-1
        # token's logit essentially never lands back exactly on its clean
        # value by coincidence.
        assert result.concept_outcome.top1_logit_after != pytest.approx(
            result.concept_outcome.top1_logit_before, abs=1e-6
        )
        assert result.control_outcome.top1_logit_after != pytest.approx(
            result.control_outcome.top1_logit_before, abs=1e-6
        )

    def test_warns_on_multi_token_concept(self, tiny_fitted_lens):
        with pytest.warns(UserWarning, match="single token"):
            with pytest.raises(ValueError, match="tokenizes to"):
                debug.verify_ablate(tiny_fitted_lens, _PROMPT, "multi token phrase")


class TestVerifySteer:
    def test_excludes_final_layer_by_default(self, tiny_fitted_lens):
        result = debug.verify_steer(tiny_fitted_lens, _PROMPT, " fox", alpha=5.0)
        assert tiny_fitted_lens.n_layers - 1 not in result.layers
        assert result.layers == [0, 1]

    def test_runs_and_reports_control_side_by_side(self, tiny_fitted_lens):
        result = debug.verify_steer(tiny_fitted_lens, _PROMPT, " fox", alpha=10.0, k=5)
        assert isinstance(result.concept_outcome, debug.SteerOutcome)
        assert isinstance(result.control_outcome, debug.SteerOutcome)
        assert result.concept_outcome.rank_before >= 0
        assert result.concept_outcome.rank_after >= 0
        assert result.concept_outcome.logit_after != pytest.approx(
            result.concept_outcome.logit_before, abs=1e-6
        )

    def test_entered_topk_matches_rank(self, tiny_fitted_lens):
        result = debug.verify_steer(tiny_fitted_lens, _PROMPT, " fox", alpha=10.0, k=5)
        assert result.concept_outcome.entered_topk == (result.concept_outcome.rank_after < 5)
        assert result.control_outcome.entered_topk == (result.control_outcome.rank_after < 5)

    def test_warns_on_multi_token_concept(self, tiny_fitted_lens):
        with pytest.warns(UserWarning, match="single token"):
            with pytest.raises(ValueError, match="tokenizes to"):
                debug.verify_steer(tiny_fitted_lens, _PROMPT, "multi token phrase", alpha=1.0)
