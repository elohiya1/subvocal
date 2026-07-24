"""Integration tests for :class:`~subvocal.lens.FittedLens`.

Uses a tiny, randomly initialized ``Qwen3ForCausalLM`` (no network weights,
just the gpt2 tokenizer) and a real ``jlens.fit()`` run over a couple of
short prompts. This exercises the real ``jlens.from_hf`` layout detection and
the ``FittedLens`` wiring end-to-end -- not the paper's actual Qwen3.5-4B
weights (too large to fetch in CI), so it validates plumbing, not the
CLAUDE.md sanity checks, which require the real pinned model.
"""

from __future__ import annotations

import jlens
import pytest
import torch
import transformers

from subvocal import metrics
from subvocal.lens import FittedLens

# tiny_fitted_lens (the fixture built from these) lives in conftest.py, shared
# with test_debug.py's M4 tests; PROMPTS is duplicated here rather than
# imported cross-module, since it's only ever used as inline literal text.
PROMPTS = [
    "The quick brown fox jumps over the lazy dog near the river bank today.",
    "Scientists discovered a new species of beetle living deep within the cave system.",
]


class TestFittedLens:
    def test_shapes_match_hf_config(self, tiny_fitted_lens):
        assert tiny_fitted_lens.n_layers == 3
        assert tiny_fitted_lens.d_model == 16
        assert tiny_fitted_lens.fitted_layers == [0, 1]

    def test_encode_matches_real_tokenization(self, tiny_fitted_lens):
        toks = tiny_fitted_lens.encode(PROMPTS[0])
        assert len(toks) > 5
        assert all(isinstance(t, str) for t in toks)

    def test_residual_shape_and_caching(self, tiny_fitted_lens):
        h1 = tiny_fitted_lens.residual(PROMPTS[0], 3, 0)
        assert h1.shape == (16,)
        assert PROMPTS[0] in tiny_fitted_lens._residual_cache
        h2 = tiny_fitted_lens.residual(PROMPTS[0], 3, 0)
        import numpy as np

        np.testing.assert_array_equal(h1, h2)

    def test_residual_works_on_unfitted_layer(self, tiny_fitted_lens):
        h = tiny_fitted_lens.residual(PROMPTS[0], 3, 2)
        assert h.shape == (16,)

    def test_readout_shape_on_fitted_layer(self, tiny_fitted_lens):
        scores = tiny_fitted_lens.readout(PROMPTS[0], 3, 0)
        assert scores.shape == (len(tiny_fitted_lens.vocab),)

    def test_readout_on_final_layer_is_untransported(self, tiny_fitted_lens):
        final = tiny_fitted_lens.n_layers - 1
        scores = tiny_fitted_lens.readout(PROMPTS[0], 3, final)
        assert scores.shape == (len(tiny_fitted_lens.vocab),)

    def test_readout_on_unfitted_non_final_layer_raises(self):
        # A lens fitted only at layer 0 (target=3) leaves layer 1 unfitted
        # and non-final -- the case tiny_fitted_lens's [0, 1]-fitted, 3-layer
        # setup can't produce, since every layer there is either fitted or
        # the final layer.
        torch.manual_seed(0)
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        config = transformers.Qwen3Config(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
        model = transformers.Qwen3ForCausalLM(config)
        lens_model = jlens.from_hf(model, tokenizer)
        jacobian_lens = jlens.fit(
            lens_model,
            PROMPTS,
            source_layers=[0],
            target_layer=3,
            dim_batch=4,
            max_seq_len=32,
            skip_first=2,
            checkpoint_every=None,
        )
        gapped_lens = FittedLens(model, tokenizer, jacobian_lens, device="cpu")
        with pytest.raises(ValueError, match="no fitted Jacobian"):
            gapped_lens.readout(PROMPTS[0], 3, 1)

    def test_topk_length_and_ordering(self, tiny_fitted_lens):
        top = tiny_fitted_lens.topk(PROMPTS[0], 3, 0, k=5)
        assert len(top) == 5
        scores = [s for _, s in top]
        assert scores == sorted(scores, reverse=True)

    def test_concept_direction_single_token(self, tiny_fitted_lens):
        toks = tiny_fitted_lens.encode(PROMPTS[0])
        direction = tiny_fitted_lens.concept_direction(toks[3], 0)
        assert direction.shape == (16,)

    def test_concept_direction_multi_token_raises(self, tiny_fitted_lens):
        with pytest.raises(ValueError, match="tokenizes to"):
            tiny_fitted_lens.concept_direction("the quick brown", 0)

    def test_concept_direction_out_of_range_layer_raises(self, tiny_fitted_lens):
        with pytest.raises(ValueError, match="out of range"):
            tiny_fitted_lens.concept_direction(" fox", 99)

    def test_direction_cache_is_lru_bounded(self, tiny_fitted_lens):
        # Real-vocab direction matrices are ~2.5GB apiece (Qwen3.5-4B); an
        # unbounded per-layer cache would blow past a dev machine's RAM
        # sweeping the paper's 25 subsampled layers. Default bound is 1.
        assert tiny_fitted_lens._max_cached_direction_layers == 1
        tiny_fitted_lens.concept_direction(" fox", 0)
        assert list(tiny_fitted_lens._direction_cache) == [0]
        tiny_fitted_lens.concept_direction(" fox", 1)
        assert list(tiny_fitted_lens._direction_cache) == [1]

    def test_clear_cache_resets_residuals(self, tiny_fitted_lens):
        tiny_fitted_lens.residual(PROMPTS[1], 0, 0)
        assert PROMPTS[1] in tiny_fitted_lens._residual_cache
        tiny_fitted_lens.clear_cache()
        assert tiny_fitted_lens._residual_cache == {}
        assert tiny_fitted_lens._direction_cache == {}


class TestFittedLensWithMetrics:
    def test_loading_grid(self, tiny_fitted_lens):
        grid = metrics.loading_grid(
            tiny_fitted_lens, PROMPTS[0], " fox", layers=tiny_fitted_lens.fitted_layers
        )
        n_pos = len(tiny_fitted_lens.encode(PROMPTS[0]))
        assert grid.shape == (n_pos, 2)

    def test_occupancy_and_fve(self, tiny_fitted_lens):
        occ = metrics.occupancy_grid(
            tiny_fitted_lens,
            PROMPTS[0],
            layers=tiny_fitted_lens.fitted_layers,
            k_max=10,
        )
        n_pos = len(tiny_fitted_lens.encode(PROMPTS[0]))
        assert occ.shape == (n_pos, 2)
        fve = metrics.fve_per_layer(
            tiny_fitted_lens,
            PROMPTS[0],
            layers=tiny_fitted_lens.fitted_layers,
            occupancy=occ,
            k_max=10,
        )
        assert fve.shape == (2,)

    def test_build_profile_end_to_end(self, tiny_fitted_lens):
        layers = tiny_fitted_lens.fitted_layers + [tiny_fitted_lens.n_layers - 1]
        profile = metrics.build_profile(
            tiny_fitted_lens,
            PROMPTS[0],
            concepts=[" fox", " dog"],
            layers=layers,
            k_max=10,
            topk_k=5,
            boundary_prompts=PROMPTS,
        )
        n_pos = len(tiny_fitted_lens.encode(PROMPTS[0]))
        assert profile.occupancy().shape == (n_pos, len(layers))
        assert profile.fve().shape == (len(layers),)
        assert profile.loading(" fox").shape == (n_pos, len(layers))
        boundaries = profile.boundaries()
        assert isinstance(boundaries.disagreement, bool)
