"""Shared fixtures for real-``FittedLens`` integration tests.

``tiny_fitted_lens``: a tiny, randomly initialized ``Qwen3ForCausalLM`` (no
network weights beyond the gpt2 tokenizer) with a real ``jlens.fit()`` run
over a couple of short prompts. Exercises real ``jlens.from_hf`` layout
detection and real forward-pass hooking end to end -- not the paper's actual
Qwen3.5-4B weights (too large to fetch in CI), so it validates plumbing, not
CLAUDE.md's real-lens sanity checks or M2/M4's real-lens findings, which need
the real pinned model.
"""

from __future__ import annotations

import jlens
import pytest
import torch
import transformers

from subvocal.lens import FittedLens

PROMPTS = [
    "The quick brown fox jumps over the lazy dog near the river bank today.",
    "Scientists discovered a new species of beetle living deep within the cave system.",
]


@pytest.fixture(scope="module")
def tiny_fitted_lens() -> FittedLens:
    torch.manual_seed(0)
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    config = transformers.Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
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
        source_layers=[0, 1],
        target_layer=2,
        dim_batch=4,
        max_seq_len=32,
        skip_first=2,
        checkpoint_every=None,
    )
    return FittedLens(model, tokenizer, jacobian_lens, device="cpu")
