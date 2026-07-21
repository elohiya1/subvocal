import numpy as np
import pytest

from subvocal.lens import StubLens, resolve_device


def test_resolve_device_explicit_override():
    assert resolve_device("cpu").type == "cpu"


class TestStubLens:
    def test_residual_shape_and_unit_norm(self):
        lens = StubLens(n_layers=4, d_model=16, vocab_size=8)
        h = lens.residual("hello world", position=0, layer=1)
        assert h.shape == (16,)
        assert h.dtype == np.float32
        assert np.isclose(np.linalg.norm(h), 1.0, atol=1e-5)

    def test_concept_direction_shape_and_unit_norm(self):
        lens = StubLens(n_layers=4, d_model=16, vocab_size=8)
        v = lens.concept_direction("dog", layer=2)
        assert v.shape == (16,)
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)

    def test_readout_shape(self):
        lens = StubLens(n_layers=4, d_model=16, vocab_size=8)
        scores = lens.readout("hello world", position=0, layer=1)
        assert scores.shape == (8,)

    def test_topk_length_and_ordering(self):
        lens = StubLens(n_layers=4, d_model=16, vocab_size=8)
        top = lens.topk("hello world", position=0, layer=1, k=5)
        assert len(top) == 5
        scores = [s for _, s in top]
        assert scores == sorted(scores, reverse=True)
        assert all(tok in lens.vocab for tok, _ in top)

    def test_topk_matches_full_readout_argsort(self):
        lens = StubLens(n_layers=4, d_model=16, vocab_size=8)
        top = lens.topk("hello world", position=0, layer=1, k=8)
        scores = lens.readout("hello world", position=0, layer=1)
        expected_order = [lens.vocab[i] for i in np.argsort(-scores)]
        assert [tok for tok, _ in top] == expected_order

    def test_deterministic_across_calls(self):
        lens = StubLens(n_layers=4, d_model=16, vocab_size=8)
        h1 = lens.residual("hello world", position=0, layer=1)
        h2 = lens.residual("hello world", position=0, layer=1)
        np.testing.assert_array_equal(h1, h2)

    def test_deterministic_across_instances(self):
        a = StubLens(n_layers=4, d_model=16, vocab_size=8, seed=42)
        b = StubLens(n_layers=4, d_model=16, vocab_size=8, seed=42)
        np.testing.assert_array_equal(
            a.residual("p", 0, 1), b.residual("p", 0, 1)
        )

    def test_different_seed_differs(self):
        a = StubLens(n_layers=4, d_model=16, vocab_size=8, seed=1)
        b = StubLens(n_layers=4, d_model=16, vocab_size=8, seed=2)
        assert not np.allclose(a.residual("p", 0, 1), b.residual("p", 0, 1))

    def test_different_position_differs(self):
        lens = StubLens(n_layers=4, d_model=16, vocab_size=8)
        assert not np.allclose(
            lens.residual("p", 0, 1), lens.residual("p", 1, 1)
        )

    def test_layer_out_of_range_raises(self):
        lens = StubLens(n_layers=4, d_model=16, vocab_size=8)
        with pytest.raises(ValueError):
            lens.residual("p", 0, 4)
        with pytest.raises(ValueError):
            lens.concept_direction("dog", -1)

    def test_topk_k_out_of_range_raises(self):
        lens = StubLens(n_layers=4, d_model=16, vocab_size=8)
        with pytest.raises(ValueError):
            lens.topk("p", 0, 0, k=0)
        with pytest.raises(ValueError):
            lens.topk("p", 0, 0, k=9)

    def test_encode_is_whitespace_split(self):
        lens = StubLens()
        assert lens.encode("hello there world") == ["hello", "there", "world"]

    def test_custom_vocab(self):
        lens = StubLens(n_layers=2, d_model=8, vocab=["cat", "dog", "fish"])
        assert lens.vocab == ["cat", "dog", "fish"]
        top = lens.topk("p", 0, 0, k=3)
        assert {tok for tok, _ in top} == {"cat", "dog", "fish"}
