"""Tests for metrics.py.

Deliberately does NOT assert the three real-lens-only sanity checks from
CLAUDE.md (occupancy near zero in the first third of layers / plateaus near
25 mid-band; fve never exceeds 0.10; the five boundary signals agree within
10% of depth) — those are empirical claims about a real fitted lens, and
StubLens's fake random dictionaries have no reason to satisfy them. Those
checks get run and reported once a real lens exists.
"""

import numpy as np
import pytest

from subvocal.lens import StubLens
from subvocal.metrics import (
    boundary_from_curve,
    build_profile,
    cka_signal,
    concept_dictionary,
    cosine_loading,
    effective_dim_signal,
    fve_from_residuals,
    fve_per_layer,
    gradient_pursuit,
    kurtosis_signal,
    linear_cka,
    loading_grid,
    loading_trace,
    occupancy_from_residuals,
    occupancy_grid,
    random_dictionary,
    reindex_to_depth,
    subsample_layers,
    topk_accuracy_signal,
    variance_explained,
)
from subvocal.profile import Boundaries


# --------------------------------------------------------------------------
# Layer subsampling
# --------------------------------------------------------------------------


class TestSubsampleLayers:
    def test_returns_all_layers_when_fewer_than_n(self):
        assert subsample_layers(10, n=25) == list(range(10))

    def test_subsamples_to_n_when_more_layers(self):
        layers = subsample_layers(100, n=25)
        assert len(layers) == 25
        assert layers[0] == 0
        assert layers[-1] == 99
        assert layers == sorted(layers)

    def test_rejects_zero_layers(self):
        with pytest.raises(ValueError):
            subsample_layers(0)


class TestReindexToDepth:
    def test_first_layer_is_zero(self):
        assert reindex_to_depth(0, 25) == 0.0

    def test_last_layer_is_hundred(self):
        assert reindex_to_depth(24, 25) == 100.0

    def test_midpoint(self):
        assert reindex_to_depth(12, 25) == pytest.approx(50.0)

    def test_single_layer_model(self):
        assert reindex_to_depth(0, 1) == 0.0


# --------------------------------------------------------------------------
# Dictionaries
# --------------------------------------------------------------------------


class TestDictionaries:
    def test_concept_dictionary_shape_default_vocab(self):
        lens = StubLens(n_layers=4, d_model=16, vocab_size=8)
        concepts, D = concept_dictionary(lens, layer=1)
        assert concepts == lens.vocab
        assert D.shape == (8, 16)

    def test_concept_dictionary_custom_concepts(self):
        lens = StubLens(n_layers=4, d_model=16, vocab_size=8)
        concepts, D = concept_dictionary(lens, layer=1, concepts=["dog", "cat"])
        assert concepts == ["dog", "cat"]
        assert D.shape == (2, 16)

    def test_random_dictionary_shape_and_unit_norm(self):
        D = random_dictionary(16, 5, seed=0)
        assert D.shape == (5, 16)
        norms = np.linalg.norm(D, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_random_dictionary_deterministic(self):
        a = random_dictionary(16, 5, seed=7)
        b = random_dictionary(16, 5, seed=7)
        np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


class TestCosineLoading:
    def test_parallel_vectors(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_loading(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_antiparallel_vectors(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_loading(v, -v) == pytest.approx(-1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_loading(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_zero_vector_does_not_raise(self):
        a = np.zeros(3, dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_loading(a, b) == 0.0

    def test_batched(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        b = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        result = cosine_loading(a, b)
        np.testing.assert_allclose(result, [1.0, 0.0], atol=1e-5)


class TestLoadingGridAndTrace:
    def test_loading_grid_shape(self):
        lens = StubLens(n_layers=8, d_model=16, vocab_size=8)
        grid = loading_grid(lens, "a b c", "tok0", positions=[0, 1, 2], layers=[0, 3, 7])
        assert grid.shape == (3, 3)
        assert np.all(grid >= -1.0 - 1e-5) and np.all(grid <= 1.0 + 1e-5)

    def test_loading_grid_default_positions_and_layers(self):
        lens = StubLens(n_layers=8, d_model=16, vocab_size=8)
        grid = loading_grid(lens, "a b c d", "tok0")
        assert grid.shape == (4, 8)

    def test_loading_trace_averages_over_positions(self):
        lens = StubLens(n_layers=8, d_model=16, vocab_size=8)
        grid = loading_grid(lens, "a b c", "tok0", positions=[0, 1, 2], layers=[0, 3, 7])
        trace = loading_trace(lens, "a b c", "tok0", positions=[0, 1, 2], layers=[0, 3, 7])
        np.testing.assert_allclose(trace, grid.mean(axis=0))

    def test_loading_trace_rejects_empty_positions(self):
        lens = StubLens()
        with pytest.raises(ValueError):
            loading_trace(lens, "a b c", "tok0", positions=[])


# --------------------------------------------------------------------------
# Gradient Pursuit
# --------------------------------------------------------------------------


def orthonormal_dictionary(n_atoms: int, d_model: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((d_model, d_model))
    q, _ = np.linalg.qr(m)
    return q[:n_atoms].astype(np.float32)


class TestGradientPursuit:
    def test_recovers_known_sparse_combination(self):
        D = orthonormal_dictionary(10, 10)
        h = 0.7 * D[2] + 0.3 * D[5]
        c, active = gradient_pursuit(h[None, :], D, k=2)
        assert set(np.nonzero(active[0])[0]) == {2, 5}
        recon = c @ D
        np.testing.assert_allclose(recon[0], h, atol=1e-4)

    def test_coefficients_always_nonnegative(self):
        D = orthonormal_dictionary(10, 10)
        h = -0.7 * D[2] - 0.3 * D[5]  # best fit is negative; must not go negative
        c, _ = gradient_pursuit(h[None, :], D, k=3)
        assert np.all(c >= 0.0)

    def test_rejects_non_2d_h(self):
        D = orthonormal_dictionary(4, 4)
        with pytest.raises(ValueError):
            gradient_pursuit(D[0], D, k=1)

    def test_per_row_k_array(self):
        D = orthonormal_dictionary(10, 10)
        h1 = 0.9 * D[0]
        h2 = 0.6 * D[1] + 0.4 * D[2]
        h = np.stack([h1, h2])
        c, active = gradient_pursuit(h, D, k=np.array([1, 2]))
        assert active[0].sum() <= 1
        assert active[1].sum() <= 2

    def test_k_zero_yields_zero_reconstruction(self):
        D = orthonormal_dictionary(4, 4)
        h = D[0][None, :]
        c, active = gradient_pursuit(h, D, k=0)
        assert not active.any()
        np.testing.assert_array_equal(c, np.zeros_like(c))


class TestVarianceExplained:
    def test_exact_reconstruction_is_one(self):
        h = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        assert variance_explained(h, h)[0] == pytest.approx(1.0, abs=1e-5)

    def test_zero_reconstruction_is_zero(self):
        h = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        recon = np.zeros_like(h)
        assert variance_explained(h, recon)[0] == pytest.approx(0.0, abs=1e-5)

    def test_improves_monotonically_with_more_atoms(self):
        D = orthonormal_dictionary(10, 10)
        h = 0.6 * D[0] + 0.3 * D[1] + 0.1 * D[2]
        ve_prev = 0.0
        for k in range(1, 4):
            c, _ = gradient_pursuit(h[None, :], D, k=k)
            ve = variance_explained(h[None, :], c @ D)[0]
            assert ve >= ve_prev - 1e-6
            ve_prev = ve


class TestOccupancyFromResiduals:
    def test_shape_and_bounds(self):
        D = orthonormal_dictionary(20, 10)
        control = random_dictionary(10, 20, seed=1)
        h = np.stack([0.8 * D[0] + 0.2 * D[1], 0.5 * D[3]])
        occ = occupancy_from_residuals(h, D, control, k_max=10)
        assert occ.shape == (2,)
        assert np.all(occ >= 1) and np.all(occ <= 10)

    def test_sparse_signal_beats_pure_noise(self):
        # A position whose residual is a clean combination of a few real
        # dictionary atoms should get *no more* occupancy than a position
        # that is pure random noise unrelated to the dictionary.
        D = orthonormal_dictionary(30, 10)
        control = random_dictionary(10, 30, seed=2)
        rng = np.random.default_rng(3)
        clean = 0.9 * D[0] + 0.1 * D[1]
        noise = rng.standard_normal(10).astype(np.float32)
        noise /= np.linalg.norm(noise)
        h = np.stack([clean, noise])
        occ = occupancy_from_residuals(h, D, control, k_max=20)
        assert occ[0] <= occ[1] + 1  # allow a little slack for the heuristic


class TestFveFromResiduals:
    def test_shape_and_reasonable_bounds(self):
        D = orthonormal_dictionary(10, 10)
        control = random_dictionary(10, 10, seed=4)
        h = np.stack([D[0], D[1]])
        excess = fve_from_residuals(h, D, control, k=1)
        assert excess.shape == (2,)
        assert np.all(excess <= 1.0 + 1e-5)


# --------------------------------------------------------------------------
# Boundary signals
# --------------------------------------------------------------------------


class TestBoundaryFromCurve:
    def test_rising_step_function(self):
        depths = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        values = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        boundary = boundary_from_curve(values, depths, edge_n=1)
        assert 25.0 <= boundary <= 50.0

    def test_falling_step_function_is_direction_agnostic(self):
        depths = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        values = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        boundary = boundary_from_curve(values, depths, edge_n=1)
        assert 25.0 <= boundary <= 50.0

    def test_linear_interpolation(self):
        depths = np.array([0.0, 100.0])
        values = np.array([0.0, 1.0])
        boundary = boundary_from_curve(values, depths, edge_n=1)
        assert boundary == pytest.approx(50.0, abs=1e-5)

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError):
            boundary_from_curve(np.zeros(3), np.zeros(4))

    def test_rejects_too_few_layers(self):
        with pytest.raises(ValueError):
            boundary_from_curve(np.zeros(1), np.zeros(1))


class TestLinearCka:
    def test_identical_matrices_give_one(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 8)).astype(np.float32)
        assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-4)

    def test_rejects_mismatched_sample_count(self):
        X = np.zeros((5, 4))
        Y = np.zeros((6, 4))
        with pytest.raises(ValueError):
            linear_cka(X, Y)


class TestBoundarySignalsAgainstStubLens:
    """Shape/type/bounds only -- StubLens is fake data with no reason to
    satisfy the real-lens statistical sanity checks."""

    def test_topk_accuracy_signal_shape_and_bounds(self):
        lens = StubLens(n_layers=8, d_model=16, vocab_size=8)
        acc = topk_accuracy_signal(lens, ["a b c", "d e f"], k=3, layers=[0, 3, 7])
        assert acc.shape == (3,)
        assert np.all(acc >= 0.0) and np.all(acc <= 1.0)

    def test_kurtosis_signal_shape_and_finite(self):
        lens = StubLens(n_layers=8, d_model=16, vocab_size=8)
        kurt = kurtosis_signal(lens, ["a b c", "d e f"], layers=[0, 3, 7])
        assert kurt.shape == (3,)
        assert np.all(np.isfinite(kurt))

    def test_effective_dim_signal_shape_and_bounds(self):
        lens = StubLens(n_layers=8, d_model=16, vocab_size=8)
        eff = effective_dim_signal(lens, layers=[0, 3, 7])
        assert eff.shape == (3,)
        assert np.all(eff > 0.0) and np.all(eff <= 1.0)

    def test_cka_signal_shape_and_bounds(self):
        lens = StubLens(n_layers=8, d_model=16, vocab_size=8)
        cka = cka_signal(lens, layers=[0, 3, 7])
        assert cka.shape == (3,)
        assert np.all(cka >= -1.0 - 1e-5) and np.all(cka <= 1.0 + 1e-5)


# --------------------------------------------------------------------------
# Profile builder integration
# --------------------------------------------------------------------------


class TestBuildProfile:
    def test_build_profile_shapes_match_interface(self):
        lens = StubLens(n_layers=8, d_model=16, vocab_size=10)
        profile = build_profile(
            lens,
            "the quick brown fox",
            concepts=["tok0", "tok1"],
            layers=[0, 3, 7],
            k_max=5,
            topk_k=5,
        )
        assert profile.positions == [0, 1, 2, 3]
        assert profile.layers == [0, 3, 7]
        assert profile.occupancy().shape == (4, 3)
        assert profile.fve().shape == (3,)
        assert isinstance(profile.boundaries(), Boundaries)
        assert len(profile.topk(0, 3, k=5)) == 5
        assert profile.loading("tok0").shape == (4, 3)

    def test_build_profile_save_load_roundtrip(self, tmp_path):
        lens = StubLens(n_layers=8, d_model=16, vocab_size=10)
        profile = build_profile(
            lens,
            "the quick brown fox",
            concepts=["tok0"],
            layers=[0, 3, 7],
            k_max=5,
            topk_k=5,
        )
        path = tmp_path / "profile"
        profile.save(str(path))
        loaded = profile.load(str(path))
        np.testing.assert_array_equal(loaded.occupancy(), profile.occupancy())
        np.testing.assert_array_equal(loaded.fve(), profile.fve())
        assert loaded.boundaries() == profile.boundaries()
