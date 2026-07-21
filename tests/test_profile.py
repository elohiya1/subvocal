import numpy as np
import pytest

from subvocal.lens import StubLens
from subvocal.profile import Boundaries, Profile


def make_boundaries(spread: bool = False) -> Boundaries:
    return Boundaries.from_signals(
        topk_accuracy=40.0,
        kurtosis=42.0,
        autocorrelation=38.0,
        effective_dim=41.0,
        cka=39.0 if not spread else 60.0,
    )


def make_profile(
    positions=(0, 1, 2),
    layers=(0, 12, 24),
    max_k=5,
    concepts=("dog", "cat"),
    boundaries=None,
) -> Profile:
    n_pos, n_layer = len(positions), len(layers)
    rng = np.random.default_rng(0)
    occupancy = rng.random((n_pos, n_layer)).astype(np.float32) * 25
    fve = rng.random(n_layer).astype(np.float32) * 0.1
    topk_tokens = np.array(
        [
            [[f"tok{p}_{l}_{i}" for i in range(max_k)] for l in range(n_layer)]
            for p in range(n_pos)
        ],
        dtype=object,
    )
    topk_scores = np.sort(rng.random((n_pos, n_layer, max_k)).astype(np.float32))[
        :, :, ::-1
    ]
    loadings = {
        c: rng.random((n_pos, n_layer)).astype(np.float32) for c in concepts
    }
    return Profile(
        positions=list(positions),
        layers=list(layers),
        occupancy=occupancy,
        fve=fve,
        boundaries=boundaries or make_boundaries(),
        topk_tokens=topk_tokens,
        topk_scores=topk_scores,
        loadings=loadings,
    )


class TestBoundaries:
    def test_no_disagreement_within_threshold(self):
        b = make_boundaries(spread=False)
        assert b.disagreement is False

    def test_disagreement_beyond_threshold(self):
        b = make_boundaries(spread=True)
        assert b.disagreement is True

    def test_as_dict_roundtrips_into_constructor(self):
        b = make_boundaries()
        b2 = Boundaries(**b.as_dict())
        assert b2 == b


class TestProfileAccessors:
    def test_loading_shape_and_values(self):
        profile = make_profile()
        loading = profile.loading("dog")
        assert loading.shape == (3, 3)

    def test_loading_unknown_concept_raises(self):
        profile = make_profile()
        with pytest.raises(KeyError):
            profile.loading("nonexistent")

    def test_loading_returns_copy(self):
        profile = make_profile()
        loading = profile.loading("dog")
        loading[0, 0] = 999.0
        assert profile.loading("dog")[0, 0] != 999.0

    def test_occupancy_shape_and_copy(self):
        profile = make_profile()
        occ = profile.occupancy()
        assert occ.shape == (3, 3)
        occ[0, 0] = -1.0
        assert profile.occupancy()[0, 0] != -1.0

    def test_boundaries_passthrough(self):
        b = make_boundaries()
        profile = make_profile(boundaries=b)
        assert profile.boundaries() == b

    def test_fve_shape_and_copy(self):
        profile = make_profile()
        fve = profile.fve()
        assert fve.shape == (3,)
        fve[0] = -1.0
        assert profile.fve()[0] != -1.0

    def test_topk_default_k(self):
        profile = make_profile(max_k=25)
        top = profile.topk(0, 12, k=25)
        assert len(top) == 25

    def test_topk_respects_k(self):
        profile = make_profile(max_k=5)
        top = profile.topk(1, 24, k=3)
        assert len(top) == 3
        assert all(isinstance(tok, str) and isinstance(score, float) for tok, score in top)

    def test_topk_scores_descending(self):
        profile = make_profile()
        top = profile.topk(0, 0, k=5)
        scores = [s for _, s in top]
        assert scores == sorted(scores, reverse=True)

    def test_topk_k_exceeds_max_k_raises(self):
        profile = make_profile(max_k=5)
        with pytest.raises(ValueError):
            profile.topk(0, 0, k=6)

    def test_topk_unknown_position_raises(self):
        profile = make_profile()
        with pytest.raises(KeyError):
            profile.topk(99, 0, k=1)

    def test_topk_unknown_layer_raises(self):
        profile = make_profile()
        with pytest.raises(KeyError):
            profile.topk(0, 99, k=1)

    def test_topk_uses_actual_layer_values_not_offsets(self):
        # layers=(0, 12, 24): layer=1 is not a stored layer value even though
        # it would be a valid array offset.
        profile = make_profile()
        with pytest.raises(KeyError):
            profile.topk(0, 1, k=1)
        profile.topk(0, 12, k=1)  # does not raise


class TestProfileValidation:
    def test_duplicate_positions_rejected(self):
        with pytest.raises(ValueError):
            make_profile(positions=(0, 0, 1))

    def test_duplicate_layers_rejected(self):
        with pytest.raises(ValueError):
            make_profile(layers=(0, 0, 1))

    def test_occupancy_shape_mismatch_rejected(self):
        with pytest.raises(ValueError):
            Profile(
                positions=[0, 1],
                layers=[0, 1, 2],
                occupancy=np.zeros((2, 2)),  # wrong: should be (2, 3)
                fve=np.zeros(3),
                boundaries=make_boundaries(),
                topk_tokens=np.empty((2, 3, 1), dtype=object),
                topk_scores=np.zeros((2, 3, 1)),
            )

    def test_fve_shape_mismatch_rejected(self):
        with pytest.raises(ValueError):
            Profile(
                positions=[0, 1],
                layers=[0, 1, 2],
                occupancy=np.zeros((2, 3)),
                fve=np.zeros(2),  # wrong: should be (3,)
                boundaries=make_boundaries(),
                topk_tokens=np.empty((2, 3, 1), dtype=object),
                topk_scores=np.zeros((2, 3, 1)),
            )

    def test_topk_tokens_scores_shape_mismatch_rejected(self):
        with pytest.raises(ValueError):
            Profile(
                positions=[0, 1],
                layers=[0, 1, 2],
                occupancy=np.zeros((2, 3)),
                fve=np.zeros(3),
                boundaries=make_boundaries(),
                topk_tokens=np.empty((2, 3, 1), dtype=object),
                topk_scores=np.zeros((2, 3, 2)),  # wrong: max_k mismatch
            )

    def test_loading_shape_mismatch_rejected(self):
        with pytest.raises(ValueError):
            Profile(
                positions=[0, 1],
                layers=[0, 1, 2],
                occupancy=np.zeros((2, 3)),
                fve=np.zeros(3),
                boundaries=make_boundaries(),
                topk_tokens=np.empty((2, 3, 1), dtype=object),
                topk_scores=np.zeros((2, 3, 1)),
                loadings={"dog": np.zeros((2, 2))},  # wrong shape
            )


class TestProfileSerialization:
    def test_save_load_roundtrip(self, tmp_path):
        profile = make_profile()
        path = tmp_path / "profile"
        profile.save(str(path))
        loaded = Profile.load(str(path))

        assert loaded.positions == profile.positions
        assert loaded.layers == profile.layers
        assert loaded.boundaries() == profile.boundaries()
        np.testing.assert_array_equal(loaded.occupancy(), profile.occupancy())
        np.testing.assert_array_equal(loaded.fve(), profile.fve())
        np.testing.assert_array_equal(loaded.loading("dog"), profile.loading("dog"))
        np.testing.assert_array_equal(loaded.loading("cat"), profile.loading("cat"))
        assert loaded.topk(0, 12, k=5) == profile.topk(0, 12, k=5)

    def test_save_appends_npz_extension(self, tmp_path):
        profile = make_profile()
        path = tmp_path / "profile_no_ext"
        profile.save(str(path))
        assert (tmp_path / "profile_no_ext.npz").exists()

    def test_save_load_roundtrip_no_loadings(self, tmp_path):
        profile = make_profile(concepts=())
        path = tmp_path / "profile_empty_loadings"
        profile.save(str(path))
        loaded = Profile.load(str(path))
        with pytest.raises(KeyError):
            loaded.loading("dog")
        np.testing.assert_array_equal(loaded.occupancy(), profile.occupancy())


class TestProfileWithStubLens:
    def test_profile_built_from_stub_lens_readouts(self):
        lens = StubLens(n_layers=8, d_model=16, vocab_size=10)
        prompt = "the quick brown fox"
        positions = [0, 1, 2, 3]
        layers = [0, 3, 7]
        k = 5

        topk_tokens = np.empty((len(positions), len(layers), k), dtype=object)
        topk_scores = np.zeros((len(positions), len(layers), k), dtype=np.float32)
        for pi, pos in enumerate(positions):
            for li, layer in enumerate(layers):
                for ki, (tok, score) in enumerate(lens.topk(prompt, pos, layer, k=k)):
                    topk_tokens[pi, li, ki] = tok
                    topk_scores[pi, li, ki] = score

        profile = Profile(
            positions=positions,
            layers=layers,
            occupancy=np.zeros((len(positions), len(layers)), dtype=np.float32),
            fve=np.zeros(len(layers), dtype=np.float32),
            boundaries=make_boundaries(),
            topk_tokens=topk_tokens,
            topk_scores=topk_scores,
        )

        assert profile.topk(2, 3, k=5) == lens.topk(prompt, 2, 3, k=5)
