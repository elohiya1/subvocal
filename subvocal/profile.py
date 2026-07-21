"""The ``Profile`` object: a frozen snapshot of a prompt's J-space occupancy.

Everything downstream (metrics, debug, report) consumes a ``Profile``, so its
public surface here is the interface contract for the rest of the package.
``Profile`` itself is a plain data container plus (de)serialization; the
algorithms that populate one (loading, occupancy, fve, boundaries) belong to
``metrics.py``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Boundaries:
    """The five independent workspace-boundary signals.

    Each is a layer depth, subsampled to 25 evenly spaced layers and
    reindexed to 0-100 (paper convention), at which that signal locates the
    start of the workspace band.

    Attributes:
        topk_accuracy: Boundary by top-k accuracy of the lens predicting the
            model's actual next token.
        kurtosis: Boundary by excess kurtosis of the readout logit distribution.
        autocorrelation: Boundary by autocorrelation of the top-1 lens token
            across positions vs. a position-shuffled null.
        effective_dim: Boundary by effective linear dimensionality of
            ``W_U @ J_l``.
        cka: Boundary by CKA between layers' J-lens gram matrices.
        disagreement: True when the signals span more than 10% of depth
            (10.0 on the 0-100 scale).
    """

    topk_accuracy: float
    kurtosis: float
    autocorrelation: float
    effective_dim: float
    cka: float
    disagreement: bool

    def as_dict(self) -> dict[str, float | bool]:
        return asdict(self)

    @classmethod
    def from_signals(
        cls,
        *,
        topk_accuracy: float,
        kurtosis: float,
        autocorrelation: float,
        effective_dim: float,
        cka: float,
        disagreement_threshold: float = 10.0,
    ) -> Boundaries:
        """Build a ``Boundaries``, computing ``disagreement`` from the spread
        of the five signals."""
        signals = (topk_accuracy, kurtosis, autocorrelation, effective_dim, cka)
        disagreement = (max(signals) - min(signals)) > disagreement_threshold
        return cls(
            topk_accuracy=topk_accuracy,
            kurtosis=kurtosis,
            autocorrelation=autocorrelation,
            effective_dim=effective_dim,
            cka=cka,
            disagreement=disagreement,
        )


def _npz_path(path: str) -> str:
    return path if path.endswith(".npz") else f"{path}.npz"


def _to_str_nested_list(arr: np.ndarray) -> list:
    """``(n_pos, n_layer, max_k)`` object array -> nested list of ``str``, for
    JSON encoding."""
    return [[[str(tok) for tok in cell] for cell in row] for row in arr]


class Profile:
    """A frozen snapshot of one prompt's J-space content.

    Holds, per (position, layer): occupancy, ranked top-k tokens, and (per
    concept) loading; plus per-layer FVE and the workspace boundary signals.
    Positions and layers are the caller's own index values (e.g. sequence
    position, layer number) — :meth:`topk` etc. look up by those values, not
    by array offset.
    """

    def __init__(
        self,
        *,
        positions: Sequence[int],
        layers: Sequence[int],
        occupancy: np.ndarray,
        fve: np.ndarray,
        boundaries: Boundaries,
        topk_tokens: np.ndarray,
        topk_scores: np.ndarray,
        loadings: dict[str, np.ndarray] | None = None,
    ) -> None:
        positions = list(positions)
        layers = list(layers)
        if len(set(positions)) != len(positions):
            raise ValueError(f"positions must be unique, got {positions}")
        if len(set(layers)) != len(layers):
            raise ValueError(f"layers must be unique, got {layers}")

        n_pos, n_layer = len(positions), len(layers)
        expected_2d = (n_pos, n_layer)

        occupancy = np.asarray(occupancy, dtype=np.float32)
        if occupancy.shape != expected_2d:
            raise ValueError(f"occupancy shape {occupancy.shape} != {expected_2d}")

        fve = np.asarray(fve, dtype=np.float32)
        if fve.shape != (n_layer,):
            raise ValueError(f"fve shape {fve.shape} != {(n_layer,)}")

        topk_tokens = np.asarray(topk_tokens, dtype=object)
        topk_scores = np.asarray(topk_scores, dtype=np.float32)
        if topk_tokens.shape != topk_scores.shape:
            raise ValueError(
                f"topk_tokens shape {topk_tokens.shape} != "
                f"topk_scores shape {topk_scores.shape}"
            )
        if topk_tokens.ndim != 3 or topk_tokens.shape[:2] != expected_2d:
            raise ValueError(
                f"topk_tokens/scores shape {topk_tokens.shape} != "
                f"({n_pos}, {n_layer}, max_k)"
            )

        loadings = dict(loadings) if loadings else {}
        for concept, arr in loadings.items():
            arr = np.asarray(arr, dtype=np.float32)
            if arr.shape != expected_2d:
                raise ValueError(
                    f"loading for {concept!r} has shape {arr.shape} != {expected_2d}"
                )
            loadings[concept] = arr

        self.positions = positions
        self.layers = layers
        self.max_k = topk_tokens.shape[2]
        self._occupancy = occupancy
        self._fve = fve
        self._boundaries = boundaries
        self._topk_tokens = topk_tokens
        self._topk_scores = topk_scores
        self._loadings = loadings
        self._pos_index = {p: i for i, p in enumerate(positions)}
        self._layer_index = {l: i for i, l in enumerate(layers)}

    def _resolve_position(self, pos: int) -> int:
        if pos not in self._pos_index:
            raise KeyError(f"position {pos} not in profile positions {self.positions}")
        return self._pos_index[pos]

    def _resolve_layer(self, layer: int) -> int:
        if layer not in self._layer_index:
            raise KeyError(f"layer {layer} not in profile layers {self.layers}")
        return self._layer_index[layer]

    def loading(self, concept: str) -> np.ndarray:
        """Cosine-similarity loading trace for ``concept``: shape ``(n_pos, n_layer)``."""
        if concept not in self._loadings:
            raise KeyError(
                f"no loading recorded for concept {concept!r}; "
                f"available: {sorted(self._loadings)}"
            )
        return self._loadings[concept].copy()

    def occupancy(self) -> np.ndarray:
        """Number of active J-lens concepts per (position, layer): shape
        ``(n_pos, n_layer)``."""
        return self._occupancy.copy()

    def boundaries(self) -> Boundaries:
        """The five workspace-boundary signals for this profile."""
        return self._boundaries

    def topk(self, pos: int, layer: int, k: int = 25) -> list[tuple[str, float]]:
        """Top-``k`` ranked (token, score) pairs at ``(pos, layer)``."""
        if not 1 <= k <= self.max_k:
            raise ValueError(f"k={k} out of range; profile stores up to {self.max_k}")
        pi = self._resolve_position(pos)
        li = self._resolve_layer(layer)
        tokens = self._topk_tokens[pi, li, :k]
        scores = self._topk_scores[pi, li, :k]
        return [(str(tok), float(score)) for tok, score in zip(tokens, scores)]

    def fve(self) -> np.ndarray:
        """Fraction of variance explained per layer: shape ``(n_layer,)``."""
        return self._fve.copy()

    def save(self, path: str) -> None:
        """Serialize to ``path`` (``.npz`` appended if missing)."""
        concepts = sorted(self._loadings)
        if concepts:
            loading_stack = np.stack([self._loadings[c] for c in concepts])
        else:
            loading_stack = np.zeros(
                (0, len(self.positions), len(self.layers)), dtype=np.float32
            )
        meta = {
            "positions": self.positions,
            "layers": self.layers,
            "concepts": concepts,
            "boundaries": self._boundaries.as_dict(),
            "topk_tokens": _to_str_nested_list(self._topk_tokens),
        }
        np.savez(
            _npz_path(path),
            loading_stack=loading_stack,
            occupancy=self._occupancy,
            fve=self._fve,
            topk_scores=self._topk_scores,
            meta=np.array(json.dumps(meta)),
        )

    @classmethod
    def load(cls, path: str) -> Profile:
        """Load a profile previously written by :meth:`save`."""
        with np.load(_npz_path(path), allow_pickle=False) as data:
            meta = json.loads(data["meta"].item())
            concepts = meta["concepts"]
            loading_stack = data["loading_stack"]
            loadings = {c: loading_stack[i] for i, c in enumerate(concepts)}
            topk_tokens = np.array(meta["topk_tokens"], dtype=object)
            return cls(
                positions=meta["positions"],
                layers=meta["layers"],
                occupancy=data["occupancy"],
                fve=data["fve"],
                boundaries=Boundaries(**meta["boundaries"]),
                topk_tokens=topk_tokens,
                topk_scores=data["topk_scores"],
                loadings=loadings,
            )
