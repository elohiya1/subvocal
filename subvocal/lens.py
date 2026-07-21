"""Device handling and the Jacobian-lens readout surface subvocal builds on.

Real usage wraps a fitted ``jlens.JacobianLens`` and an HF model (arriving in
a later milestone, once the model size is pinned and ``artifacts/lens.pt``
exists). Until then, :class:`StubLens` implements the same readout surface —
per-(position, layer) residual vectors, per-(concept, layer) J-lens
directions, and the ranked-token decode — with deterministic fake data, so
``metrics.py`` and ``Profile`` can be built and tested against a stand-in.

All device resolution for the package goes through :func:`resolve_device`;
nothing else in subvocal should call ``.cuda()`` or a bare ``.to(device)``.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence

import numpy as np
import torch


def resolve_device(preferred: str | torch.device | None = None) -> torch.device:
    """Resolve the torch device subvocal should run on.

    Args:
        preferred: Explicit override (e.g. ``"cpu"`` for tests). When
            ``None``, prefers MPS, falls back to CUDA, then CPU.

    Returns:
        The resolved device.

    Raises:
        RuntimeError: MPS is available but ``PYTORCH_ENABLE_MPS_FALLBACK=1``
            is not set in the environment; several ops jlens/transformers use
            have no native MPS kernel and silently fall back only when this
            is set.
    """
    if preferred is not None:
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            raise RuntimeError(
                "MPS is available but PYTORCH_ENABLE_MPS_FALLBACK is not set "
                "to '1'; set it before importing torch-heavy modules."
            )
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def empty_cache(device: torch.device) -> None:
    """Release cached allocator memory for ``device``. No-op on CPU."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _seeded_unit_vector(dim: int, *parts: object, seed: int) -> np.ndarray:
    """Deterministic pseudo-random unit vector, keyed by ``(seed, *parts)``.

    Same key always yields the same vector (within a process and across
    processes/platforms), which is what makes :class:`StubLens` a reliable
    stand-in for tests.
    """
    key = "|".join(str(p) for p in (seed, *parts)).encode()
    digest = hashlib.sha256(key).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class StubLens:
    """Deterministic fake Jacobian-lens readout, for developing against
    before a fitted lens exists.

    Fabricates a residual vector per (prompt, position, layer) and a J-lens
    direction per (concept, layer), both unit vectors on the ``d_model``
    sphere, seeded so the same inputs always reproduce the same outputs.
    :meth:`readout` and :meth:`topk` are cosine similarities against
    :attr:`vocab`, so they are internally consistent with :meth:`residual`
    and :meth:`concept_direction` the way a real lens's decode is consistent
    with its own transport.

    Attributes:
        n_layers: Number of fake layers.
        d_model: Fake residual-stream width.
        vocab: Fake vocabulary; every entry doubles as a "concept" name.
    """

    def __init__(
        self,
        *,
        n_layers: int = 24,
        d_model: int = 64,
        vocab: Sequence[str] | None = None,
        vocab_size: int = 64,
        seed: int = 0,
    ) -> None:
        self.n_layers = n_layers
        self.d_model = d_model
        self.vocab: list[str] = list(vocab) if vocab is not None else [
            f"tok{i}" for i in range(vocab_size)
        ]
        self.seed = seed

    def encode(self, prompt: str) -> list[str]:
        """Fake tokenization: whitespace split. Deterministic, not real BPE."""
        return prompt.split()

    def _check_layer(self, layer: int) -> None:
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer={layer} out of range for {self.n_layers} layers")

    def residual(self, prompt: str, position: int, layer: int) -> np.ndarray:
        """Fake residual-stream vector at ``(position, layer)``: shape ``(d_model,)``."""
        self._check_layer(layer)
        return _seeded_unit_vector(
            self.d_model, "residual", prompt, position, layer, seed=self.seed
        )

    def concept_direction(self, concept: str, layer: int) -> np.ndarray:
        """Fake J-lens direction for ``concept`` at ``layer``: shape ``(d_model,)``."""
        self._check_layer(layer)
        return _seeded_unit_vector(
            self.d_model, "concept", concept, layer, seed=self.seed
        )

    def readout(self, prompt: str, position: int, layer: int) -> np.ndarray:
        """Fake decode over the full vocab at ``(position, layer)``.

        Cosine similarity between :meth:`residual` and every vocab entry's
        :meth:`concept_direction`, mirroring ``unembed(J_l @ h)`` scored
        against each vocabulary direction. Shape ``(len(vocab),)``.
        """
        h = self.residual(prompt, position, layer)
        directions = np.stack(
            [self.concept_direction(tok, layer) for tok in self.vocab]
        )
        return directions @ h

    def topk(
        self, prompt: str, position: int, layer: int, k: int = 25
    ) -> list[tuple[str, float]]:
        """Top-``k`` ranked tokens at ``(position, layer)`` by :meth:`readout`."""
        if not 1 <= k <= len(self.vocab):
            raise ValueError(f"k={k} out of range for vocab of size {len(self.vocab)}")
        scores = self.readout(prompt, position, layer)
        order = np.argsort(-scores)[:k]
        return [(self.vocab[i], float(scores[i])) for i in order]
