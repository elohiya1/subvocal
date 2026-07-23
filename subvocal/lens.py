"""Device handling and the Jacobian-lens readout surface subvocal builds on.

:class:`FittedLens` wraps a real HF model plus a fitted ``jlens.JacobianLens``
(pinned model: Qwen3.5-4B, see :data:`QWEN3_5_4B` — small enough for M4
unified memory in bf16, with a pre-fitted lens already on the Hub so subvocal
doesn't need its own Modal fitting run). :class:`StubLens` implements the same
readout surface with deterministic fake data, for tests and for developing
metrics/``Profile`` code without a model or network access.

All device resolution for the package goes through :func:`resolve_device`;
nothing else in subvocal should call ``.cuda()`` or a bare ``.to(device)``.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import jlens
import numpy as np
import torch
import transformers


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


@dataclass(frozen=True)
class LensSource:
    """Where to fetch a pinned model and its pre-fitted Jacobian lens from."""

    model_name: str
    lens_repo: str
    lens_file: str
    lens_revision: str | None = None


#: Reference model/lens pairs with pre-fitted lenses already on the Hub (see
#: jacobian-lens/walkthrough.ipynb). Pinned default for subvocal.
QWEN3_5_4B = LensSource(
    model_name="Qwen/Qwen3.5-4B",
    lens_repo="neuronpedia/jacobian-lens",
    lens_file="qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
    lens_revision="qwen-n1000",
)

#: The paper's larger reference model. Its weights don't fit M4 unified
#: memory even in bf16 (~54GB) -- kept here for completeness, not the
#: default subvocal loads.
QWEN3_6_27B = LensSource(
    model_name="Qwen/Qwen3.6-27B",
    lens_repo="neuronpedia/jacobian-lens",
    lens_file="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    lens_revision="qwen-n1000",
)


class FittedLens:
    """Real Jacobian-lens readout: a loaded HF model wrapped by
    ``jlens.from_hf``, plus a fitted ``jlens.JacobianLens``.

    Implements the same duck-typed surface as :class:`StubLens`
    (``subvocal.metrics.MetricsLens``) so ``metrics.py`` runs unchanged
    against either. Two of those methods are exact and two are an explicitly
    flagged approximation:

    - :meth:`residual` and :meth:`readout` go through ``jlens``'s own
      ``transport``/``unembed`` on real forward-pass activations, so they
      match ``JacobianLens.apply`` exactly.
    - :meth:`concept_direction` (and therefore :meth:`loading` in
      ``metrics.py``, plus the Gradient Pursuit dictionaries built from it)
      needs a fixed linear direction per concept, which the real lens doesn't
      have on its own: ``unembed`` = ``lm_head`` after a final RMSNorm
      (``g * x / rms(x)``), and RMSNorm's per-example denominator ``rms(x)``
      makes the true readout direction depend on the residual itself. This
      class approximates it by pulling the unembedding row for the concept's
      token back through ``J_l`` *without* that denominator
      (``direction_l = (w_token * g) @ J_l``) -- the same linearization
      "direct logit attribution" / logit-lens tooling uses elsewhere. This is
      exact for anything scale-invariant in the readout direction (cosine
      similarity, top-k ranking): ``rms(x)`` is a single positive scalar
      shared by every concept at a given ``(prompt, position, layer)``, since
      ``x = J_l @ h`` doesn't depend on which concept is being read out, so
      dropping it changes no ranking or cosine similarity. The learned
      per-channel gain ``g`` is *not* dropped -- it does not appear in ``J_l``
      (which is fit purely on pre-norm residuals; ``g`` is only applied
      downstream, inside ``unembed``) and omitting it measurably hurt
      reconstruction quality in practice. This is a judgment call, not a
      paper-verified formula.

    Caches aggressively per CLAUDE.md: one forward pass per prompt covers
    every layer's residual (:meth:`residual`) and feeds :meth:`readout`
    without rerunning the model; per-layer concept-direction matrices
    (:meth:`concept_direction`) are built once from ``(W_U * g) @ J_l`` and
    reused across all vocab lookups at that layer, so the
    ``concept_dictionary`` loop in ``metrics.py`` stays cheap even when it
    draws its default-sized sample from a real ~250k-token vocab.
    Call :meth:`clear_cache` to release it (e.g. between prompts/chunks, per
    CLAUDE.md's ``torch.mps.empty_cache()`` guidance).
    """

    def __init__(
        self,
        hf_model: torch.nn.Module,
        tokenizer: object,
        jacobian_lens: jlens.JacobianLens,
        *,
        device: str | torch.device | None = None,
        max_cached_direction_layers: int = 1,
    ) -> None:
        self.device = resolve_device(device)
        self._max_cached_direction_layers = max_cached_direction_layers
        hf_model = hf_model.to(self.device)
        self._model = jlens.from_hf(hf_model, tokenizer)
        self._tokenizer = tokenizer
        self._jacobian_lens = jacobian_lens
        self.n_layers = self._model.n_layers
        self.d_model = self._model.d_model
        self.fitted_layers: list[int] = jacobian_lens.source_layers

        output_embeddings = hf_model.get_output_embeddings()
        if output_embeddings is None:
            raise ValueError(
                f"{type(hf_model).__name__} has no output embedding "
                "(get_output_embeddings() returned None)"
            )
        self._unembed_matrix = (
            output_embeddings.weight.detach().to(torch.float32).cpu().numpy()
        )  # (vocab_size, d_model)

        # The final RMSNorm's learned per-channel gain, applied downstream of
        # J_l (see the class docstring) -- reached off jlens's own layout
        # detection (`_final_norm`) rather than re-deriving which module that
        # is per architecture, since that's exactly the "don't reimplement
        # the lens" line CLAUDE.md draws. Falls back to an identity gain if
        # the resolved norm module has no `.weight` (e.g. a bias-free,
        # non-affine norm), rather than assuming every architecture has one.
        final_norm_weight = getattr(self._model._final_norm, "weight", None)
        self._final_norm_gain: np.ndarray = (
            final_norm_weight.detach().to(torch.float32).cpu().numpy()
            if final_norm_weight is not None
            else np.ones(self.d_model, dtype=np.float32)
        )

        self._vocab_cache: list[str] | None = None
        self._residual_cache: dict[str, dict[int, torch.Tensor]] = {}
        # LRU, not unbounded: each entry is a full (vocab_size, d_model)
        # matrix -- ~2.5GB apiece at Qwen3.5-4B's ~248k-token vocab, so
        # caching all 25 subsampled layers at once (as occupancy_grid /
        # fve_per_layer sweep through) would need ~60GB+ RAM. metrics.py
        # processes one layer fully before moving to the next, so a
        # small LRU keeps peak memory bounded while still caching the
        # within-layer reuse across loading/occupancy/fve for that layer.
        self._direction_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._token_id_cache: dict[str, int] = {}

    @classmethod
    def from_pretrained(
        cls,
        source: LensSource = QWEN3_5_4B,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
    ) -> FittedLens:
        """Download (or reuse a cached) model and pre-fitted lens, and wrap
        them. ``dtype`` is the model's compute dtype, not a numerics
        precision choice -- gradient pursuit and friends still run in
        float32 regardless (see CLAUDE.md's float64 ban, which is about
        numerical routines, not model weights)."""
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            source.model_name, dtype=dtype
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(source.model_name)
        jacobian_lens = jlens.JacobianLens.from_pretrained(
            source.lens_repo, filename=source.lens_file, revision=source.lens_revision
        )
        return cls(hf_model, tokenizer, jacobian_lens, device=device)

    def _check_layer(self, layer: int) -> None:
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer={layer} out of range for {self.n_layers} layers")

    def _is_final_layer(self, layer: int) -> bool:
        """``fit()`` never stores a ``J`` for its own target layer (transport
        into the final-layer basis from the final layer is the identity), so
        ``readout``/``concept_direction`` there skip the transport rather
        than treating it as unfitted. This is what makes CLAUDE.md's
        boundary-signal proxy for "the model's actual next token" (the
        lens's own final-layer readout) correct rather than a KeyError."""
        return layer == self.n_layers - 1

    def _check_readable_layer(self, layer: int) -> None:
        self._check_layer(layer)
        if layer not in self._jacobian_lens.jacobians and not self._is_final_layer(
            layer
        ):
            raise ValueError(
                f"layer={layer} has no fitted Jacobian; fitted layers are "
                f"{self.fitted_layers} (plus the final layer "
                f"{self.n_layers - 1}, read directly without transport)"
            )

    @property
    def vocab(self) -> list[str]:
        """Every token in the tokenizer's vocabulary, decoded, indexed by id."""
        if self._vocab_cache is None:
            vocab_size = self._unembed_matrix.shape[0]
            self._vocab_cache = [
                self._tokenizer.decode([i]) for i in range(vocab_size)
            ]
            # First occurrence wins on decode collisions; used by _token_id
            # to skip re-tokenizing strings that are already known vocab
            # entries -- the common case when metrics.py's concept_dictionary
            # draws its default sample from this vocab.
            for i, tok in enumerate(self._vocab_cache):
                self._token_id_cache.setdefault(tok, i)
        return self._vocab_cache

    def encode(self, prompt: str) -> list[str]:
        """Real tokenization: decoded tokens in sequence order, matching the
        positions :meth:`residual`/:meth:`readout` index into."""
        ids = self._model.encode(prompt)[0].tolist()
        return [self._tokenizer.decode([i]) for i in ids]

    def _activations(self, prompt: str) -> dict[int, torch.Tensor]:
        cached = self._residual_cache.get(prompt)
        if cached is not None:
            return cached
        input_ids = self._model.encode(prompt)
        with torch.no_grad(), jlens.ActivationRecorder(
            self._model.layers, at=range(self.n_layers)
        ) as recorder:
            self._model.forward(input_ids)
        acts = {
            i: recorder.activations[i][0].detach().float().cpu()
            for i in range(self.n_layers)
        }
        self._residual_cache[prompt] = acts
        return acts

    def _resolve_position(self, seq_len: int, position: int) -> int:
        pos = position if position >= 0 else seq_len + position
        if not 0 <= pos < seq_len:
            raise ValueError(f"position={position} out of range for length {seq_len}")
        return pos

    def residual(self, prompt: str, position: int, layer: int) -> np.ndarray:
        """Real residual-stream vector at ``(position, layer)``: shape
        ``(d_model,)``. One forward pass per prompt is cached and reused
        across every ``(position, layer)`` query."""
        self._check_layer(layer)
        acts = self._activations(prompt)
        pos = self._resolve_position(acts[layer].shape[0], position)
        return acts[layer][pos].numpy()

    def _directions_for_layer(self, layer: int) -> np.ndarray:
        """``(W_U * g) @ J_l``: a ``(vocab_size, d_model)`` matmul, cheap
        enough to do once per layer but big enough (full vocab, e.g. ~250k
        rows) to be worth running on-device rather than in CPU numpy. ``g``
        is the final RMSNorm's gain (see the class docstring for why it's
        included but the norm's data-dependent denominator isn't). At the
        final layer ``J_l`` is the identity (see :meth:`_is_final_layer`), so
        the directions are just ``W_U * g``. LRU-bounded (see the cache's
        declaration in ``__init__``); moves ``layer`` to most-recently-used
        on every hit."""
        cached = self._direction_cache.get(layer)
        if cached is not None:
            self._direction_cache.move_to_end(layer)
            return cached
        W_U = torch.from_numpy(self._unembed_matrix * self._final_norm_gain[None, :]).to(
            self.device
        )
        if self._is_final_layer(layer):
            directions = W_U.float().cpu().numpy()
        else:
            J = self._jacobian_lens.jacobians[layer].to(self.device)
            directions = (W_U @ J).float().cpu().numpy()
        self._direction_cache[layer] = directions
        if len(self._direction_cache) > self._max_cached_direction_layers:
            self._direction_cache.popitem(last=False)
        return directions

    def _token_id(self, concept: str) -> int:
        cached = self._token_id_cache.get(concept)
        if cached is not None:
            return cached
        ids = self._tokenizer.encode(concept, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"concept {concept!r} tokenizes to {len(ids)} tokens {ids}; "
                "the lens only sees single tokens"
            )
        self._token_id_cache[concept] = ids[0]
        return ids[0]

    def concept_direction(self, concept: str, layer: int) -> np.ndarray:
        """Approximate J-lens direction for ``concept`` at ``layer``: shape
        ``(d_model,)``. See the class docstring for the linearization this
        drops relative to the true (norm-dependent) readout."""
        self._check_readable_layer(layer)
        token_id = self._token_id(concept)
        return self._directions_for_layer(layer)[token_id]

    def readout(self, prompt: str, position: int, layer: int) -> np.ndarray:
        """Exact decode over the full vocab at ``(position, layer)``: real
        ``transport`` + ``unembed`` (RMSNorm included) on a cached residual,
        matching ``JacobianLens.apply`` without repeating its forward pass.
        At the final layer this is the model's own logits, untransported --
        the proxy ``metrics.py`` relies on for "the model's actual next
        token". Shape ``(vocab_size,)``."""
        self._check_readable_layer(layer)
        h = self.residual(prompt, position, layer)
        h_t = torch.from_numpy(h).to(self.device)
        if self._is_final_layer(layer):
            transported = h_t
        else:
            transported = self._jacobian_lens.transport(h_t, layer)
        logits = self._model.unembed(transported)
        return logits.detach().float().cpu().numpy()

    def topk(
        self, prompt: str, position: int, layer: int, k: int = 25
    ) -> list[tuple[str, float]]:
        """Top-``k`` ranked tokens at ``(position, layer)`` by :meth:`readout`."""
        if not 1 <= k <= len(self.vocab):
            raise ValueError(f"k={k} out of range for vocab of size {len(self.vocab)}")
        scores = self.readout(prompt, position, layer)
        order = np.argpartition(-scores, k - 1)[:k]
        order = order[np.argsort(-scores[order])]
        return [(self.vocab[i], float(scores[i])) for i in order]

    def clear_cache(self) -> None:
        """Drop cached residuals/directions and release device allocator
        memory. Cheap to rebuild -- call between prompts or heavy chunks."""
        self._residual_cache.clear()
        self._direction_cache.clear()
        empty_cache(self.device)
