# subvocal

A workspace occupancy profiler and prompt debugger built on the Jacobian lens
from Anthropic's [*Verbalizable Representations Form a Global Workspace in
Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
(2026). Not affiliated with Anthropic.

Wraps [`jlens`](https://github.com/anthropics/jacobian-lens), the paper's
reference implementation, with measurement and debugging tooling; does not
reimplement the lens itself.

**Status: M1 + M2.** The `Profile` interface, a deterministic `StubLens`, and
metrics (occupancy via Gradient Pursuit, loading, FVE, the five boundary
signals) are in place, plus `FittedLens`, wrapping the pinned model
(`Qwen/Qwen3.5-4B`) and its pre-fitted Jacobian lens. The contrast-pair
debugger, ablation/steering verification, and the HTML report are not yet
built — see `CLAUDE.md` for the milestone plan.

## Setup

```bash
uv sync --extra dev
```

This installs `jlens` as an editable local dependency from `jacobian-lens/`,
plus torch, transformers, and numpy.

MPS (Apple Silicon) requires the fallback env var to be set before any
torch-heavy import:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

`subvocal.lens.resolve_device()` raises rather than silently continuing if
MPS is available and this isn't set.

## Tests

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run pytest
```

## Module layout

```
subvocal/
  lens.py       jlens wrapper, device handling, lens + residual caching
  metrics.py    occupancy, loading, boundaries, autocorrelation, CKA, FVE
  profile.py    the Profile object, serialization
  debug.py      contrast pairs, ranking, ablation + steering verification
  report.py     HTML output
tests/
modal_fit.py    lives outside the package, runs once
```

`lens.py`, `metrics.py`, and `profile.py` exist so far. `debug.py` and
`report.py` are next; `modal_fit.py` may not be needed at all, since subvocal
uses the paper's pre-fitted Qwen3.5-4B lens rather than fitting its own.

## The `Profile` interface

Everything downstream is built against this shape:

```python
Profile.loading(concept: str) -> ndarray      # (n_pos, n_layer)
Profile.occupancy() -> ndarray                # (n_pos, n_layer)
Profile.boundaries() -> Boundaries            # five signals + disagreement flag
Profile.topk(pos, layer, k=25) -> list[tuple[str, float]]
Profile.fve() -> ndarray                      # (n_layer,)
Profile.save(path) / Profile.load(path)
```

`pos` and `layer` in `topk()` are the caller's own index values (e.g. actual
sequence position, actual layer number), not array offsets.

### `StubLens`

Until a fitted lens exists at `artifacts/lens.pt`, `subvocal.lens.StubLens`
stands in for it. It fabricates a residual vector per `(prompt, position,
layer)` and a J-lens direction per `(concept, layer)` — both deterministic,
seeded unit vectors — and derives `readout`/`topk` from their cosine
similarity, the same relationship a real lens has between its transport and
its decode. This lets metrics and tests be built and exercised before the
real lens lands, without depending on fake data that only happens to have the
right shape.

```python
from subvocal.lens import StubLens

lens = StubLens(n_layers=24, d_model=64, vocab_size=64)
lens.topk("the quick brown fox", position=2, layer=12, k=5)
```

### `FittedLens`

The real thing: `subvocal.lens.FittedLens` wraps a loaded HF model (via
`jlens.from_hf`) and a fitted `jlens.JacobianLens`, implementing the same
surface as `StubLens` so `metrics.py` and `Profile` run unchanged against
either. `FittedLens.from_pretrained()` downloads the pinned model and its
pre-fitted lens from the Hub — no local fitting run needed:

```python
from subvocal.lens import FittedLens, QWEN3_5_4B

lens = FittedLens.from_pretrained(QWEN3_5_4B)
lens.topk("the quick brown fox", position=2, layer=12, k=5)
```

`residual`/`readout` run the model's real forward pass (cached per prompt);
`concept_direction` (and the loading/occupancy machinery built on it)
pulls each concept's unembedding row back through `J_l`, dropping the final
RMSNorm's data-dependent rescaling — a standard direct-logit-attribution
linearization, not a paper-verified formula. See the class docstring for the
full reasoning.
