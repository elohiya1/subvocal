# subvocal

A workspace occupancy profiler and prompt debugger built on the Jacobian lens
from Anthropic's [*Verbalizable Representations Form a Global Workspace in
Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
(2026). Not affiliated with Anthropic.

Wraps [`jlens`](https://github.com/anthropics/jacobian-lens), the paper's
reference implementation, with measurement and debugging tooling; does not
reimplement the lens itself.

**Status: M1 + M2 + M3 (partial).** The `Profile` interface, a deterministic
`StubLens`, and metrics (occupancy via Gradient Pursuit, loading, FVE, the
five boundary signals) are in place, plus `FittedLens`, wrapping the pinned
model (`Qwen/Qwen3.5-4B`) and its pre-fitted Jacobian lens. `debug.py` adds
`probe()` and `contrast()`; `propose()` is a stretch goal and was skipped.
Ablation/steering verification (M4) and the HTML report (M5) are not yet
built — see `CLAUDE.md` for the milestone plan. M2's real-lens sanity checks
have been run and reported — see Limitations below; two of the three fail,
and per CLAUDE.md that failure is being carried forward as a documented
finding rather than fixed by tuning thresholds.

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

`lens.py`, `metrics.py`, `profile.py`, and `debug.py` exist so far (`debug.py`
currently has `probe()`/`contrast()` only; ablation/steering verification is
M4). `report.py` is next; `modal_fit.py` may not be needed at all, since
subvocal uses the paper's pre-fitted Qwen3.5-4B lens rather than fitting its
own.

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

## `debug.py`: prompt debugging

Two of CLAUDE.md's three M3 modes (`propose()` is a stretch goal, skipped):

```python
from subvocal import debug

# probe(): does `concept` show up where you expect it to?
result = debug.probe(lens, "The cat sat on the mat.", " cat")
result.peak_layer, result.peak_loading   # where loading("cat") peaks, and how high

# contrast(): why did `failing_prompt` behave differently from `working_prompt`?
result = debug.contrast(lens, working_prompt, failing_prompt)
result.ranked(10)   # concepts strongly present in working, weak/absent in failing
```

`contrast()` requires the two prompts to tokenize to the same length (so
position `i` means the same slot in both) and normalizes each concept's
loading delta against how much that concept naturally varies across a
baseline corpus, per CLAUDE.md — see the function docstring for the exact
judgment calls (band selection, baseline corpus, normalization floor). Both
functions warn (not raise) when a caller-supplied concept tokenizes to more
than one token; `FittedLens` itself still raises if asked to resolve a
concept that isn't a real single token.

## Limitations

- **Single-token vocabulary.** The lens only reads out individual tokens;
  multi-token concepts raise at the API boundary (`FittedLens.concept_direction`
  / `_token_id`).
- **M2's real-lens sanity checks (CLAUDE.md) mostly fail on this model.** Run
  against `FittedLens.from_pretrained(QWEN3_5_4B)` over four ~60-100 token
  prompts, positions restricted to `jlens.fitting.valid_position_mask`'s
  definition of valid (excludes the first 16 positions, which the lens was
  never fit on — they act as attention sinks with atypical residual
  statistics):
  - FVE ≤ 0.10: **passes** (max observed 0.0133).
  - Occupancy shape (near-zero early, plateau near 25 mid-depth): **fails**.
    Mean occupancy stays in the 1.4-5.9 range across every sampled layer and
    every prompt tried — no near-zero start, no plateau anywhere near 25.
  - Five boundary signals within 10% of depth: **fails**. `topk_accuracy`
    (depth 85.7) and `kurtosis` (83.3) agree with each other; `autocorrelation`
    (7.8) and `cka` (8.4) agree with each other; `effective_dim` (35.1) sits
    between the two clusters. Spread: 77.8.

  Investigation ruled out several candidate bugs before landing here:
  `concept_direction`'s RMSNorm-gain approximation reproduces the exact
  `readout()` ranking almost perfectly (22-24/25 top-k overlap, Spearman
  ρ≈0.96 across layers); occupancy is flat regardless of dictionary size
  (200 atoms through the full ~248k-token vocab, all landing in the same 2-3
  mean range); restricting to fit-valid positions moves the numbers by
  tenths, not by the order of magnitude that would be needed. The Hub-fitted
  lens (`qwen-n1000`) converged well before its 1000-prompt budget
  (417 prompts, `stop_at_delta=0.002` reached), so this isn't simple
  data-starvation either. Nothing found points at a bug in this package.
  The leading (untested further) explanation is scale: this is a 4B model
  with a lens fit on a few hundred wikitext excerpts, not whatever scale the
  paper's own experiments used — consistent with the next bullet.
- **Absence from the J-space is not proof of absence.** A concept missing
  from `topk`/`occupancy` may have been used by the model outside the
  workspace; see the paper's line-counting selectivity experiments.
- **One small open model.** All real-lens results here are from
  `Qwen/Qwen3.5-4B` with a community-fitted lens, not the models studied in
  the paper.
