# subvocal

A workspace occupancy profiler and prompt debugger built on the Jacobian lens
from Anthropic's [*Verbalizable Representations Form a Global Workspace in
Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
(2026). Not affiliated with Anthropic.

Wraps [`jlens`](https://github.com/anthropics/jacobian-lens), the paper's
reference implementation, with measurement and debugging tooling; does not
reimplement the lens itself.

**Status: M1-M5, including the M3 stretch goal.** The `Profile` interface, a
deterministic `StubLens`, and metrics (occupancy via Gradient Pursuit,
loading, FVE, the five boundary signals) are in place, plus `FittedLens`,
wrapping the pinned model (`Qwen/Qwen3.5-4B`) and its pre-fitted Jacobian
lens. `debug.py` has all three M3 modes (`probe()`, `contrast()`, and
`propose()`) and M4's `verify_ablate()`/`verify_steer()`. `report.py` (M5)
renders a `Profile` summary alongside `jlens`'s own d3 slice view. See "A
worked example" below for all of it run against the real model, and
Limitations for what does and doesn't hold up on this particular model. M2's
real-lens sanity checks have been run and reported there too; two of the
three fail, and per CLAUDE.md that failure is being carried forward as a
documented finding rather than fixed by tuning thresholds -- along with a
follow-up investigation into why (transport fidelity by depth), also in
Limitations.

## Setup

```bash
git clone https://github.com/anthropics/jacobian-lens.git
uv sync --extra dev
```

`jacobian-lens/` is gitignored (it's a separate upstream checkout with its
own `.git`, not part of this repo) -- `uv sync` installs `jlens` as an
editable local dependency from that path, so it has to exist first, or
`uv sync` fails immediately with `Distribution not found`. This also
installs torch, transformers, and numpy.

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

All five package modules exist. `modal_fit.py` was never needed: subvocal
uses the paper's pre-fitted Qwen3.5-4B lens from the Hub rather than fitting
its own.

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

`subvocal.lens.StubLens` stands in for `FittedLens`: same readout surface,
deterministic fake data, no model or network access needed. It fabricates a
residual vector per `(prompt, position, layer)` and a J-lens direction per
`(concept, layer)` — both deterministic, seeded unit vectors — and derives
`readout`/`topk` from their cosine similarity, the same relationship a real
lens has between its transport and its decode. This is what `metrics.py`'s
and `debug.py`'s test suites run against; `verify_ablate`/`verify_steer`
need a real forward pass and don't work with it (see M4 below).

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

All three of CLAUDE.md's M3 modes, including the `propose()` stretch goal:

```python
from subvocal import debug

# probe(): does `concept` show up where you expect it to?
result = debug.probe(lens, "The cat sat on the mat.", " cat")
result.peak_layer, result.peak_loading   # where loading("cat") peaks, and how high

# contrast(): why did `failing_prompt` behave differently from `working_prompt`?
result = debug.contrast(lens, working_prompt, failing_prompt)
result.ranked(10)   # concepts strongly present in working, weak/absent in failing

# propose(): what stands out in this one prompt, with no hypothesis at all?
result = debug.propose(lens, prompt)
result.ranked(10)   # concepts unusually present here vs. a baseline corpus
```

`contrast()` requires the two prompts to tokenize to the same length (so
position `i` means the same slot in both) and normalizes each concept's
loading delta against how much that concept naturally varies across a
baseline corpus, per CLAUDE.md — see the function docstring for the exact
judgment calls (band selection, baseline corpus, normalization floor). All
three functions warn (not raise) when a caller-supplied concept tokenizes to
more than one token; `FittedLens` itself still raises if asked to resolve a
concept that isn't a real single token.

`propose()` is the natural third mode: `probe()` needs a concept you already
suspect, `contrast()` needs a second, token-aligned prompt to diff against;
`propose()` needs neither. It z-scores each concept's loading in the one
prompt against that same concept's mean and spread across the baseline
corpus — the same normalization `contrast()` uses for its delta, just
against a corpus instead of a second prompt. Run for real on the worked
example's "small" prompt below with no concept list beyond the same
hypothesis-driven set, it puts `trophy` and `suitcase` at the top, each at
its own token position — genuine self-discovery of the prompt's actual
content words, not something hand-fed to it.

`contrast()`/`propose()`'s default concept dictionary (a random vocab
sample, same as `metrics.concept_dictionary`) works fine for the sanity
checks in the Limitations section, but is a poor default for diagnosing one
specific prompt: a random sample of a ~248k-token vocabulary is mostly
non-English fragments with no relevance to your prompt. Pass an explicit
`concepts=[...]` list once you have a hypothesis — see the worked example.

### `verify_ablate()` / `verify_steer()` (M4)

Necessity and sufficiency checks, always reported against a required
random-direction control (CLAUDE.md: "Without it any effect could be generic
perturbation damage"). Both need a real forward-pass re-run past the
intervened layer, which only `FittedLens` can do — `StubLens` doesn't
implement them.

```python
# verify_ablate(): does removing `concept`'s direction actually change the model?
result = debug.verify_ablate(lens, prompt, " tiny")
result.skipped            # True if `concept` was already in the clean top-10 (paper convention)
result.concept_outcome    # AblationOutcome: did top-1 change, how much was it suppressed
result.control_outcome    # same, for a matched random direction -- always reported side by side

# verify_steer(): does adding `concept`'s direction in make it "recover" into the output?
result = debug.verify_steer(lens, prompt, " tiny", alpha=6.0)
result.concept_outcome    # SteerOutcome: rank before/after, did it enter the top-k
result.control_outcome    # same, for a matched random direction
```

`verify_ablate` skips entirely when `concept` is already in the clean
forward pass's top-10 — ablating it there would just remove it from its own
imminent output, not test whether it was used in internal reasoning, per
CLAUDE.md. Both exclude the model's actual final layer from the
intervention band: ablating/steering there is just editing the logits
directly, not a hidden-state intervention.

### `report.py` (M5)

One HTML report per prompt: a `Profile` summary (boundary signals, FVE/
occupancy per layer, concept loading peaks — plain tables, no charting) atop
`jlens.vis`'s own interactive d3 slice view, embedded in an iframe
unmodified. CLAUDE.md: "reusing the d3 slice view from ../jacobian-lens. Do
not build a new visualization."

```python
from subvocal import report

page = report.build_report(lens, prompt, concepts=[" tiny", " small", " large"])
open("report.html", "w").write(page)
```

## A worked example

A real diagnosed case, run against `FittedLens.from_pretrained(QWEN3_5_4B)`.
Two token-aligned (14 tokens each) minimal-pair prompts:

```
working: "The trophy did not fit in the suitcase because it was too small."
failing: "The trophy did not fit in the suitcase because it was too large."
```

Neither is a "failure" in the sense of the model outputting something overtly
wrong — this small base model's raw next-token continuation after a period is
almost always generic (`"\n"`, connective words), regardless of prompt. The
interesting question is whether the two prompts differ *internally*, even
when their surface output doesn't.

**`contrast()`, hypothesis-driven concepts** (the default random-vocab
dictionary is noise for a single prompt pair — see above):

```python
concepts = [" trophy", " suitcase", " small", " large", " big", " fit",
            " broken", " heavy", " size", " tiny", " huge", " box"]
result = debug.contrast(lens, working, failing, concepts=concepts,
                         layers=[4, 8, 12, 16, 20, 24, 28])
```

| concept | score | best layer |
|---|---|---|
| ` tiny` | 1.68 | 16 |
| ` small` | 1.56 | 12 |
| ` big` | 0.94 | 8 |
| ` broken` | 0.88 | 8 |
| ` fit` | 0.62 | 8 |
| ` suitcase` | 0.56 | 28 |
| ` trophy` | 0.39 | 16 |

`small`, unsurprisingly, tops the list — it's the literal word that differs.
More interesting: `tiny` ranks *above* it, and `tiny` never appears in either
prompt. `contrast()` surfaced a genuine semantic neighbor of the concept that
differs, not just an echo of the input tokens. The two entities (`trophy`,
`suitcase`) — the classic Winograd-schema referents — show much weaker
deltas; this lens doesn't show clean evidence of the antecedent-tracking
shift the small/large flip is classically used to probe (consistent with
Limitations below).

**`propose()`, same concepts, no working/failing pair at all** — just the
"small" prompt on its own, scored against the baseline corpus:

```python
result = debug.propose(lens, working, concepts=concepts,
                        layers=[4, 8, 12, 16, 20, 24, 28])
```

| concept | score | best layer | best position |
|---|---|---|---|
| ` trophy` | 5.62 | 8 | 1 (where "trophy" is) |
| ` suitcase` | 5.42 | 8 | 7 (where "suitcase" is) |
| ` fit` | 4.22 | 24 | 4 (where "fit" is) |
| ` large` | 3.82 | 28 | 11 (where "small" is) |
| ` small` | 2.88 | 8 | 12 (its own position) |

With zero hypothesis about what this prompt is "about," `propose()`
correctly surfaces `trophy` and `suitcase` — the actual content words — each
peaking at its own token position, ahead of every other concept in the list
including the literal word `small`. That's a genuine self-discovery result,
not something hand-fed to it the way `contrast()`'s concept list was.

**`verify_ablate()`**: does removing `tiny`'s direction from the *working*
("small") prompt do anything, versus a matched random direction?

```python
debug.verify_ablate(lens, working, " tiny", layers=[4, 8, 12, 16, 20, 24, 28])
```

|  | top-1 logit before | top-1 logit after | Δ |
|---|---|---|---|
| concept (`tiny`) | 18.625 | 18.250 | **-0.375** |
| random control | 18.625 | 18.750 | +0.125 |

Ablating `tiny` measurably suppresses the model's own top prediction; the
random-direction control doesn't (if anything, it nudges the other way).
Top-1 itself (`"\n"`) doesn't flip — a generic connective token this far
into a full vocabulary is a high bar to dislodge with one concept's removal
— but the *specific vs. generic* separation is exactly what the required
control is for.

**`verify_steer()`**: does adding `tiny`'s direction *into* the failing
("large") prompt — where it's currently almost entirely absent — recover it?

```python
debug.verify_steer(lens, failing, " tiny", alpha=6.0, layers=[4, 8, 12, 16, 20, 24, 28])
```

|  | rank before | rank after | entered top-25 |
|---|---|---|---|
| concept (`tiny`) | 9773 | **0** | yes |
| random control | 9773 | 17901 | no |

Steering `tiny` in doesn't just nudge it — it becomes the model's literal
top prediction (rank 0), while the matched random control makes it *less*
likely. This is the clean result in this worked example: `tiny`'s direction
has real, specific causal power over the model's output; a same-magnitude
random perturbation does not.

Generate the full HTML report (`Profile` summary + `jlens`'s interactive
slice view) for either variant with the `report.py` snippet above --
`artifacts/` is gitignored (run output, not a tracked asset), so it isn't
checked into the repo:

```python
page = report.build_report(lens, working, concepts=concepts, layers=[4, 8, 12, 16, 20, 24, 28, 31])
open("artifacts/trophy_suitcase_small.html", "w").write(page)
```

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
  The leading explanation is scale: this is a 4B model with a lens fit on a
  few hundred wikitext excerpts, not whatever scale the paper's own
  experiments used — consistent with the next two bullets.
- **Follow-up: `J_l`'s transport fidelity, measured directly.** `readout()`
  (and everything built on it) decodes `unembed(J_l @ h_l)` -- how well does
  `J_l @ h_l` actually resemble the model's real final-layer residual
  `h_final` from the same forward pass? Measured directly (cosine similarity
  and R², pooled over positions across 3 prompts, `QWEN3_5_4B`'s real
  32-layer lens): fidelity rises smoothly and monotonically with depth --
  R² ≈ 0 at layers 0-2, R² ≈ 0.13-0.36 across layers 10-20 (the mid-depth
  band `subsample_layers` samples most densely, and where the paper's
  occupancy plateau lives), climbing to R² ≈ 0.70-0.83 by layers 28-30. No
  jump or reversal anywhere that would suggest the hybrid linear-attention
  layers broke the fit specifically -- this is the shape you'd expect from
  *any* single averaged Jacobian linearizing a deep nonlinear stack, the
  paper's models presumably included. It does mean that at exactly the
  depths occupancy is measured, `J_l @ h_l` explains under 40% of
  `h_final`'s variance -- most of what the network still goes on to do with
  that residual is, by construction, invisible to a purely linear readout
  there. A concrete, quantified mechanism consistent with the scale
  explanation above, though it doesn't distinguish "this model's workspace
  really is narrower" from "a bigger model would show the same fidelity
  curve and still hit a 25-concept plateau anyway" -- that needs a second
  model to compare against, which doesn't fit this machine (see below).
- **Weak occupancy doesn't mean weak causal power.** M4's verification (see
  the worked example) found individual concept directions with real,
  specific causal effects on the model's output — ablating one measurably
  suppressed the top prediction while a matched random direction didn't;
  steering one in moved a token from rank ~9800 to rank 0 while a matched
  random direction moved it further away. That's the mechanism working as
  intended. It's the *occupancy/boundary* read on this model and lens — how
  many concepts, arranged into what depth structure — that doesn't match the
  paper's numbers, not the underlying "does a concept's direction do
  anything" question.
- **Absence from the J-space is not proof of absence.** A concept missing
  from `topk`/`occupancy` may have been used by the model outside the
  workspace; see the paper's line-counting selectivity experiments.
- **One small open model.** All real-lens results here are from
  `Qwen/Qwen3.5-4B` with a community-fitted lens, not the models studied in
  the paper.
