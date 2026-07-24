# Findings: M2's real-lens sanity checks

Full results from running CLAUDE.md's M2 gate against the real pinned lens
(`FittedLens.from_pretrained(QWEN3_5_4B)`), and the follow-up investigation
into why two of the three checks fail. The README's Limitations section has
the condensed version; this is the primary source, reproducible with the
scripts in `scripts/`.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/m2_sanity_check.py
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/transport_fidelity_check.py
```

Both download `Qwen/Qwen3.5-4B` (~9.3GB) on first run and take several
minutes — see the README's "Resource requirements".

## 1. Sanity-check results

Run: 4 prompts (60-100 tokens each), 25 subsampled layers, occupancy/FVE
positions restricted to `jlens.fitting.valid_position_mask`'s definition of
valid (excludes each prompt's first 16 positions — attention-sink positions
the lens was never fit on).

**Occupancy** (expect near-zero in the first third of layers, plateau near
25 in the middle band):

```
layer:  0    1    3    4    5    6    8    9   10   12   13   14   16   17   18   19   21   22   23   25   26   27   28   30   31
depth:  0    3   10   13   16   19   26   29   32   39   42   45   52   55   58   61   68   71   74   81   84   87   90   97  100
occ:  1.43 1.89 2.16 2.38 2.19 3.41 2.92 2.65 2.67 2.97 2.73 3.07 2.69 2.62 3.42 3.43 4.27 4.46 4.99 4.78 4.79 5.87 4.28 2.65 3.09
```

- first-third mean: **2.38** (expect ~0)
- middle-band mean: **2.95**, peak 5.87 at depth 87 (expect plateau ~25)
- **Verdict: fails.** No near-zero start, and nowhere close to a 25-concept
  plateau at any sampled depth.

**FVE** (must not exceed 0.10):

- mean per layer ranges 0.0000-0.0096; **max observed: 0.0133**
- **Verdict: passes.**

**Boundary signals** (must land within 10% of depth of each other):

| signal | depth |
|---|---|
| topk_accuracy | 85.7 |
| kurtosis | 83.3 |
| autocorrelation | 7.8 |
| effective_dim | 35.1 |
| cka | 8.4 |

- spread: **77.8** (threshold: 10)
- **Verdict: fails.** `topk_accuracy`/`kurtosis` cluster late; `autocorrelation`/`cka`
  cluster early; `effective_dim` sits alone in the middle.

## 2. Ruling out bugs before landing on "real finding"

Per CLAUDE.md: report failures plainly, don't tune thresholds to force a
pass. Before accepting the above as a real (if unexpected) property of this
model + lens, the following candidate bugs were tested and ruled out:

- **`concept_direction`'s RMSNorm-gain approximation.** Compared its scoring
  against the exact `readout()` path (real transport + RMSNorm) at several
  layers: 22-24/25 top-k overlap, Spearman ρ≈0.96-0.97 every time. Not the
  cause.
- **Concept-dictionary size.** Occupancy at one layer/prompt, swept from 200
  to 2,000 to 20,000 to the full ~248k-token vocab: mean occupancy stayed in
  a 2.1-2.9 band regardless of dictionary size. Not the cause.
- **Attention-sink position contamination.** An earlier run using ~16-token
  prompts probed positions 0-15 exclusively — precisely the positions
  `jlens.fitting.SKIP_FIRST_N_POSITIONS` (=16) excludes from the lens's own
  fit. Rerunning with longer prompts and positions restricted to
  `valid_position_mask`'s definition of valid moved the numbers by tenths
  (first-third mean 2.42 → 2.38; middle-band 2.80 → 2.95) — a real fix,
  worth keeping, but not the order-of-magnitude difference that would
  explain the gap.
- **Lens data-starvation.** The Hub-fitted lens (`qwen-n1000`) converged
  well before its 1000-prompt budget (417 prompts used,
  `stop_at_delta=0.002` reached, per its `config.yaml`). Not simple
  undersampling.

Nothing found points at a bug in this package.

## 3. Follow-up: `J_l` transport fidelity by depth

`readout()` (and everything `occupancy`/`fve`/`loading` build on) decodes
`unembed(J_l @ h_l)`. This measures how well `J_l @ h_l` actually resembles
the model's real final-layer residual `h_final` from the same forward pass —
cosine similarity and R² (fraction of variance explained), pooled over
positions across 3 prompts, at every one of the lens's 31 fitted layers:

| layer | cosine (mean) | R² (mean) | | layer | cosine (mean) | R² (mean) |
|---|---|---|---|---|---|---|
| 0 | 0.024 | -0.0002 | | 16 | 0.521 | 0.234 |
| 1 | 0.070 | 0.006 | | 17 | 0.528 | 0.248 |
| 2 | 0.086 | 0.008 | | 18 | 0.540 | 0.272 |
| 3 | 0.184 | 0.030 | | 19 | 0.574 | 0.314 |
| 4 | 0.241 | 0.046 | | 20 | 0.608 | 0.359 |
| 5 | 0.248 | 0.051 | | 21 | 0.626 | 0.389 |
| 6 | 0.298 | 0.067 | | 22 | 0.645 | 0.419 |
| 7 | 0.340 | 0.082 | | 23 | 0.681 | 0.467 |
| 8 | 0.373 | 0.099 | | 24 | 0.711 | 0.507 |
| 9 | 0.399 | 0.117 | | 25 | 0.727 | 0.522 |
| 10 | 0.409 | 0.129 | | 26 | 0.737 | 0.524 |
| 11 | 0.412 | 0.136 | | 27 | 0.808 | 0.632 |
| 12 | 0.445 | 0.161 | | 28 | 0.852 | 0.701 |
| 13 | 0.471 | 0.182 | | 29 | 0.887 | 0.748 |
| 14 | 0.484 | 0.198 | | 30 | 0.927 | 0.835 |
| 15 | 0.494 | 0.207 | | | | |

Fidelity rises smoothly and monotonically with depth — no jump or reversal
anywhere that would suggest the hybrid linear/full-attention layers broke
the fit specifically. This is the shape any single averaged Jacobian would
plausibly produce linearizing a deep nonlinear stack (the paper's own models
included), not evidence of a defect. But it does mean that across the
mid-depth layers `subsample_layers` samples most densely — and where the
paper's occupancy plateau lives — `J_l @ h_l` explains well under 40% of
`h_final`'s actual variance. Most of what the network still goes on to do
with that residual is, by construction, invisible to a purely linear
readout at that depth.

## 4. Where this leaves things

The leading explanation for the occupancy/boundary failures is scale: this
is a 4B model with a lens fit on a few hundred wikitext excerpts, not
whatever scale the paper's own experiments used. The transport-fidelity
result is a concrete, quantified mechanism consistent with that explanation
— but it doesn't fully resolve it. It can't distinguish "this model's
workspace really is narrower" from "a bigger model would show the same
fidelity curve at the same depths and still hit a 25-concept plateau
anyway." Answering that needs a second, larger model to compare against
(the paper's own reference model, ~54GB in bf16, doesn't fit this
project's dev machine — see the README).

Separately: M4's ablation/steering verification found real, specific causal
effects from individual concept directions on this same model and lens
(see the README's worked example) — clearly separated from a matched random
control. Whatever is limiting the occupancy/boundary read on this model, it
isn't that concept directions do nothing.
