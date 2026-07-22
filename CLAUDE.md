# subvocal

A workspace occupancy profiler and prompt debugger built on the Jacobian lens.

## Context you need before writing code

Read these first:
- Paper: https://transformer-circuits.pub/2026/workspace/index.html
- Upstream repo: jacobian-lens/ (already checked out, read the README and jlens/fitting.py docstring)

The Jacobian lens computes `J_l = E[dh_final/dh_l]` averaged over prompts and
positions, then decodes `unembed(J_l @ h)` into a ranked token list. The J-space
is the set of points expressible as a sparse nonnegative combination of J-lens
vectors. It holds roughly 25 concepts at a time and accounts for under 10% of
activation variance.

We are NOT reimplementing the lens. We import `jlens` and build measurement and
debugging tooling on top.

## Environment

- Dev machine is an Apple M4, device is `mps`, not `cuda`
- `PYTORCH_ENABLE_MPS_FALLBACK=1` must be set
- MPS does not support fp64. Never use float64. If a numerical routine needs it,
  use float32 and add a CPU-reference test
- Centralize all device handling in `subvocal/lens.py`. No `.cuda()` or bare
  `.to(device)` anywhere else in the package
- Model: pinned to `Qwen/Qwen3.5-4B` (~9.3GB in bf16, fits M4 unified memory).
  Uses the pre-fitted lens already on the Hub (`neuronpedia/jacobian-lens`,
  revision `qwen-n1000`) instead of a Modal fitting run -- see
  `subvocal.lens.QWEN3_5_4B`. It's a hybrid linear-attention/full-attention
  architecture (`Qwen3_5ForCausalLM`), not a plain transformer; `jlens.from_hf`
  and `ActivationRecorder` were smoke-tested against it and work (falls back
  to a pure-PyTorch path for the linear-attention layers, confirmed on MPS)
- A fitted lens lives at the Hub location above, wrapped by
  `subvocal.lens.FittedLens`. `StubLens` (Milestone 1) is still what tests run
  against; only ad hoc sanity-check scripts should load the real model

## Module layout

Exactly these. Do not add modules without asking.

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

## Milestones

Stop at the end of each and report. Do not continue past a gate without me.

### M1: Profile interface + stub

Define the `Profile` object and freeze it. Everything downstream depends on this
interface, so get it right before implementing anything.

```python
Profile.loading(concept: str) -> ndarray      # (n_pos, n_layer)
Profile.occupancy() -> ndarray                # (n_pos, n_layer)
Profile.boundaries() -> Boundaries            # per-signal, with disagreement flag
Profile.topk(pos, layer, k=25) -> list[tuple[str, float]]
Profile.fve() -> ndarray                      # (n_layer,)
Profile.save(path) / Profile.load(path)
```

Build a `StubLens` that returns deterministic fake readouts with the right
shapes, so metrics and tests can be developed before the real lens lands.

Gate: interface reviewed by me. Do not proceed until I confirm.

### M2: metrics.py

Implement against the paper's definitions:

- **loading(concept, h)**: cosine similarity between residual stream and the
  concept's J-lens vector, averaged over specified positions
- **occupancy(h)**: sparse nonnegative reconstruction from K J-lens vectors via
  gradient pursuit. Sweep K, threshold at the point where marginal improvement
  falls below a same-size random control set. Use an adaptive sweep, not
  exhaustive. Batch across positions
- **fve(h, K)**: variance explained by top-K J-lens vectors in excess of a
  same-size random control
- **boundaries()**: five independent signals, all reported, plus a flag when
  they disagree by more than 10% of depth:
  1. top-k accuracy of the lens at predicting the model's actual next token
  2. excess kurtosis of the readout logit distribution
  3. autocorrelation of top-1 lens token across positions vs a position-shuffled null
  4. effective linear dimensionality of `W_U @ J_l`
  5. CKA between layers' J-lens gram matrices

Subsample to 25 evenly spaced layers, reindexed to 0-100 as the paper does.

Sanity checks that must pass on the real lens:
- occupancy is near zero in the first third of layers and plateaus near 25 in
  the middle band
- fve never exceeds 0.10
- all five boundary signals land within 10% of each other

If any of these fail, STOP and tell me. Do not tune thresholds to make them pass.

Gate: sanity checks reported.

### M3: debug.py, contrast pairs

Three modes:

1. `probe(prompt, concept)` - user supplies the expected intermediate, returns
   loading trace and peak layer
2. `contrast(working_prompt, failing_prompt)` - the main feature. Diff J-space
   contents position by position and layer by layer. Return concepts strongly
   present in working and weak or absent in failing, ranked
3. `propose(prompt)` - stretch goal only, skip unless M1-M3 finish early

For `contrast`:
- Require the two prompts to be token-aligned. Raise a clear error if not
- Normalize the loading delta against a baseline of how much that concept varies
  across unrelated prompt pairs, or you will surface junk
- Report best rank over the workspace band (paper convention) AND expose the
  per-layer trace

Warn at the API boundary whenever a user-supplied concept tokenizes to more than
one token. The lens only sees single tokens and this is a real limitation, not
an edge case.

### M4: verification

- `verify_ablate(prompt, concept)` - project out the concept's J-lens direction
  across the workspace band, confirm the model's answer changes. Follow the
  paper: skip ablating any token in the top-10 of the clean forward pass, so we
  target internal reasoning rather than report
- `verify_steer(prompt, concept, alpha)` - steer the concept in, check recovery
- **Required control**: random-direction ablation matched for norm and layer
  band. Without it any effect could be generic perturbation damage. Report both
  numbers side by side, always

### M5: report.py + README

- HTML report reusing the d3 slice view from ../jacobian-lens. Do not build a
  new visualization
- README with one worked contrast-pair example showing a real diagnosed failure
- Limitations section stating plainly:
  - single-token vocabulary restriction
  - absence from the J-space does not prove the model never inferred the concept,
    it may have been used outside the workspace (see the paper's language and
    line-counting selectivity experiments)
  - results are from one small open model, not the models in the paper

## Rules

- Write tests as you go. `pytest` must pass at every gate
- Type hints on all public functions
- No new dependencies beyond torch, transformers, numpy, and whatever jlens
  already pulls in. Ask before adding anything
- Cache aggressively: residual streams per prompt, occupancy per (prompt, layer).
  These get recomputed constantly during development
- Call `torch.mps.empty_cache()` between heavy chunks
- When a sanity check fails, report it. Never adjust a threshold to make a check
  pass
- Do not optimize the upstream fitting code. That is explicitly out of scope