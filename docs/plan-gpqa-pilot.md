# Pilot: GPQA Diamond across the quantization ladder

**Status:** planned, not yet run
**Model:** `Qwen/Qwen3-32B` (see [Model selection](#model-selection))
**Target system:** TACC Vista (`gh` queue, 1× H200 96 GB per node)
**Estimated cost:** ~20 SU

The goal of this pilot is to shake out the harness end-to-end on a single
model, not to produce a publishable result. GPQA Diamond is small enough to
iterate on quickly and hard enough that quantization damage should be visible
if it exists anywhere.

## Model selection

The ladder below assumes `Qwen/Qwen3-32B`. Qwen3 ships 14B, 32B and the
30B-A3B MoE; there is no 27B in the family (27B is Gemma 3's size). If the
intended target is 30B-A3B, Gemma 3 27B, or a later Qwen release, only the
model ID and the file sizes change — the experimental structure is unchanged.

Pin the exact revision hash of whatever is chosen and record it with the
results.

## The statistical problem, and the design that works around it

GPQA Diamond has **198 items**. At ~50% accuracy the binomial standard error
is ~3.6 points, giving a 95% CI of roughly ±7 points on any single score. A
Q4 quant scoring 3 points below BF16 is therefore indistinguishable from
noise, and comparing aggregate scores across the ladder would be a waste of
compute.

Every quant answers the *same* 198 items, so the comparison is **paired**:

- Score each item per quant and build the flip matrix against the BF16
  reference (BF16 correct → quant wrong, and the reverse).
- Test with McNemar rather than comparing overlapping confidence intervals.
- Treat the per-item flips as the primary artifact. *Which* questions break
  first is the finding; the scalar accuracy is not.

### Reasoning models amplify variance

Qwen3 is a reasoning model. With thinking enabled it emits thousands of
tokens per item, so a single perturbed token early in the trace cascades
through everything after it. Accordingly:

- Run **both thinking and non-thinking modes**.
- Take **n ≥ 5 samples per item** in thinking mode. The spread is a real
  effect being measured, not a nuisance to average away.

## Steps

### 1. Pin the baseline

Download `Qwen/Qwen3-32B` to `$WORK`. BF16 is ~64 GB and fits on one Vista
H200. Record the revision hash.

### 2. Build the quant ladder

GGUF via llama.cpp — it has the richest ladder and it is what the target
audience actually runs. Convert once; run the quantization on Frontera's CPU
nodes while they still exist (queues close 2026-10-01).

| Quant | Approx. size |
|---|---|
| BF16 | 64 GB |
| Q8_0 | ~34 GB |
| Q6_K | ~27 GB |
| Q5_K_M | ~23 GB |
| Q4_K_M | ~20 GB |
| IQ4_XS | ~18 GB |
| Q3_K_M | ~16 GB |

Keep these in `$WORK`. `$SCRATCH` purges files not accessed in 10 days, and
login-node reads do not refresh the access time — these weights are slow to
regenerate.

### 3. Wire the harness

Run `llama-server` (OpenAI-compatible) per GPU and point `lm-eval` at it via
the `local-completions` backend, task `gpqa_diamond_cot_zeroshot`. Decoupling
the backend from the eval means the same harness can drive four servers on
one Stampede3 node.

- Thinking mode: `temp=0.6, top_p=0.95, top_k=20`
- Non-thinking mode: `temp=0.7`
- Context: 16384, to hold full traces
- Log the complete generation plus per-item correctness to **one JSONL per
  cell** — not one file per question (Lustre metadata load)

### 4. Pilot before the sweep

Run 20 items at BF16 and Q4_K_M only. This confirms answer parsing and, more
importantly, that thinking traces are not being truncated. Silent truncation
looks exactly like quantization damage and is the most likely way this pilot
produces a confidently wrong answer.

### 5. Run the matrix

7 quants × 2 modes × 5 samples × 198 items ≈ 2M generated tokens per cell,
or roughly 1 GPU-hour per cell with batching. Compute is not the constraint
here; re-running is cheap.

### 6. Measure KL divergence alongside

This is the project's central claim, so make the comparison direct: capture
next-token KL against the BF16 reference on the same 198 prompts. The
deliverable is a statement about where KL predicted degradation that GPQA did
not show, and where it missed degradation that GPQA did.

### 7. Analyze

- Accuracy with confidence intervals
- McNemar, each quant vs. BF16
- Flip rate and flip direction
- Per-subject breakdown (physics / chemistry / biology)
- Thinking-trace length vs. quant level — trace length plausibly drifts before
  accuracy does, which would make it a useful early-warning signal

## Known limitations

**198 items is too few to carry conclusions.** This pilot is a pipeline
shakedown. Once the harness works, add a larger benchmark (MMLU-Pro, ~12k
items); the paired-item machinery built here transfers directly.

**A 32B model does not fit 16 GB even at Q3**, so this pilot cannot run on
Frontera's RTX 5000 nodes and is Vista-only. That gap is itself worth
recording: the 27–32B class is out of reach for the consumer VRAM tier this
project is fundamentally about. The consumer-hardware arm needs smaller
models.
