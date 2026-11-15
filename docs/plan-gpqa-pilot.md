# Pilot: GPQA Diamond across the quantization ladder

**Status:** planned, not yet run
**Model:** `Qwen/Qwen3.8-27B` (see [Model selection](#model-selection))
**Target systems:** local — AMD Strix Halo (128 GB unified), NVIDIA RTX PRO
6000 Blackwell (96 GB), Radeon 7900 (see [Hardware](#hardware))

The goal of this pilot is to shake out the harness end-to-end on a single
model, not to produce a publishable result. GPQA Diamond is small enough to
iterate on quickly and hard enough that quantization damage should be visible
if it exists anywhere.

## Model selection

`Qwen/Qwen3.8-27B` is a dense 27B model built on the Qwen3.5 architecture.
Two properties matter here:

- **It mixes attention types.** Three of every four layers use linear
  attention and the fourth uses full attention, so it
  needs a recent llama.cpp or vLLM build. Record the engine version alongside
  every result.
- **It is natively vision-language.** The GGUF repos ship `mmproj` files for
  the vision encoder. This pilot is text-only, so they aren't needed.

Qwen reports **89.2% on GPQA Diamond** for this model in thinking mode. That
is the number the BF16 baseline should land near. If it doesn't, suspect the
harness (parsing, truncation, sampling) before the model.

Pin the revision hash of every repo used and record it with the results.

## Which quants

The inventory of everything published is in
[`quant-inventory.md`](quant-inventory.md). This pilot tests **every quant
from unsloth and bartowski, plus Qwen's official files**:

- **unsloth** — the most-downloaded quants of this model by a wide margin.
  Most rungs are unsloth's "UD" dynamic mixes, not standard llama.cpp quants.
- **bartowski** — standard llama.cpp quant types, all calibrated with an
  imatrix. Comparing the two at matched sizes separates the effect of bit
  width from the effect of unsloth's recipe.
- **Qwen (official)** — BF16 as the reference, and the official FP8. These
  are the only two formats Qwen publishes for this model.

### GGUF (llama.cpp)

One BF16 reference: unsloth's BF16 GGUF (54.7 GB). bartowski's bf16 GGUF
should be an identical conversion; check it with KL divergence (expect ~0)
rather than a full GPQA run.

**unsloth — `unsloth/Qwen3.8-27B-GGUF` (25 quants).** All are UD dynamic
mixes except Q8_0, Q4_0 and Q4_1.

| Bits | Quants (GB) |
|---|---|
| 8 | Q8_K_XL (31.5), Q8_0 (29.1), Q8_K_L (28.1) |
| 6 | Q6_K_XL (25.3), Q6_K_L (24.2), Q6_K_M (23.1), Q6_K (22.0) |
| 5 | Q5_K_XL (21.0), Q5_K_M (19.9), Q5_K_S (18.8) |
| 4 | Q4_K_XL (17.7), Q4_1 (17.5), Q4_K_M (16.6), Q4_0 (16.1), Q4_K_S (15.4), IQ4_XS (14.4) |
| 3 | Q3_K_XL (13.2), IQ3_S (12.0), IQ3_XXS (10.9) |
| 2 | Q2_K_XL (10.0), IQ2_S (8.4), IQ2_XS (7.6), IQ2_XXS (7.2) |
| 1 | IQ1_M (6.7), IQ1_S (6.2) |

**bartowski — `bartowski/Qwen3.8-27B-GGUF` (27 quants).**

| Bits | Quants (GB) |
|---|---|
| 8 | Q8_0 (29.1) |
| 6 | Q6_K_L (25.0), Q6_K (23.9), Q6_K_S (22.9) |
| 5 | Q5_K_L (21.5), Q5_K_M (20.9), Q5_K_S (19.6) |
| 4 | Q4_K_L (18.8), Q4_1 (17.8), Q4_K_M (17.4), IQ4_NL (17.4), Q4_K_S (16.4), Q4_0 (16.4), IQ4_XS (15.5) |
| 3 | Q3_K_XL (16.4), IQ3_M (14.9), Q3_K_L (14.1), Q3_K_M (13.4), IQ3_XS (12.8), Q3_K_S (12.7), IQ3_XXS (12.3) |
| 2 | Q2_K_L (13.1), Q2_K (10.8), IQ2_M (10.5), IQ2_S (9.7), IQ2_XS (9.1), IQ2_XXS (8.9) |

The same name does not mean the same file: unsloth's UD-Q4_K_M and
bartowski's Q4_K_M differ by 0.8 GB, and bartowski's Q3_K_XL is larger than
its Q4_K_S. Compare quants by measured size (bits per weight) as well as by
name.

### Native formats (vLLM, RTX PRO 6000 only)

| Repo | Format | Size |
|---|---|---|
| `Qwen/Qwen3.8-27B` | BF16 (reference) | 55.6 GB |
| `Qwen/Qwen3.8-27B-FP8` | FP8, official | 30.9 GB |
| `unsloth/Qwen3.8-27B-FP8` | FP8 | 30.9 GB |
| `unsloth/Qwen3.8-27B-NVFP4` | NVFP4, mixed with FP8 | 23.4 GB |
| `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` | bitsandbytes 4-bit | 22.3 GB |

unsloth's FP8 has the same layout as Qwen's (e4m3, 128×128 blocks). If the two
turn out to be bit-identical, drop one.

**Run the bitsandbytes cell last.** bitsandbytes is mainly a fine-tuning
format: it is what QLoRA trains on, and few people serve a model with it.
This file uses NF4 with double quantization and keeps `lm_head`, the vision
encoder and some linear-attention projections (`in_proj_qkv`, `in_proj_a`,
`in_proj_b`) in 16-bit, which is why it is larger than the 4-bit GGUFs. It
answers whether the copy people fine-tune on behaves like the copy they run.
But it is the slowest cell to generate with and the least representative of
everyday use, so it is the first cell to drop if time runs short.

**Never compare across engines.** vLLM and llama.cpp differ in kernels,
sampling and numerics. Each engine's quants are scored against that engine's
own BF16 run, and the gap between the two BF16 runs measures the engine
effect itself.

## Hardware

| Machine | Memory | Runs | Role |
|---|---|---|---|
| AMD Strix Halo | 128 GB unified, ~256 GB/s | Every GGUF, BF16 included (llama.cpp, Vulkan) | Harness development, the 20-item pilot, KL divergence |
| RTX PRO 6000 Blackwell | 96 GB, ~1.8 TB/s | Everything: every GGUF (llama.cpp, CUDA), plus BF16, FP8 and NVFP4 (vLLM) | The full matrix |
| Radeon 7900 | 24 GB (XTX) or 48 GB (PRO W7900) | GGUF only, through llama.cpp (ROCm or Vulkan): up to Q5 on 24 GB, up to Q8_0 on 48 GB | Optional cross-vendor check |

Generation speed is limited by memory bandwidth, so the Strix Halo is slow at
producing long thinking traces: roughly 4 tok/s single-stream at BF16 and
about 13 tok/s at Q4. That is fine for KL divergence, which only processes
text, and for the 20-item pilot. It is not enough for the full matrix, which
belongs on the RTX PRO 6000.

On the Strix Halo, raise the GPU memory limit (the BIOS UMA setting and, on
Linux, the GTT limit) before loading BF16. The default is often far below
what the machine can support.

## The statistical problem, and the design that works around it

GPQA Diamond has **198 items**. At ~89% accuracy the binomial standard error
is ~2.2 points, giving a 95% CI of roughly ±4 points on any single score. A
Q4 quant scoring 2 points below BF16 is therefore indistinguishable from
noise, and comparing aggregate scores across the ladder would be a waste of
compute.

High baseline accuracy also means BF16 gets only about 21 items wrong. Nearly
all the signal is in the ~177 items BF16 answers correctly, and in how many
of those each quant loses.

Every quant answers the *same* 198 items, so the comparison is **paired**:

- Score each item per quant and build the flip matrix against the BF16
  reference (BF16 correct → quant wrong, and the reverse).
- Test with McNemar rather than comparing overlapping confidence intervals.
- Treat the per-item flips as the primary artifact. *Which* questions break
  first is the finding; the scalar accuracy is not.

### Reasoning models amplify variance

Qwen3.8 thinks by default and emits thousands of tokens per item, so a single
perturbed token early in the trace cascades through everything after it.
Accordingly:

- Run **both thinking and non-thinking modes**.
- Take **n ≥ 5 samples per item** in thinking mode. The spread is a real
  effect being measured, not a nuisance to average away.

## Steps

### 1. Pin the baseline

Download `unsloth/Qwen3.8-27B-GGUF` BF16 (the llama.cpp reference) and
`Qwen/Qwen3.8-27B` (the vLLM reference). Record both revision hashes.

### 2. Download the quant ladder

Download the published files in the tables above rather than quantizing
locally — the point is to test what people actually run. Record each repo's
revision hash and each file's SHA256. Publishers sometimes re-upload
quants under the same name.

### 3. Wire the harness

Run `llama-server` (OpenAI-compatible) per GPU and point `lm-eval` at it via
the `local-completions` backend, task `gpqa_diamond_cot_zeroshot`. Decoupling
the backend from the eval means the same harness can drive llama.cpp and vLLM
servers on any of the machines.

Use the sampling settings from the model card:

- Thinking mode: `temperature=1.0, top_p=0.95, top_k=20, min_p=0,
  presence_penalty=0`, `reasoning_effort` left at its default (`xhigh`)
- Non-thinking mode (`enable_thinking: false`): `temperature=0.7, top_p=0.8,
  top_k=20, min_p=0, presence_penalty=1.5`
- Max output: start at 32768 tokens and adjust after step 4. The model card
  recommends far longer limits for agentic work, and Qwen's 89.2% was
  presumably measured with long traces
- Log the complete generation plus per-item correctness to **one JSONL per
  cell**, not one file per question

### 4. Pilot before the sweep

On the Strix Halo, run 20 items at BF16 and UD-Q4_K_M only. This confirms
answer parsing and, more importantly, that thinking traces are not being
truncated. Silent truncation looks exactly like quantization damage and is
the most likely way this pilot produces a confidently wrong answer.

Record the trace-length distribution here, at the 50th, 95th and maximum. It
sets the output limit for the sweep and the compute estimate in step 5.

### 5. Run the matrix

Cells: 53 GGUF (BF16 plus 25 unsloth and 27 bartowski quants) and 5 vLLM,
58 in all, each in thinking and non-thinking mode.

Tokens per cell ≈ 198 items × samples × mean trace length. At an assumed
8k-token mean trace and 5 samples, a thinking-mode cell is ~8M tokens, so
all 58 thinking cells come to roughly 460M generated tokens. Non-thinking
cells are far smaller. Replace the assumed trace length with the one measured
in step 4, and measure the RTX PRO 6000's batched throughput, before
committing to the full sweep.

Run it in stages so every quant has a result early and the expensive part
comes last:

1. **Perplexity and KL divergence** (step 6): both on all 52 GGUF quants,
   and perplexity on the vLLM cells. No generation, so it is cheap.
2. **Non-thinking mode on all 58 cells.** Short answers, so this is cheap
   too.
3. **Thinking mode, 1 sample per item, on all 58 cells.**
4. **Thinking mode, up to 5 samples per item,** in that order of priority
   if time runs short: the rungs where stages 1–3 show the knee in the
   curve, then the rest.

#### Time estimate: one RTX PRO 6000

Assumptions, to be replaced with measurements from step 4:

- **Throughput** (total across parallel requests): ~500 tok/s for GGUF quants
  in llama.cpp (plausible range 300–1,000) and ~1,000 tok/s for FP8 and
  NVFP4 in vLLM.
- **Mean output length:** 8k tokens per answer in thinking mode, ~1k in
  non-thinking mode.

| Stage | Tokens generated | Time at ~500 tok/s | Plausible range |
|---|---|---|---|
| Download all 59 files | ~1.1 TB | ~3 h on a 1 Gbps line | — |
| 1. Perplexity and KL, all quants | almost none | ~1–2 h, mostly model loading | — |
| 2. Non-thinking, 58 cells × 1 sample | ~12M | ~6 h | 3–11 h |
| 3. Thinking, 58 cells × 1 sample | ~92M | ~2 days | 1–3.5 days |
| 4. Thinking, 4 more samples per item | ~370M | ~8.5 days | 4–14 days |
| **Total** | **~470M** | **~11 days** | **~6–18 days** |

Stages 1–3 take about 2–3 days and give every quant a GPQA score. Stage 4
is most of the cost and can stop partway.

What moves the estimate:

- **Trace length scales it linearly.** At the default `xhigh` reasoning
  effort, GPQA traces may average 12–15k tokens rather than 8k, which would
  push the full run past 3 weeks.
- **Throughput scales it inversely.** In step 4, spend an hour measuring
  tok/s at 16, 32 and 64 parallel requests, at BF16 and at Q4.
- **The bitsandbytes cell is slow,** perhaps 100–200 tok/s, so at 5 samples it
  alone takes most of a day.

Ways to cut it if needed:

- 3 thinking samples per item instead of 5: saves ~40% of the total.
- `reasoning_effort: medium`: shorter traces, but the BF16 baseline will no
  longer be comparable to Qwen's reported 89.2%.
- Drop near-duplicate quants, such as unsloth's Q8_K_L next to Q8_0, or
  Q4_0 and Q4_1. Each dropped cell saves ~4–5 hours at full depth.

Disk: all 59 files take ~1.1 TB. Download and delete in batches if space is
short.

### 6. Measure perplexity and KL divergence alongside

This is the project's central claim, so make the comparison direct: capture
the two metrics practitioners normally rely on, on the same 198 prompts, and
compare them with GPQA.

- **Perplexity**: how well the quant predicts the actual text. It needs no
  reference model, but it only looks at the probability given to the correct
  next token, so errors can cancel out.
- **KL divergence against BF16**: how far the quant's full next-token
  distribution has drifted from the reference. It is more sensitive than
  perplexity.

The deliverable is a statement, for each metric, about where it predicted
degradation that GPQA did not show, and where it missed degradation that
GPQA did. If KL tracks GPQA better than perplexity does, or neither does,
that is itself a finding.

Use `llama-perplexity`: save the BF16 logits once with
`--kl-divergence-base`, then score each GGUF quant with `--kl-divergence`.
One run per quant records all of:

- perplexity, and its change from BF16
- mean KL, plus the 99th-percentile and maximum KL (the mean hides the tail)
- "same top token %": how often the quant's most likely token matches BF16's

The logits file is large, roughly 0.5 MB per token with this model's ~248k
vocabulary, so keep the scored text to the GPQA prompts. The file is
portable, so the 7900 can score its quants against a reference generated on
another machine.

Coverage differs for the vLLM cells. Perplexity only needs the probability
of the actual next token, which vLLM returns (`prompt_logprobs`), so it
covers every cell. KL needs the full distribution, and vLLM returns only the
top few log-probabilities per token, so full KL for FP8, NVFP4 and
bitsandbytes is not available without extra tooling.

### 7. Analyze

- Accuracy with confidence intervals
- McNemar, each quant vs. BF16
- Flip rate and flip direction
- Perplexity and KL vs. GPQA flips, quant by quant: which metric, if
  either, predicts the task damage?
- Accuracy, perplexity and KL plotted against bits per weight, one curve per
  publisher:
  at the same size, does unsloth's recipe beat bartowski's, or does only the
  bit width matter?
- Within a publisher, quants of the same size but different types (for
  example Q4_K_S vs. IQ4_NL vs. Q4_0): does the quant type matter at a fixed
  size?
- Per-subject breakdown (physics / chemistry / biology)
- Thinking-trace length vs. quant level — trace length plausibly drifts before
  accuracy does, which would make it a useful early-warning signal

## Known limitations

**198 items is too few to carry conclusions.** This pilot is a pipeline
shakedown. Once the harness works, add a larger benchmark (MMLU-Pro, ~12k
items); the paired-item machinery built here transfers directly.

**GPQA is close to this model's ceiling.** At 89%, a quant can only show
damage on the ~177 items BF16 gets right, and improvements are nearly
impossible to detect. A harder benchmark, or one where the model scores
nearer 50%, would give more room in both directions.

**The 16 GB consumer tier is only just reachable.** UD-IQ4_XS (14.4 GB) and
the 3-bit rungs (~13 GB) fit a 16 GB card, but leave little room for the KV
cache that long thinking traces need. Whether a 27B model at 3–4 bits with a
short context beats a smaller model at higher precision is exactly the
tradeoff the project's VRAM map is meant to settle.
