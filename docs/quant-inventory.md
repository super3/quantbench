# Qwen3.8-27B quant inventory

Every Qwen3.8-27B quant from Qwen and the established quant publishers,
pulled from the Hugging Face API on 2026-11-14. One row per repo. Sizes are
in GB, as reported by Hugging Face (10^9 bytes). Abliterated, uncensored,
merged and finetuned variants are excluded.

Repos change after publication. Pin the revision hash of every file used in a
run and record it with the results.

| Publisher | Repo | Format / runtime | Quants (size GB) |
|---|---|---|---|
| **Qwen (official)** | `Qwen/Qwen3.8-27B` | BF16 safetensors · vLLM/SGLang/Transformers | BF16 (55.6) |
| **Qwen (official)** | `Qwen/Qwen3.8-27B-FP8` | FP8 e4m3, block 128 · vLLM/SGLang | FP8 (30.9) |
| **NVIDIA** | `nvidia/Qwen3.8-27B-NVFP4` | ModelOpt NVFP4, mixed with FP8 · Blackwell | NVFP4 (21.9) |
| **Red Hat** | `RedHatAI/Qwen3.8-27B-FP8` | FP8 block 128 · vLLM | FP8 (30.9) |
| **Red Hat** | `RedHatAI/Qwen3.8-27B-INT4` | INT4 W4A16, group 128 · vLLM | INT4 (19.5) |
| **Red Hat** | `RedHatAI/Qwen3.8-27B-NVFP4` | NVFP4, mixed with FP8 · vLLM, Blackwell | NVFP4 (24.7) |
| **Red Hat** | `RedHatAI/Qwen3.8-27B-MXFP4` | MXFP4 · vLLM, Blackwell | MXFP4 (19.8) |
| **Red Hat** | `RedHatAI/Qwen3.8-27B-MXFP4xFP8` | MXFP4 mixed with FP8 · vLLM, Blackwell | MXFP4+FP8 (22.7) |
| **Intel** | `Intel/Qwen3.8-27B-bpw2.8-AutoRound` | AutoRound 2-bit, group 64 · vLLM | 2.8 bpw (14.8) |
| **Intel** | `Intel/Qwen3.8-27B-q4km-AutoRound` | GGUF, AutoRound-tuned · llama.cpp | Q4_K_M (16.8) |
| **Intel** | `Intel/Qwen3.8-27B-q2ks-AutoRound` | GGUF, AutoRound-tuned · llama.cpp | Q2_K_S (10.4) |
| **Intel** | `intel-ai/Qwen3.8-27B-MXFP4-AutoRound` | MXFP4 · vLLM, Blackwell | MXFP4 (21.6) |
| **Intel** | `intel-ai/Qwen3.8-27B-MXFP8-MXFP8MTP-CT-AutoRound` | MXFP8 · vLLM | MXFP8 (31.6) |
| **AMD** | `amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16` | Quark AWQ INT4 · vLLM | INT4 (19.5) |
| **AMD** | `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` | Quark AWQ MXFP4 · vLLM | MXFP4 (19.8) |
| **AMD** | `amd/Qwen3.8-27B-w8a8-llmcompressor` | INT8 W8A8 · vLLM | W8A8 (35.9) |
| **turboderp** | `turboderp/Qwen3.8-27B-exl3` | EXL3 · ExLlamaV3, NVIDIA only | One branch per bpw: 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, plus experimental SC_ branches from 1.4 to 6.0 |
| **unsloth** | `unsloth/Qwen3.8-27B-FP8` | FP8 block 128 · vLLM | FP8 (30.9) |
| **unsloth** | `unsloth/Qwen3.8-27B-NVFP4` | NVFP4, mixed with FP8 · vLLM, Blackwell | NVFP4 (23.4) |
| **unsloth** | `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` | bitsandbytes · Transformers | 4-bit (22.3) |
| **unsloth** | `unsloth/Qwen3.8-27B-GGUF` | GGUF · llama.cpp; quants marked UD are unsloth's dynamic mixes | BF16 (54.7), Q8_K_XL (31.5), Q8_0 (29.1), Q8_K_L (28.1), Q6_K_XL (25.3), Q6_K_L (24.2), Q6_K_M (23.1), Q6_K (22.0), Q5_K_XL (21.0), Q5_K_M (19.9), Q5_K_S (18.8), Q4_K_XL (17.7), Q4_1 (17.5), Q4_K_M (16.6), Q4_0 (16.1), Q4_K_S (15.4), IQ4_XS (14.4), Q3_K_XL (13.2), IQ3_S (12.0), IQ3_XXS (10.9), Q2_K_XL (10.0), IQ2_S (8.4), IQ2_XS (7.6), IQ2_XXS (7.2), IQ1_M (6.7), IQ1_S (6.2) |
| **bartowski** | `bartowski/Qwen3.8-27B-GGUF` | GGUF with imatrix · llama.cpp | BF16 (54.7), Q8_0 (29.1), Q6_K_L (25.0), Q6_K (23.9), Q6_K_S (22.9), Q5_K_L (21.5), Q5_K_M (20.9), Q5_K_S (19.6), Q4_K_L (18.8), Q4_1 (17.8), Q4_K_M (17.4), IQ4_NL (17.4), Q3_K_XL (16.4), Q4_K_S (16.4), Q4_0 (16.4), IQ4_XS (15.5), IQ3_M (14.9), Q3_K_L (14.1), Q3_K_M (13.4), Q2_K_L (13.1), IQ3_XS (12.8), Q3_K_S (12.7), IQ3_XXS (12.3), Q2_K (10.8), IQ2_M (10.5), IQ2_S (9.7), IQ2_XS (9.1), IQ2_XXS (8.9) |
| **mradermacher** | `mradermacher/Qwen3.8-27B-GGUF` | GGUF, static (no imatrix) · llama.cpp | Q8_0 (29.1), Q6_K (22.4), Q5_K_M (19.5), Q5_K_S (19.0), Q4_K_M (16.8), Q4_K_S (15.8), IQ4_XS (15.4), Q3_K_L (14.6), Q3_K_M (13.5), Q3_K_S (12.3), Q2_K (10.9) |
| **mradermacher** | `mradermacher/Qwen3.8-27B-i1-GGUF` | GGUF with imatrix · llama.cpp | Q6_K (22.4), Q5_K_M (19.5), Q5_K_S (19.0), Q4_1 (17.3), Q4_K_M (16.8), Q4_K_S (15.8), Q4_0 (15.8), IQ4_XS (15.3), Q3_K_L (14.6), Q3_K_M (13.5), IQ3_M (12.8), IQ3_S (12.6), Q3_K_S (12.3), Q2_K (10.9) |
| **ggml-org** | `ggml-org/Qwen3.8-27B-GGUF` | GGUF · llama.cpp | BF16 (53.8), Q8_0 (28.6), Q4_K_M (19.4) |
| **lmstudio-community** | `lmstudio-community/Qwen3.8-27B-GGUF` | GGUF · llama.cpp | Q8_0 (29.1), Q6_K (22.4), Q4_K_M (16.8) |
| **lmstudio-community** | `lmstudio-community/Qwen3.8-27B-MLX-{4,5,6,8}bit` | MLX · Apple only | 4-, 5-, 6- and 8-bit (four repos) |
| **ISTA-DASLab** | `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` | GGUF, GSQ-RCO method · llama.cpp | IQ3_S (11.8), IQ3_XXS (10.1), IQ2_S (9.3), IQ2_XS (8.4); each also has an `-mtp` variant |
| **ISTA-DASLab** | `ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ` | GSQ 3-bit · vLLM | 3-bit |
| **ISTA-DASLab** | `ISTA-DASLab/Qwen3.8-27B-NVFP4-prefiller` | NVFP4 · Blackwell | NVFP4 |
| **byteshape** | `byteshape/Qwen3.8-27B-GGUF` | GGUF · llama.cpp | IQ4_XS 3.84 bpw (13.1), IQ3_S 3.23 bpw (11.0), IQ3_XS 3.01 bpw (10.4), IQ3_XXS 2.88 bpw (9.9), IQ2_XXS 2.56 bpw (8.8) |
| cyankiwi (popular community release) | `cyankiwi/Qwen3.8-27B-AWQ-INT4` | AWQ INT4, group 32 · vLLM | INT4 (21.0) |
| RadixArk (popular community release) | `RadixArk/Qwen3.8-27B-NVFP4` | ModelOpt NVFP4, mixed · Blackwell | NVFP4 (21.9) |

Sizes were not checked for the ISTA-DASLab 3-bit GSQ and NVFP4 repos or for
the MLX repos.
