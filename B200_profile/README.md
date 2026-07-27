# NVIDIA B200 LLMServingSim Profile

Status: complete. Both phase-1 base profiles passed validation, and the
deadline-protected Qwen Skew pass also completed for TP 1/2/4. The Vast.ai
instance has been destroyed and the post-destroy active-instance count is 0.

## Run summary

- Vast.ai instance: `45910875`
- Hourly total at the billing snapshot: `$5.9666666667`
- Rental started: `2026-07-26T13:22:14Z`
  (`2026-07-26 21:22:14` China Standard Time)
- Llama base completed: `2026-07-26T14:41:03Z`
- Qwen base plus explicit TP2/TP4 MoE repair completed:
  `2026-07-26T15:27:50Z`
- Qwen Skew completed after one incremental resume:
  `2026-07-26T16:32:55Z`
- Final billing snapshot and instance destruction:
  `2026-07-26T16:39:48Z`
- Total rental time at the final snapshot: `3.2929` hours
- Estimated rental cost at the final snapshot: `$19.6477`
  (wall time x `dph_total`; the Vast.ai invoice is authoritative)
- Llama base: 4 TP directories, 12 required CSV files
- Qwen base + Skew: 3 TP directories, 18 CSV files including
  `skew.csv` and `skew_fit.csv`
- Total audited: 30 CSV files, 24,701 rows
- Independent numeric audit: all latency values finite and positive; no
  literal `nan` or `inf` cell tokens
- No model weights were downloaded.

## Fixed software state

- LLMServingSim commit: `c84e58b7fec0bc71b174481a7fe06b1d5589a5fc`
- ASTRA-Sim submodule commit: `f82fb3d861614a2a61febfaf87a1360db05efd81`
- Required vLLM version: `0.19.0`
- Required profiler add-on: `pandas>=2.2,<3` (`2.3.3` used in this run)
- Docker image:
  - CUDA 12.x server: `vllm/vllm-openai:v0.19.0`
  - CUDA 13.x-compatible server: `vllm/vllm-openai:v0.19.0-cu130`
- Profiler mode: dummy weights, one decoder layer, tokenizer initialization disabled

## Measured hardware

- Platform: Vast.ai
- GPU: 1 x NVIDIA B200
- Allocation: exclusive single GPU
- GPU memory reported by PyTorch device properties: `191495471104` bytes
  (about 178.3 GiB / 182624 MiB)
- HBM bandwidth configuration: 8.0 TB/s
- Docker image: `vllm/vllm-openai:v0.19.0-cu130`
- Driver version: `595.71.05`
- CUDA runtime used by PyTorch: `13.0`
- Compute capability: `10.0`

The provider image contains an `/usr/local/bin/nvidia-smi` workaround wrapper
whose target `/usr/bin/nvidia-smi` is absent. The scan therefore records this
provider anomaly verbatim and uses successful PyTorch CUDA initialization,
device properties, and `/proc/driver/nvidia/version` as the runtime hardware
evidence.

## Phase-1 scan parameters

Common:

- dtype: `bfloat16`
- max num batched tokens: `2048`
- max num sequences: `256`
- attention max KV: `16384`
- attention chunk factor: `4.0` (cost-optimized first-stage grid)
- attention KV factor: `4.0` (cost-optimized first-stage grid)
- measurement iterations: `3`
- skew: disabled
- resume: enabled by default

Models:

- `meta-llama/Llama-3.1-405B-Instruct`: TP `1,2,4,8`
- `Qwen/Qwen3-235B-A22B`: TP `1,2,4`

The three target workloads (1024/128, 2048/512, and 2048/2048 input/output
tokens) are not direct Profiler CLI inputs. They are covered by the layer and
attention grids; the selected 16K KV cap safely exceeds the approximately 4K
largest phase-1 sequence. The 4x cost-optimized attention grid explicitly
contains chunk points 1024 and 2048 plus KV points 1024 and 4096; KV=2048 is
interpolated between adjacent profiled points. This choice was made after the
default 2x TP1 attention grid had run for 26 minutes without finishing.

## Output layout

```text
B200_profile/
├── profiles/
│   └── B200/<org>/<model>/bf16/
│       ├── meta.yaml
│       └── tp<N>/
│           ├── dense.csv
│           ├── per_sequence.csv
│           ├── attention.csv
│           └── moe.csv        # Qwen only
├── logs/
├── hardware_info/
└── README.md
```

## Prepared commands

Dry-run locally:

```bash
DRY_RUN=1 ./scripts/profile_b200_llama405b.sh
DRY_RUN=1 ./scripts/profile_b200_qwen235b.sh
```

Verify and refresh the gated Llama metadata before any GPU run:

```bash
HF_TOKEN=... ./scripts/verify_llama405b_metadata.sh
```

Run inside the pinned vLLM container on the verified B200 server:

```bash
./scripts/profile_b200_llama405b.sh
./scripts/profile_b200_qwen235b.sh
```

Both scripts refuse to run unless exactly one visible B200 is present, reported memory is at least 175000 MiB, Blackwell compute capability is visible, and vLLM is exactly `0.19.0`. The Llama script additionally refuses to use a GPU until `verify_llama405b_metadata.sh` has produced an official verification record. Each run saves a timestamped log and hardware snapshot, resumes existing CSVs by default, and calls `check_profile.py` after completion.

The live rental used the already verified dummy-weight model metadata. Because
Meta's official Hugging Face config remained gated, the exact Llama metadata
provenance limitation described below still applies; this does not imply that
the gated file was accessed.

## Model metadata provenance

- Qwen: all three JSON files were copied from the official public `Qwen/Qwen3-235B-A22B` repository at revision `04127485f4b84439c34689fdee54a616531bf00d`.
- Llama: Meta's official repository returned HTTP 401 for all three JSON files. Local files were copied from NVIDIA's public `nvidia/Llama-3.1-405B-Instruct-FP8` repository at revision `b2cbce00b6238d2fe04939ee8c9851c78ac83046`. The config itself declares BF16 and contains no quantization block.
- Before spending GPU time, compare the local Llama `config.json` with the current official Meta file using an HF token that has accepted the Llama 3.1 license.

No model weights are required or downloaded.

## Skew policy

Phase 1 passes `--skip-skew`. After both base profiles pass validation, the
deadline-protected optional pass profiles Qwen only with `--only-skew` and
`--skew-{n,pc,kp,kvs}-factor 8.0`. This produces 645 feasible cases per TP
while retaining the fixed physical skew sweep at 1.5/2/4/8/16. It starts only
before `2026-07-26T15:35:00Z` (23:35 China Standard Time) and has a 60-minute
hard runtime limit so local download and instance destruction finish before
the 01:30 balance deadline. Llama Skew is intentionally omitted.

The first Qwen Skew pass reached TP1=645, TP2=645, TP4=600 rows at its
60-minute limit. The resume pass recognized and skipped all 1,890 existing
rows, measured only the final 45 TP4 cases, then generated per-TP
`skew_fit.csv` files. Final fit p99 relative errors were 3.36% (TP1), 3.05%
(TP2), and 2.49% (TP4).

## Validation

Server-side validation commands:

```bash
python3 check_profile.py \
  B200_profile/profiles/B200/meta-llama/Llama-3.1-405B-Instruct/bf16 \
  --model-id meta-llama/Llama-3.1-405B-Instruct \
  --hardware B200 --tp 1,2,4,8 --expected-vllm-version 0.19.0

python3 check_profile.py \
  B200_profile/profiles/B200/Qwen/Qwen3-235B-A22B/bf16 \
  --model-id Qwen/Qwen3-235B-A22B \
  --hardware B200 --tp 1,2,4 --moe --skew \
  --expected-vllm-version 0.19.0
```

Both passed. `logs/qwen235b-moe-repair.log` records the explicit official
`profiler slice` refresh of TP2/TP4 MoE files after the full runner's
tp-stable replication did not create them. `logs/qwen-skew-pass1-timeout.log`
and the final `logs/qwen-skew.log` preserve the bounded first pass and its
resume.
