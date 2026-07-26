---
title: HBF static-weight backend
---

The HBF backend models a large, GPU-attached memory tier for **static
model weights**. It is additive to the existing HBM, CPU, CXL, PIM,
KV-cache, prefix-cache, and parallelism paths.

## Scope

The first schema version deliberately has a narrow boundary:

- static embedding, projection, normalization, head, and MoE expert
  weights may reside in HBF;
- KV cache, prefix cache, activations, communication buffers, and
  dynamic expert caching remain on their existing paths;
- HBF has no ASTRA-Sim memory-backend type. The Python trace generator
  resolves the complete HBF-backed operator latency, then lowers the
  logical `HBF` location to `LOCAL` before Chakra conversion;
- HBF energy is not modeled. Runtime metadata sets
  `hbf_energy_unmodeled=true`, and HBF bytes are not charged to the
  DRAM/CXL energy model.

Configuring `kv_loc` or `kv_evict_loc` as `hbf` fails with an explicit
"reserved but not implemented" error. Per-stack placement such as
`hbf:3` is also intentionally unavailable in schema version 1.

## Instance configuration

`hbf_mem` is a sibling of `npu_mem`:

```json
{
  "npu_mem": {
    "mem_size": 180,
    "mem_bw": 8000,
    "mem_latency": 0
  },
  "hbf_mem": {
    "schema_version": 1,
    "num_stacks": 8,
    "stack_capacity_gb": 512,
    "performance": {
      "source": "scale",
      "latency_scale": 1.0
    }
  },
  "placement": {
    "default": {
      "weights": "hbf",
      "kv_loc": "npu",
      "kv_evict_loc": "cpu"
    }
  }
}
```

Capacity uses the simulator's existing binary convention:
`1 GB = 2^30 bytes`. Total HBF capacity is derived from
`num_stacks * stack_capacity_gb`; it is never hard-coded in the
allocator. Both stack fields and the scale must be positive.

Placement precedence remains `default -> block -> layer`, with later
rules overriding earlier rules. HBM and HBF residency is calculated
per rank after TP, PP, and EP sharding. Startup fails before scheduling
when either tier is too small, reporting the instance, tier, required
bytes, and available bytes. Moving weights to HBF releases the
corresponding HBM capacity for KV cache.

When `hbf_mem` is absent, no HBF object or performance source is
constructed and the original path is unchanged.

## Coefficient source

The scale source applies only when an operator has nonzero static
weight bytes and its logical weight placement is HBF:

```text
HBF latency = round(baseline hardware latency * latency_scale)
```

The baseline must come from the same hardware, model, dtype variant,
and TP bundle. Attention kernels (zero static weight), KV operations,
collectives, PIM attention, and HBM/CXL-resident layers are unchanged.
`latency_scale=1.0` is therefore the paired identity case.

A single run may override all scale-based HBF instances:

```bash
python -m serving \
  --cluster-config configs/cluster/hbf_b200_llama405b_tp8.json \
  --dataset workloads/hbf-saturated-1024-128-300.jsonl \
  --no-enable-prefix-caching \
  --hbf-latency-scale 1.5 \
  --output outputs/hbf-k1p5.csv \
  --hbf-summary-output outputs/hbf-k1p5-runtime.json
```

The checked-in target templates intentionally do not contain invented
B200 or RTX PRO 6000 target-model measurements. A run fails at the
existing Profile Bundle gate until the matching baseline bundle is
installed.

## External simulator Profile Bundle

A reliable external simulator uses the mutually exclusive `profile`
source:

```json
{
  "performance": {
    "source": "profile",
    "profile_root": "/absolute/path/to/hbf/perf",
    "profile_hardware": "B200_HBF"
  }
}
```

The directory and lookup keys are identical to a normal profiler
bundle:

```text
<profile_root>/<profile_hardware>/<model>/<variant>/tp<N>/
  dense.csv
  per_sequence.csv
  attention.csv
  moe.csv
  skew.csv
  skew_fit.csv
```

Only files required by the model architecture need data, but every
queried HBF weight operator must be present. `meta.yaml` must identify
the exact hardware, model, and variant and include:

```yaml
hardware: B200_HBF
model: meta-llama/Llama-3.1-405B-Instruct
variant: bf16
hbf_profile:
  schema_version: 1
  producer: reliable-simulator-name-and-version
  source: cycle-level HBF simulation
```

Missing metadata, model/variant mismatches, absent TP directories, or
missing operator rows fail directly. The implementation never mixes an
external bundle with baseline values or a coefficient fallback.

## Paper-core sweep

The independent-process sweep runner defaults to:

```text
1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.5, 10.0
```

```bash
python scripts/hbf_sweep.py \
  --cluster-config configs/cluster/hbf_rtxpro6000_llama405b_tp8.json \
  --dataset workloads/hbf-saturated-2048-512-300.jsonl \
  --output-dir outputs/hbf-paper-core \
  --num-reqs 300 \
  -- --no-enable-prefix-caching
```

Use `--scales 1 1.5 6.5 13.3` for a custom stress scan. Each point
starts a fresh `python -m serving` process and gets a distinct run ID
and directory.

Each point contains:

- `requests.csv`: original per-request simulator output;
- `simulator.log`: complete process output;
- `runtime.json`: per-rank HBM/HBF weight residency, remaining HBM KV
  capacity, timing source, device count, and energy caveat;
- `manifest.json`: Git SHA, command, config/workload SHA-256, profile
  tree hashes, scale, evidence label, and failure state.

The root `summary.csv` recomputes TTFT and TPOT mean/p50/p90/p99 in
milliseconds, prompt/generation/total token throughput, and all three
per-device throughput values from `requests.csv`. A device is a
participating GPU, not an HBM or HBF stack.

Scale results are labeled `sensitivity-analysis`. A successful profile
run is labeled `external-simulator-backed` only after its provenance
gate passes. Neither label means measured HBF silicon.

## Why these coefficients?

[TileLens (arXiv:2607.04031)](https://arxiv.org/abs/2607.04031)
reports conventional-layout geometric-mean slowdowns of
`1.61x-6.49x`, individual severe points above `8x-10x`, and performance
within about 1% of HBM after tile-major layout and adaptive prefetching.

[FlashAccel (arXiv:2607.10186)](https://arxiv.org/abs/2607.10186)
reports latency within about 4% of HBM with all optimizations, while
plain HBF loses about 65% throughput. The throughput result motivates
including degraded scenarios but is **not** converted directly into an
operator-latency coefficient.

The coefficients are sensitivity assumptions, not a claim that every
kernel experiences the same physical slowdown. Add isolated extreme
points such as `13.3x` explicitly instead of mixing them into the
default paper-core result.
