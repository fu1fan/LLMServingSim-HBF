---
title: HBF parallel and dual-routing matrix
---

# HBF parallel and dual-routing matrix

The versioned experiment manifest
`configs/experiments/hbf_parallel_modes_v1.json` evaluates only
static model weights in HBF, using parallel modes already implemented
by LLMServingSim:

- Llama 3.1 405B: TP, PP, independent replicas, and their supported
  combinations.
- Qwen3-235B-A22B: TP/EP, PP, independent replicas, and cross-instance
  DP-group + EP.
- No ETP, new collective, HBF KV path, dynamic expert migration, or
  ASTRA-Sim memory backend is introduced.

Qwen runs BALANCED and CUSTOM as co-equal primary cases. BALANCED is
the ideal global-load-balance boundary. CUSTOM is a
public-statistics-calibrated synthetic sensitivity case. Their
difference must not be presented as a measured production delta.

## Generate and run

First inspect the exact plan:

```bash
python scripts/hbf_parallel_matrix.py \
  --phase stage1 \
  --output-dir results/hbf-parallel-v1 \
  --dry-run
```

Then omit `--dry-run` to execute. The runner creates one simulator
process per point and can resume completed run directories.

Phases are:

| Phase | Contract |
| --- | --- |
| `stage1` | 219 HBF k=1 runs plus their 219 HBM capacity/performance pairs |
| `anchors` | 588 non-identity-scale runs; k=1 is reused from stage1 |
| `routing` | Five CUSTOM expert-ID mapping seeds and a small RAND control |
| `network` | Optimistic, central, and pessimistic estimated network cases |

The HBF scales are `1, 1.25, 1.5, 2, 3, 4, 6.5, 10`. Scale results
are sensitivity analysis, not measured HBF silicon.

## Validate independently

After stage1:

```bash
python scripts/validate_hbf_parallel_results.py \
  --output-dir results/hbf-parallel-v1
```

The validator rereads request CSV files, recomputes all latency and
throughput metrics, weights memory totals by each instance's device
count, parses per-layer EP load logs, checks HBM/HBF k=1 identity,
and writes:

- `summary.csv`
- `failures.csv`
- `comparisons.csv`
- `winners.json`

`winners.json` implements the fixed 4/8/16-device selection rule and
can drive the automatic non-anchor scan:

```bash
python scripts/hbf_parallel_matrix.py \
  --phase winners \
  --selection results/hbf-parallel-v1/winners.json \
  --output-dir results/hbf-parallel-v1
```

## Evidence boundaries

The bundled Qwen profile retains only the ranked aggregate skew shape
of the public count vector. The source does not establish transferable
expert IDs, a fully specified model revision, per-layer workload
composition, or production token traces. Accordingly, results carry
`community-routing-statistics` and
`public-stat-calibrated-synthetic-routing`, never
`external-trace-backed-routing`.

The imported B200 bundle backs operator lookup. HBF timing still uses
coefficient sensitivity unless a complete external HBF profile is
provided. Network bandwidth is specification-informed, while latency
and effective-bandwidth reductions are explicitly estimated. HBF
energy remains unmodeled.

Sources:

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Qwen3 MoE Router Statistics](https://github.com/sionic-ai/qwen3-moe-analyzer)
- [NVIDIA DGX B200 User Guide](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html)
