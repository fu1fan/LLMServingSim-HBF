# LLMServingSim (HBF fork)

This is an experimental fork of
[`casys-kaist/LLMServingSim`](https://github.com/casys-kaist/LLMServingSim) hosting
the **HBF (Host-Based Flash / CXL-style host-attached flash) weight-residency**
extensions used by the HBFServingSim research repo.

> Upstream's own README (LLMServingSim website, Docker workflow, etc.) does **not**
> apply here. This fork is consumed directly as a submodule and is prepared with a
> plain Python environment, not the upstream Docker container.

## What this fork adds relative to upstream `main`

Baseline upstream: `casys-kaist/LLMServingSim@main` (`c84e58b`). `hbf-main` is **45
commits ahead**.

- **HBF weight residency (parameter-driven)** — `serving/core/hbf_model.py`,
  `hbf_performance.py`, `hbf_summary.py`, `memory_model.py`, `config_builder.py`,
  `trace_generator.py`. Model static weights can be placed in HBF (`"weights": "hbf"`)
  at a per-operator latency scale `k`, freeing HBM for KV cache.
- **KV generalization** — `memory_model.py`, `scheduler.py`, `config_builder.py`.
  KV eviction target (`kv_evict_loc`) may be HBF with correct per-rank vs
  full-cluster byte accounting; `--prefix-storage HBF` for per-GPU prefix caching.
- **KV backpressure fix (generic, upstream-portable)** — `serving/core/radix_tree.py`
  bounds prefix-cache inserts so a locked working set degrades gracefully instead of
  crashing; `trace_generator.py` emits integer `kv_load`/`kv_evict` sizes.
- **Tooling & data** — `scripts/hbf_parallel_matrix.py`, `hbf_sweep.py`;
  `configs/cluster/hbf_*`; `configs/experiments/hbf_parallel_modes_v1.json`;
  `profiler/perf/B200/`; `tests/test_hbf_*.py`.

Backend submodules (each rebased onto latest upstream, +1 commit) also carry HBF
support: `astra-sim` models HBF as a first-class memory tier, `chakra` adds
`--weight-prefetch-depth` + `HBF_MEMORY`, and `analytical` adds the `HBF_MEMORY`
location type.

> **Removed features**: the experimental HBF `bandwidth` source
> (`BandwidthHBFPerformanceSource`), MoE hot/cold expert residency
> (`moe_hot_expert_frac`), and the KV spill-vs-recompute cost model
> (`serving/core/kv_policy.py`) were dropped and are **not** part of this fork's
> current difference from upstream.

## Minimum setup (any Python 3.11 environment)

The repo does not mandate a specific environment manager (mise / conda / venv).

```bash
# 1. Python deps (matplotlib is only needed for `plot`)
pip install pyyaml pytest numpy pandas pyinstrument rich msgspec protobuf==7.35.1
pip install matplotlib   # only for plot

# 2. Submodules (recursive — covers this submodule's own nested submodules)
git submodule update --init --recursive

# 3. Compile ASTRA-Sim (first use)
cd astra-sim && bash build/astra_analytical/build.sh

# 4. Make the repo's chakra converter importable (not the pip-installed copy)
export PYTHONPATH="$PWD/astra-sim/extern/graph_frontend"
```

`serving` shells out to `python -m chakra.src.converter.converter`; the `graph_frontend`
parent dir (containing `chakra/`) must be on `PYTHONPATH` so the repo's converter (with
`--weight-prefetch-depth` and the metadata-path fix) is used rather than a pip copy.

## Run a small simulation

```bash
bash serving/run.sh   # single-instance example, or:
python -m serving \
  --cluster-config configs/cluster/single_node_single_instance.json \
  --dtype float16 --block-size 16 \
  --dataset workloads/example_trace.jsonl --output outputs/example_single_run.csv \
  --num-req 10
```

HBF configuration lives in `configs/cluster/hbf_*.json`; see
`docs/docs/examples/memory-tiers/hbf-static-weights.md`.

## Tests

```bash
python -m pytest tests/ -q
```

## Regenerate Chakra proto bindings (only if needed)

Not required to run experiments. The vendored `protobuf==7.35.1` matches the checked-in
gencode; do not install 6.x.
