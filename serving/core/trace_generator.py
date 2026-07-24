import os
import re
from .request import *
from .utils import *
import pandas as pd
import yaml
from .memory_model import calculate_sizes
from .gate_function import GateRouter
from .config_builder import get_device
from .power_model import PowerModel, total_ring_data
from .pim_model import PIMModel
from .logger import get_logger
from .run_paths import input_path
from .profile_contract import (
    PROFILE_V2_PARTIAL,
    PROFILE_V2_RUNTIME_READY,
    ProfileV2RuntimeNotReadyError,
    load_profile_contract,
    validate_memory_profile_id,
)
from .memory_scenario import (
    MemoryScenarioPolicy,
    parse_instance_performance_profile,
)
import bisect
from dataclasses import dataclass, field

# ----------------------------------------------------------------------
# Profiler 类别性能库的进程内缓存。
# v1 键：(hardware, model, variant)
# v2 键：(hardware, model, variant, memory_profile_id)
# 值：包含 meta、architecture、catalog、sequence 和 tables 的字典。
# ----------------------------------------------------------------------
_perf_db_cache = {}

logger = get_logger("TraceGenerator")


# ----------------------------------------------------------------------
# Profile-data paths + variant resolution (mirrors the profiler).
# ----------------------------------------------------------------------

_PROFILER_ROOT_REL = "../profiler"

_DTYPE_SHORT = {
    "bfloat16": "bf16", "bf16": "bf16",
    "float16": "fp16", "half": "fp16", "fp16": "fp16",
    "float32": "fp32", "float": "fp32", "fp32": "fp32",
    "fp8": "fp8", "fp8_e4m3": "fp8",
    "int8": "int8", "int4": "int4",
}

# TP collective hooks keyed on canonical layer name. Applied after the
# named layer when tp_size > 1. Names must match the profiler's catalog.
_TP_ALLREDUCE_AFTER = frozenset({"o_proj", "down_proj"})

_FOUR_WAY_AUDIT_COLUMNS = (
    "hbm_read_bytes",
    "hbm_write_bytes",
    "hbf_read_bytes",
    "hbf_write_bytes",
)


def _short_dtype(d):
    if d is None:
        return None
    return _DTYPE_SHORT.get(str(d), str(d))


def resolve_variant(dtype, kv_cache_dtype, model_config=None):
    """Compute the profiler's variant folder name from runtime dtype
    choices. Matches ``ProfileArgs.effective_variant`` in the profiler.
    """
    weight = dtype
    if not weight and model_config is not None:
        weight = model_config.get("torch_dtype")
    parts = [_short_dtype(weight) if weight else "default"]
    if kv_cache_dtype and kv_cache_dtype != "auto":
        parts.append(f"kv{_short_dtype(kv_cache_dtype)}")
    return "-".join(parts)


def _arch_yaml_path(model_type):
    base = os.path.dirname(os.path.abspath(__file__))
    serving_dir = os.path.dirname(base)
    repo_root = os.path.dirname(serving_dir)
    candidate_paths = [
        os.path.join(repo_root, "profiler", "models", f"{model_type}.yaml"),
        os.path.join(serving_dir, "profiler", "models", f"{model_type}.yaml"),
    ]
    for path in candidate_paths:
        if os.path.isfile(path):
            return path
    return candidate_paths[0]


def _variant_root(hardware, model, variant):
    return f"{_PROFILER_ROOT_REL}/perf/{hardware}/{model}/{variant}"


def _profile_root(hardware, model, variant, memory_profile_id=None):
    """保持 v1 路径不变；v2 在 variant 下增加稳定身份目录。"""
    root = _variant_root(hardware, model, variant)
    if memory_profile_id is None:
        return root
    profile_id = validate_memory_profile_id(
        memory_profile_id,
        field="memory_profile_id",
    )
    return os.path.join(root, profile_id)


def _perf_db_cache_key(hardware, model, variant, memory_profile_id=None):
    """v1 沿用三元键，v2 用第四维隔离不同硬件内存配置。"""
    if memory_profile_id is None:
        return (hardware, model, variant)
    profile_id = validate_memory_profile_id(
        memory_profile_id,
        field="memory_profile_id",
    )
    return (hardware, model, variant, profile_id)


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class TraceCtx:
    """Immutable context for an entire trace generation."""
    hardware: str
    model: str
    config: dict
    perf_db: dict
    node_id: int
    fp: int
    placement: dict
    gate: object  # GateRouter or None
    enable_attn_offloading: bool
    power_model: object  # PowerModel or None
    pim_model: object  # PIMModel or None
    pim_channels: int
    n_head: int
    kv_head: int
    head_dim: int
    is_moe: bool
    pd_type: str  # 'prefill', 'decode', or None
    tp_size: int       # tensor parallel degree (for ALLREDUCE on attention/FFN)
    pp_size: int       # pipeline parallel degree
    local_ep: int      # expert parallel degree within this instance
    ep_total: int      # total EP degree across DP group
    tp_dim: list       # involved_dim for TP collectives (ALLREDUCE), None = all dims
    ep_dim: list       # involved_dim for EP collectives (ALLTOALL), None = all dims
    dp_sum_total_len: int  # sum of total_len across DP group (0 = DP inactive). Captures the post-AG gathered size for MoE compute; dummy batches are pre-padded to max by serving/__main__.py so the sum reflects vLLM's CUDA-graph padding.
    memory_scenario_policy: object = None


@dataclass
class BatchCtx:
    """Per-batch state computed from a Batch object."""
    batch: object  # Batch
    total_len: int
    prefill_chunk: int  # sum(prefill_q_list): new prefill tokens this step
    kv_prefill: int     # sum(prefill_k_list): existing kv history for prefill reqs
    n_decode: int       # number of decode requests
    kv_decode_mean: int # mean decode kv length (4D grid carries one value)
    kv_decode_max: int  # max decode kv length (for skew correction)
    kv_decode_min: int  # min decode kv length (for skew_rate in skew correction)
    lm_head_len: int    # number of sequences
    decode_lens: list   # per-PIM-channel decode lengths (None if no PIM)
    channel_split: int  # PIM channel split factor


@dataclass
class PowerAccumulator:
    """Accumulates power data for a block, then flushes to power_model."""
    npu_latencies_ns: list
    pim_latencies_ns: list
    dram_weight_bytes: int
    link_data_bytes: int

    def flush(self, ctx, enable_attn_offloading=False):
        if ctx.power_model is None:
            return
        ctx.power_model.add_dram_energy_consumption(ctx.node_id, self.dram_weight_bytes)
        ctx.power_model.add_link_energy_consumption(ctx.node_id, self.link_data_bytes)
        for lat in self.npu_latencies_ns:
            ctx.power_model.add_npu_active_energy_consumption(ctx.hardware, ctx.node_id, lat, num_npus=ctx.tp_size)
        if enable_attn_offloading:
            for lat in self.pim_latencies_ns:
                ctx.power_model.add_pim_active_energy_consumption(ctx.node_id, lat)


@dataclass(frozen=True)
class ProfileLatencySample:
    """保留一次 Profile 查询的延迟和四向物理流量。"""

    latency_ns: int
    hbm_read_bytes: int = 0
    hbm_write_bytes: int = 0
    hbf_read_bytes: int = 0
    hbf_write_bytes: int = 0

    @property
    def four_way_bytes(self):
        return {
            column: getattr(self, column)
            for column in _FOUR_WAY_AUDIT_COLUMNS
        }


# ======================================================================
# Perf DB loading and lookup (new per-category format)
# ======================================================================
#
# New layout under profiler/perf/<hw>/<model>/<variant>/:
#     meta.yaml                       profiler settings, effective engine kwargs
#     tp<N>/dense.csv                 layer, tokens, time_us
#     tp<N>/per_sequence.csv          layer, sequences, time_us
#     tp<N>/attention.csv             prefill_chunk, kv_prefill, n_decode, kv_decode, time_us
#     tp<N>/moe.csv                   tokens, activated_experts, time_us    (MoE only)
#
# Architecture structure (catalog + sequence) lives in the profiler's
# profiler/models/<model_type>.yaml and drives which canonical
# layers the simulator emits.


def _load_architecture(model_type):
    """Load catalog + sequence from profiler/models/<model_type>.yaml."""
    path = _arch_yaml_path(model_type)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Architecture yaml not found for model_type={model_type!r} at {path}. "
            f"Add profiler/models/{model_type}.yaml describing the architecture."
        )
    with open(path, "r") as f:
        arch = yaml.safe_load(f)
    if "catalog" not in arch or "sequence" not in arch:
        raise KeyError(
            f"Architecture yaml {path} must define both 'catalog' and 'sequence'."
        )
    return arch


def _load_meta(variant_root):
    path = os.path.join(variant_root, "meta.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"meta.yaml missing at {path}. Re-run the profiler to produce it."
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _hydrate_skew_fit_tables(meta, variant_root):
    """Load each TP's per-bucket alpha table from CSV into the meta dict.

    Newer profile runs move the (1k+ rows per TP) ``alpha_by_bucket``
    mapping out of meta.yaml into ``tp{N}/skew_fit.csv``. This helper
    reads those CSVs and materialises the dict in-place so
    ``_skew_alpha`` finds it where it used to be. Older meta.yamls
    that still inline the dict are left untouched.
    """
    fit = (meta or {}).get("skew_fit") if isinstance(meta, dict) else None
    if not fit or not fit.get("enabled"):
        return
    per_tp = fit.get("per_tp")
    if not isinstance(per_tp, dict):
        return
    for tp_key, entry in per_tp.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("alpha_by_bucket"):
            continue
        rel = entry.get("bucket_table")
        if not rel:
            continue
        csv_path = os.path.join(variant_root, rel)
        if not os.path.isfile(csv_path):
            logger.warning(
                "skew_fit: tp=%s bucket_table %s missing — falling back to "
                "alpha_default", tp_key, csv_path,
            )
            continue
        alphas, counts = _read_skew_fit_csv(csv_path)
        entry["alpha_by_bucket"] = alphas
        entry["n_by_bucket"] = counts


def _read_skew_fit_csv(path):
    """Return (alpha_by_bucket, n_by_bucket) keyed by the pipe-delimited
    bucket string used by ``_skew_alpha``.
    """
    df = pd.read_csv(path)
    alphas: dict = {}
    counts: dict = {}
    for row in df.itertuples(index=False):
        raw = getattr(row, "raw_key", None)
        if isinstance(raw, str) and raw:
            key = raw
        else:
            key = (
                f"pc={int(row.pc)}|{row.n_label}|{row.skew_rate_label}"
                f"|{row.kv_big_label}|{row.kp_label}"
            )
        alphas[key] = float(row.alpha)
        if hasattr(row, "n_samples"):
            try:
                counts[key] = int(row.n_samples)
            except (TypeError, ValueError):
                pass
    return alphas, counts


def _read_category_csv(path, key_cols):
    """Read a category CSV and return a dict per layer (for dense/per_sequence)
    or a list of rows (for attention/moe).

    key_cols: for dense/per_sequence this is ["tokens"] or ["sequences"];
              for attention/moe pass None to return raw rows.
    """
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, sep=",")
    # time_us -> latency_ns (int, min 1)
    df["latency_ns"] = (df["time_us"].astype(float) * 1_000.0).round().astype(int).clip(lower=1)
    return df


def _audit_values(df):
    """按 CSV 行序保留四向物理字节；v1 表返回空 mapping。"""
    if not all(column in df.columns for column in _FOUR_WAY_AUDIT_COLUMNS):
        return {}
    return {
        column: df[column].astype(int).tolist()
        for column in _FOUR_WAY_AUDIT_COLUMNS
    }


def _build_1d_table(df, layer_col, key_col):
    """Dense / per-sequence: per-layer sorted (keys, values) table."""
    out = {}
    for layer, g in df.groupby(layer_col):
        g = g.sort_values(key_col).drop_duplicates(subset=[key_col])
        out[str(layer)] = {
            "keys": g[key_col].astype(int).tolist(),
            "values": g["latency_ns"].astype(int).tolist(),
            "audit_values": _audit_values(g),
        }
    return out


def _build_attention_table(df):
    """4D attention table indexed by (prefill_chunk, n_decode) slices,
    each slice a 2D grid over (kv_prefill, kv_decode). The profiler
    sweeps all four axes on doubling grids, so the lookup interpolates
    in log-space on each axis (plus a zero-pinned fallback when the
    axis value is 0, which always comes from an exact sample).
    """
    pc_vals = sorted({int(v) for v in df["prefill_chunk"].tolist()})
    nd_vals = sorted({int(v) for v in df["n_decode"].tolist()})
    slices = {}
    for (pc, nd), g in df.groupby(["prefill_chunk", "n_decode"]):
        slice_tbl = {}
        for kp, g2 in g.groupby("kv_prefill"):
            g2 = g2.sort_values("kv_decode").drop_duplicates(subset=["kv_decode"])
            slice_tbl[int(kp)] = {
                "keys": g2["kv_decode"].astype(int).tolist(),
                "values": g2["latency_ns"].astype(int).tolist(),
                "audit_values": _audit_values(g2),
            }
        kp_vals_s = sorted(slice_tbl.keys())
        slices[(int(pc), int(nd))] = {
            "kv_prefill_vals": kp_vals_s,
            "rows": [slice_tbl[kp] for kp in kp_vals_s],
        }
    return {
        "pc_vals": pc_vals, "nd_vals": nd_vals,
        "pc_nd_pairs": sorted(slices.keys()),
        "slices": slices,
    }


def _build_moe_table(df):
    """MoE table: (tokens, activated_experts) → latency_ns."""
    tokens_by_experts = {}
    for ae, g in df.groupby("activated_experts"):
        g = g.sort_values("tokens").drop_duplicates(subset=["tokens"])
        tokens_by_experts[int(ae)] = {
            "keys": g["tokens"].astype(int).tolist(),
            "values": g["latency_ns"].astype(int).tolist(),
            "audit_values": _audit_values(g),
        }
    ae_vals = sorted(tokens_by_experts.keys())
    return {"activated_experts_vals": ae_vals,
            "rows": [tokens_by_experts[a] for a in ae_vals]}


def _build_scenario_1d_table(df, layer_col, key_col):
    """构建 layer -> scenario -> 1D table，scenario 不参与插值。"""
    df = df.copy()
    df["memory_scenario"] = df["memory_scenario"].astype(str).str.strip()
    df[layer_col] = df[layer_col].astype(str).str.strip()
    out = {}
    for scenario, scenario_rows in df.groupby("memory_scenario"):
        by_layer = _build_1d_table(scenario_rows, layer_col, key_col)
        for layer, table in by_layer.items():
            out.setdefault(layer, {})[str(scenario)] = table
    return out


def _build_scenario_attention_table(df):
    """构建 scenario -> 4D attention table。"""
    df = df.copy()
    df["memory_scenario"] = df["memory_scenario"].astype(str).str.strip()
    return {
        str(scenario): _build_attention_table(scenario_rows)
        for scenario, scenario_rows in df.groupby("memory_scenario")
    }


def _build_scenario_moe_table(df):
    """构建 scenario -> 2D MoE table。"""
    df = df.copy()
    df["memory_scenario"] = df["memory_scenario"].astype(str).str.strip()
    return {
        str(scenario): _build_moe_table(scenario_rows)
        for scenario, scenario_rows in df.groupby("memory_scenario")
    }


def _v2_skew_enabled(meta):
    skew_profile = (meta or {}).get("skew_profile") or {}
    skew_fit = (meta or {}).get("skew_fit") or {}
    return bool(skew_profile.get("enabled") or skew_fit.get("enabled"))


def _runtime_uses_moe(model_config, architecture):
    """按运行模型选择实际会执行的 MLP 分支。"""

    sequence = architecture.get("sequence") or {}
    if model_config is not None:
        experts = model_config.get(
            "num_local_experts",
            model_config.get("num_experts"),
        )
        return bool(experts)

    dense_layers = list(sequence.get("mlp_dense") or [])
    moe_layers = list(sequence.get("mlp_moe") or [])
    if dense_layers and moe_layers:
        raise ProfileV2RuntimeNotReadyError(
            "Profile v2 运行时校验需要 model_config 才能选择 MLP 分支"
        )
    return bool(moe_layers)


def _active_architecture_layers(architecture, model_config):
    """从运行时 sequence 派生真正会发起 Profile 查询的层。"""

    sequence = architecture.get("sequence")
    catalog = architecture.get("catalog")
    if not isinstance(sequence, dict) or not isinstance(catalog, dict):
        raise ProfileV2RuntimeNotReadyError(
            "architecture 必须包含 mapping 类型的 sequence 与 catalog"
        )
    categories = ("dense", "per_sequence", "attention", "moe")
    catalog_sections = {}
    for category in categories:
        section = catalog.get(category) or {}
        if not isinstance(section, dict):
            raise ProfileV2RuntimeNotReadyError(
                f"architecture catalog.{category} 必须是 mapping"
            )
        catalog_sections[category] = section

    is_moe = _runtime_uses_moe(model_config, architecture)
    dense_mlp = list(sequence.get("mlp_dense") or [])
    moe_mlp = list(sequence.get("mlp_moe") or [])
    if is_moe and not moe_mlp and dense_mlp:
        raise ProfileV2RuntimeNotReadyError(
            "模型配置选择 MoE，但 architecture 没有 mlp_moe sequence"
        )
    if not is_moe and not dense_mlp and moe_mlp:
        raise ProfileV2RuntimeNotReadyError(
            "模型配置选择 Dense，但 architecture 没有 mlp_dense sequence"
        )
    active_sections = (
        "prologue",
        "pre_attn",
        "post_attn",
        "mlp_moe" if is_moe else "mlp_dense",
        "head",
    )
    active_by_category = {category: set() for category in categories}
    for section_name in active_sections:
        layers = sequence.get(section_name) or []
        if not isinstance(layers, list):
            raise ProfileV2RuntimeNotReadyError(
                f"architecture sequence.{section_name} 必须是列表"
            )
        for layer_name in layers:
            if not isinstance(layer_name, str) or not layer_name:
                raise ProfileV2RuntimeNotReadyError(
                    f"architecture sequence.{section_name} 包含非法层名"
                )
            matches = [
                category
                for category, entries in catalog_sections.items()
                if layer_name in entries
            ]
            if len(matches) != 1:
                raise ProfileV2RuntimeNotReadyError(
                    f"活动层 {layer_name!r} 必须恰好属于一个 catalog 类别，"
                    f"实际匹配 {matches}"
                )
            active_by_category[matches[0]].add(layer_name)
    return active_by_category


def _validate_v2_runtime_bundle(
    perf_db,
    *,
    model_type,
    model_config,
    runtime_max_num_batched_tokens=None,
    runtime_max_num_seqs=None,
    runtime_block_size=None,
):
    """独立核对 manifest 声明与实际架构、目录和查询路径。"""

    if perf_db.get("bundle_readiness") != PROFILE_V2_RUNTIME_READY:
        raise ProfileV2RuntimeNotReadyError(
            "Profile v2 bundle 尚未标记为 runtime_ready"
        )
    if perf_db.get("runtime_compatible") is not True:
        raise ProfileV2RuntimeNotReadyError(
            "Profile v2 bundle 未声明 runtime_compatible=true"
        )
    if perf_db.get("scenario_binding") != "producer_verified_v1":
        raise ProfileV2RuntimeNotReadyError(
            "Profile v2 bundle 未使用 producer_verified_v1 场景绑定"
        )
    _validate_v2_runtime_limits(
        perf_db,
        runtime_max_num_batched_tokens=runtime_max_num_batched_tokens,
        runtime_max_num_seqs=runtime_max_num_seqs,
        runtime_block_size=runtime_block_size,
    )
    requirements = perf_db.get("architecture_requirements")
    if not isinstance(requirements, dict):
        raise ProfileV2RuntimeNotReadyError(
            "runtime_ready bundle 缺少 architecture_requirements"
        )
    if requirements["model_type"] != model_type:
        raise ProfileV2RuntimeNotReadyError(
            "architecture_requirements.model_type 与运行模型不一致"
        )

    required_tps = set(requirements["tp_degrees"])
    available_tps = set(perf_db["available_tps"])
    if required_tps != available_tps:
        raise ProfileV2RuntimeNotReadyError(
            "architecture_requirements.tp_degrees 与实际 tp<N> 目录不一致"
        )
    required_scenarios = set(requirements["scenario_ids"])
    actual_scenarios = set(perf_db.get("scenario_catalog") or {})
    if required_scenarios != actual_scenarios:
        raise ProfileV2RuntimeNotReadyError(
            "architecture_requirements.scenario_ids 与 scenario_catalog 不一致"
        )

    active = _active_architecture_layers(
        perf_db["architecture"],
        model_config,
    )
    if active["attention"] not in (set(), {"attention"}):
        raise ProfileV2RuntimeNotReadyError(
            "活动 attention 层必须使用 canonical 名称 'attention'"
        )
    if active["moe"] not in (set(), {"moe"}):
        raise ProfileV2RuntimeNotReadyError(
            "活动 MoE 层必须使用 canonical 名称 'moe'"
        )
    if set(requirements["dense_layers"]) != active["dense"]:
        raise ProfileV2RuntimeNotReadyError(
            "architecture_requirements.dense_layers 与活动 sequence 不一致"
        )
    if set(requirements["per_sequence_layers"]) != active["per_sequence"]:
        raise ProfileV2RuntimeNotReadyError(
            "architecture_requirements.per_sequence_layers 与活动 sequence 不一致"
        )
    if requirements["attention_required"] is not bool(active["attention"]):
        raise ProfileV2RuntimeNotReadyError(
            "architecture_requirements.attention_required 与活动 sequence 不一致"
        )
    if requirements["moe_required"] is not bool(active["moe"]):
        raise ProfileV2RuntimeNotReadyError(
            "architecture_requirements.moe_required 与活动 sequence 不一致"
        )

    # 只证明查询路径有表；数值扫描范围仍由上层配置另行约束。
    for tp in sorted(required_tps):
        for scenario in sorted(required_scenarios):
            for category in ("dense", "per_sequence"):
                for layer_name in sorted(active[category]):
                    effective_tp = _effective_tp(
                        perf_db,
                        category,
                        layer_name,
                        tp,
                        scenario,
                    )
                    scenario_tables = (
                        _tp_tables(perf_db, effective_tp)
                        .get(category, {})
                        .get(layer_name)
                    )
                    if not scenario_tables or scenario not in scenario_tables:
                        raise ProfileV2RuntimeNotReadyError(
                            f"缺少 {category} 查询覆盖：layer={layer_name!r}, "
                            f"tp={effective_tp}, scenario={scenario!r}"
                        )

            if active["attention"]:
                attention_tables = (
                    _tp_tables(perf_db, tp).get("attention") or {}
                )
                if scenario not in attention_tables:
                    raise ProfileV2RuntimeNotReadyError(
                        f"缺少 attention 查询覆盖：tp={tp}, "
                        f"scenario={scenario!r}"
                    )

    if active["moe"]:
        moe_tp = (
            1
            if 1 in perf_db["available_tps"]
            else perf_db["available_tps"][0]
        )
        moe_tables = _tp_tables(perf_db, moe_tp).get("moe") or {}
        for scenario in sorted(required_scenarios):
            if scenario not in moe_tables:
                raise ProfileV2RuntimeNotReadyError(
                    f"缺少 moe 查询覆盖：tp={moe_tp}, "
                    f"scenario={scenario!r}"
                )


def _validate_v2_runtime_limits(
    perf_db,
    *,
    runtime_max_num_batched_tokens=None,
    runtime_max_num_seqs=None,
    runtime_block_size=None,
):
    """Profile v2 不允许运行配置越过生产者验证边界。"""

    engine = perf_db.get("engine_effective")
    if not isinstance(engine, dict):
        raise ProfileV2RuntimeNotReadyError(
            "runtime-ready Profile v2 缺少 engine_effective"
        )
    checks = (
        (
            "max_num_batched_tokens",
            runtime_max_num_batched_tokens,
        ),
        ("max_num_seqs", runtime_max_num_seqs),
    )
    for field, runtime_value in checks:
        if runtime_value is None:
            continue
        runtime_value = int(runtime_value)
        if runtime_value <= 0:
            raise ProfileV2RuntimeNotReadyError(
                f"运行时 {field} 必须是正整数"
            )
        if runtime_value > engine[field]:
            raise ProfileV2RuntimeNotReadyError(
                f"运行时 {field}={runtime_value} 超过 Profile "
                f"验证范围 {engine[field]}"
            )
    if runtime_block_size is not None:
        runtime_block_size = int(runtime_block_size)
        if runtime_block_size != engine["block_size"]:
            raise ProfileV2RuntimeNotReadyError(
                f"运行时 block_size={runtime_block_size} 与 Profile "
                f"block_size={engine['block_size']} 不一致"
            )


def _load_perf_db(hardware, model, variant, tp_needed, model_type,
                  memory_profile_id=None, model_config=None,
                  runtime_max_num_batched_tokens=None,
                  runtime_max_num_seqs=None, runtime_block_size=None):
    """Load the per-category perf DB for a (hardware, model, variant)
    tuple and cache it. ``tp_needed`` is a set of int TP degrees the
    simulator will query; each must have its own ``tp<N>/`` folder.
    """
    cache_key = _perf_db_cache_key(
        hardware,
        model,
        variant,
        memory_profile_id,
    )
    if cache_key in _perf_db_cache:
        db = _perf_db_cache[cache_key]
        _check_tp_coverage(db, tp_needed, hardware, model, variant)
        if _is_profile_v2(db):
            _validate_v2_runtime_bundle(
                db,
                model_type=model_type,
                model_config=model_config,
                runtime_max_num_batched_tokens=runtime_max_num_batched_tokens,
                runtime_max_num_seqs=runtime_max_num_seqs,
                runtime_block_size=runtime_block_size,
            )
        return db

    root = _profile_root(hardware, model, variant, memory_profile_id)
    if not os.path.isdir(root):
        if memory_profile_id is None:
            raise FileNotFoundError(
                f"Profile variant folder not found: {root}. Run the profiler "
                f"with matching --dtype / --kv-cache-dtype, or pick an existing "
                f"variant under {os.path.dirname(root)}."
            )
        raise FileNotFoundError(
            f"Profile v2 bundle folder not found: {root}. Run the exporter "
            "with a matching memory_profile_id."
        )

    contract = load_profile_contract(
        root,
        requested_memory_profile_id=memory_profile_id,
    )
    if contract.is_v2 and contract.bundle_readiness == PROFILE_V2_PARTIAL:
        raise ProfileV2RuntimeNotReadyError(
            f"Profile v2 bundle {root} 是 partial，仅可做静态审计，"
            "不能进入 Serving 运行时"
        )
    if contract.is_v2 and _v2_skew_enabled(contract.meta):
        # v2 skew 尚无稳定的场景化数据契约，不能复用 v1 的共享 alpha。
        raise ProfileV2RuntimeNotReadyError(
            f"Profile v2 bundle {root} 启用了 skew，但当前 skew 数据与 "
            "lookup 尚未按 memory_scenario 隔离；请禁用 skew"
        )

    meta = contract.meta
    if not contract.is_v2:
        _hydrate_skew_fit_tables(meta, root)
    arch = _load_architecture(model_type)
    tables_per_tp = {}
    available_tps = []
    for entry in sorted(os.listdir(root)):
        if not entry.startswith("tp"):
            continue
        try:
            tp = int(entry[2:])
        except ValueError:
            continue
        tp_dir = os.path.join(root, entry)
        tables = {}

        dense_df = _read_category_csv(os.path.join(tp_dir, "dense.csv"), None)
        if dense_df is not None:
            if contract.is_v2:
                tables["dense"] = _build_scenario_1d_table(
                    dense_df,
                    "layer",
                    "tokens",
                )
            else:
                tables["dense"] = _build_1d_table(
                    dense_df,
                    "layer",
                    "tokens",
                )

        per_seq_df = _read_category_csv(os.path.join(tp_dir, "per_sequence.csv"), None)
        if per_seq_df is not None:
            if contract.is_v2:
                tables["per_sequence"] = _build_scenario_1d_table(
                    per_seq_df,
                    "layer",
                    "sequences",
                )
            else:
                tables["per_sequence"] = _build_1d_table(
                    per_seq_df,
                    "layer",
                    "sequences",
                )

        attn_df = _read_category_csv(os.path.join(tp_dir, "attention.csv"), None)
        if attn_df is not None:
            if contract.is_v2:
                tables["attention"] = _build_scenario_attention_table(attn_df)
            else:
                tables["attention"] = _build_attention_table(attn_df)

        moe_df = _read_category_csv(os.path.join(tp_dir, "moe.csv"), None)
        if moe_df is not None:
            if contract.is_v2:
                tables["moe"] = _build_scenario_moe_table(moe_df)
            else:
                tables["moe"] = _build_moe_table(moe_df)

        tables_per_tp[tp] = tables
        available_tps.append(tp)

    perf_db = {
        "meta": meta,
        "architecture": arch,
        "variant": variant,
        "hardware": hardware,
        "model": model,
        "profile_schema_version": contract.schema_version,
        "memory_profile_id": contract.memory_profile_id,
        "scenario_catalog": contract.scenario_catalog,
        "bundle_readiness": contract.bundle_readiness,
        "runtime_compatible": contract.runtime_compatible,
        "scenario_binding": contract.scenario_binding,
        "architecture_requirements": contract.architecture_requirements,
        "access_catalog": contract.access_catalog,
        "calibration": contract.calibration,
        "performance_basis": contract.performance_basis,
        "memory_integration": contract.memory_integration,
        "engine_effective": contract.engine_effective,
        "attention_grid": contract.attention_grid,
        "performance_identity": contract.performance_identity,
        "available_tps": sorted(available_tps),
        "tables": tables_per_tp,
    }
    _check_tp_coverage(perf_db, tp_needed, hardware, model, variant)
    if contract.is_v2:
        _validate_v2_runtime_bundle(
            perf_db,
            model_type=model_type,
            model_config=model_config,
            runtime_max_num_batched_tokens=runtime_max_num_batched_tokens,
            runtime_max_num_seqs=runtime_max_num_seqs,
            runtime_block_size=runtime_block_size,
        )
    _perf_db_cache[cache_key] = perf_db
    return perf_db


def _check_tp_coverage(perf_db, tp_needed, hardware, model, variant):
    missing = sorted(set(tp_needed) - set(perf_db["available_tps"]))
    if missing:
        profile_suffix = ""
        if perf_db.get("memory_profile_id") is not None:
            profile_suffix = f"/{perf_db['memory_profile_id']}"
        raise FileNotFoundError(
            f"No profile data for tp={missing} under "
            f"perf/{hardware}/{model}/{variant}{profile_suffix}/. "
            f"Re-run the profiler with "
            f"TP_DEGREES including {','.join(str(t) for t in missing)}."
        )


def warn_if_runtime_exceeds_profiled(perf_db, runtime_max_num_batched_tokens,
                                     runtime_max_num_seqs):
    """Emit logger warnings when runtime batch limits exceed the values
    the profiler swept. Lookups will extrapolate, which is less accurate.
    Invoked once per (hw, model, variant) cache-hit.
    """
    meta = perf_db.get("meta", {})
    eff = (meta or {}).get("engine_effective") or {}
    p_tok = eff.get("max_num_batched_tokens")
    p_seqs = eff.get("max_num_seqs")
    key = ("warned", perf_db["hardware"], perf_db["model"], perf_db["variant"])
    if perf_db.get("memory_profile_id") is not None:
        key = (*key, perf_db["memory_profile_id"])
    if _perf_db_cache.get(key):
        return
    _perf_db_cache[key] = True
    if p_tok and runtime_max_num_batched_tokens and \
            runtime_max_num_batched_tokens > p_tok:
        logger.warning(
            "max-num-batched-tokens=%s exceeds profiled %s for %s/%s/%s; "
            "attention/dense lookups will extrapolate",
            runtime_max_num_batched_tokens, p_tok,
            perf_db["hardware"], perf_db["model"], perf_db["variant"],
        )
    if p_seqs and runtime_max_num_seqs and runtime_max_num_seqs > p_seqs:
        logger.warning(
            "max-num-seqs=%s exceeds profiled %s for %s/%s/%s; "
            "per-sequence lookups will extrapolate",
            runtime_max_num_seqs, p_seqs,
            perf_db["hardware"], perf_db["model"], perf_db["variant"],
        )


def _linear_interpolate(x0, y0, x1, y1, query):
    """Linear interpolation (or extrapolation)."""
    if x1 == x0:
        return y0
    t = (query - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _lookup_bounds(keys, query):
    """Binary search returning (lo_idx, hi_idx) bracket.

    If query is below min, returns (0, 0).
    If query is above max, returns (len-2, len-1) to allow extrapolation.
    Otherwise returns the bracketing pair.
    """
    idx = bisect.bisect_right(keys, query)
    if idx == 0:
        return 0, 0
    if idx >= len(keys):
        if len(keys) < 2:
            return 0, 0
        return len(keys) - 2, len(keys) - 1
    return idx - 1, idx


def _lookup_1d(keys, values, query):
    """1D interpolation on sorted (keys, values)."""
    if not keys:
        return 0
    if len(keys) == 1:
        return values[0]

    lo, hi = _lookup_bounds(keys, query)
    if lo == hi:
        # Clamped or exact
        return values[lo]
    return _linear_interpolate(keys[lo], values[lo], keys[hi], values[hi], query)


def _sample_from_components(latency_ns, audit_values=None):
    audit_values = audit_values or {}
    return ProfileLatencySample(
        latency_ns=max(1, int(latency_ns)),
        **{
            column: max(0, int(audit_values.get(column, 0)))
            for column in _FOUR_WAY_AUDIT_COLUMNS
        },
    )


def _lookup_1d_sample(table, query):
    latency_ns = _lookup_1d(
        table["keys"],
        table["values"],
        query,
    )
    audit_values = {
        column: _lookup_1d(
            table["keys"],
            values,
            query,
        )
        for column, values in (table.get("audit_values") or {}).items()
    }
    return _sample_from_components(latency_ns, audit_values)


def _tp_tables(perf_db, tp):
    """Fetch the category-table dict for a given TP degree, with a
    clear error if the TP wasn't profiled.
    """
    tables = perf_db["tables"].get(tp)
    if tables is None:
        raise KeyError(
            f"No profile data for tp={tp} on {perf_db['hardware']}/"
            f"{perf_db['model']}/{perf_db['variant']}; available: "
            f"{perf_db['available_tps']}"
        )
    return tables


def _tp_stable(perf_db, category, name):
    """Return True if the catalog marks this layer as TP-stable — i.e.
    the same kernel cost at any TP and profiled once at tp=1.
    """
    section = perf_db["architecture"]["catalog"].get(category) or {}
    entry = section.get(name)
    if not entry:
        return False
    return bool(entry.get("tp_stable"))


_LEGACY_MEMORY_SCENARIO = object()


def _effective_tp(
    perf_db,
    category,
    name,
    tp,
    memory_scenario=_LEGACY_MEMORY_SCENARIO,
):
    """Layers marked ``tp_stable`` in the architecture yaml are profiled
    once at tp=1 and the writer replicates them across TP folders, so
    either lookup works. Using the current TP keeps things uniform.
    """
    if _tp_stable(perf_db, category, name) and 1 in perf_db["available_tps"]:
        if _is_profile_v2(perf_db):
            scenario = _resolve_memory_scenario(perf_db, memory_scenario)
            tables = _tp_tables(perf_db, 1).get(category)
            if category in ("dense", "per_sequence"):
                tables = (tables or {}).get(name)
            if not tables or scenario not in tables:
                return tp
        return 1
    return tp


def _is_profile_v2(perf_db):
    version = perf_db.get("profile_schema_version")
    if version is None:
        version = (perf_db.get("meta") or {}).get("profile_schema_version", 1)
    return version == 2


def _resolve_memory_scenario(perf_db, memory_scenario):
    """v2 必须显式选择 catalog 场景；v1 保持无场景调用。"""
    if not _is_profile_v2(perf_db):
        if memory_scenario in (_LEGACY_MEMORY_SCENARIO, None):
            return _LEGACY_MEMORY_SCENARIO
        raise KeyError("Profile v1 不支持 memory_scenario")

    if memory_scenario in (_LEGACY_MEMORY_SCENARIO, None):
        raise KeyError("Profile v2 lookup 必须显式提供 memory_scenario")
    if not isinstance(memory_scenario, str) or not memory_scenario:
        raise KeyError("memory_scenario 必须是非空字符串")

    catalog = perf_db.get("scenario_catalog")
    if catalog is None:
        catalog = (perf_db.get("meta") or {}).get("scenario_catalog") or {}
    if memory_scenario not in catalog:
        raise KeyError(
            f"未知 memory_scenario={memory_scenario!r}；"
            f"可用场景：{sorted(catalog)}"
        )
    return memory_scenario


def _select_scenario_table(
    perf_db,
    scenario_tables,
    memory_scenario,
    *,
    category,
    layer_name=None,
):
    """从 v2 离散场景表中取值；v1 直接返回原表。"""
    scenario = _resolve_memory_scenario(perf_db, memory_scenario)
    if scenario is _LEGACY_MEMORY_SCENARIO:
        return scenario_tables
    table = scenario_tables.get(scenario)
    if table is None:
        layer = f", layer={layer_name}" if layer_name is not None else ""
        raise KeyError(
            f"Missing {category} profile for memory_scenario={scenario!r}"
            f"{layer}."
        )
    return table


def _lookup_dense(
    perf_db,
    name,
    tp,
    tokens,
    memory_scenario=_LEGACY_MEMORY_SCENARIO,
):
    return _lookup_dense_sample(
        perf_db,
        name,
        tp,
        tokens,
        memory_scenario=memory_scenario,
    ).latency_ns


def _lookup_dense_sample(
    perf_db,
    name,
    tp,
    tokens,
    memory_scenario=_LEGACY_MEMORY_SCENARIO,
):
    tp_eff = _effective_tp(
        perf_db,
        "dense",
        name,
        tp,
        memory_scenario,
    )
    scenario_tables = _tp_tables(perf_db, tp_eff).get("dense", {}).get(name)
    if scenario_tables is None:
        raise KeyError(
            f"Missing dense profile for layer={name} on tp={tp_eff}. "
            f"Check that the architecture catalog and dense.csv agree."
        )
    tbl = _select_scenario_table(
        perf_db,
        scenario_tables,
        memory_scenario,
        category="dense",
        layer_name=name,
    )
    return _lookup_1d_sample(tbl, max(int(tokens), 1))


def _lookup_per_sequence(
    perf_db,
    name,
    tp,
    sequences,
    memory_scenario=_LEGACY_MEMORY_SCENARIO,
):
    return _lookup_per_sequence_sample(
        perf_db,
        name,
        tp,
        sequences,
        memory_scenario=memory_scenario,
    ).latency_ns


def _lookup_per_sequence_sample(
    perf_db,
    name,
    tp,
    sequences,
    memory_scenario=_LEGACY_MEMORY_SCENARIO,
):
    tp_eff = _effective_tp(
        perf_db,
        "per_sequence",
        name,
        tp,
        memory_scenario,
    )
    scenario_tables = (
        _tp_tables(perf_db, tp_eff).get("per_sequence", {}).get(name)
    )
    if scenario_tables is None:
        raise KeyError(
            f"Missing per-sequence profile for layer={name} on tp={tp_eff}."
        )
    tbl = _select_scenario_table(
        perf_db,
        scenario_tables,
        memory_scenario,
        category="per_sequence",
        layer_name=name,
    )
    return _lookup_1d_sample(tbl, max(int(sequences), 1))


def _axis_bracket(values, query):
    """Return (lo_idx, hi_idx, t) for log-space interpolation on
    ``values`` (sorted, non-negative, may include 0). ``t`` is the
    fractional position: 0 → use values[lo_idx], 1 → values[hi_idx].

    Below the min or above the max we clamp on the low side (value 0
    is treated as an exact sample) and extrapolate log-linearly on the
    high side using the top two samples.
    """
    n = len(values)
    if n == 0:
        raise KeyError("empty axis")
    if n == 1 or query <= values[0]:
        return 0, 0, 0.0
    idx = bisect.bisect_right(values, query)
    if idx >= n:
        lo, hi = n - 2, n - 1
    else:
        lo, hi = idx - 1, idx
    x0, x1 = values[lo], values[hi]
    if x1 == x0:
        return lo, hi, 0.0
    # log-space when both ends are positive; fall back to linear when
    # one end is 0 (0-valued sample is pinned exact).
    if x0 > 0 and x1 > 0:
        import math
        t = (math.log(max(query, 1e-9)) - math.log(x0)) / (math.log(x1) - math.log(x0))
    else:
        t = (query - x0) / (x1 - x0)
    return lo, hi, t


def _attn_slice_lookup(
    tbl,
    pc,
    nd,
    kv_prefill,
    kv_decode,
    audit_column=None,
):
    """Bilinear (log on each axis) within a single (pc, nd) slice."""
    slice_tbl = tbl["slices"].get((pc, nd))
    if slice_tbl is None:
        return None
    kp_vals = slice_tbl["kv_prefill_vals"]
    rows = slice_tbl["rows"]
    if not kp_vals:
        return None
    lo_kp, hi_kp, t_kp = _axis_bracket(kp_vals, max(int(kv_prefill), 0))

    def _row_lookup(row):
        ks = row["keys"]
        if audit_column is None:
            vs = row["values"]
        else:
            vs = (row.get("audit_values") or {}).get(audit_column)
            if vs is None:
                return None
        if not ks:
            return None
        if len(ks) == 1:
            return vs[0]
        lo, hi, t = _axis_bracket(ks, max(int(kv_decode), 0))
        return vs[lo] + t * (vs[hi] - vs[lo])

    v_lo = _row_lookup(rows[lo_kp])
    if lo_kp == hi_kp or v_lo is None:
        return v_lo
    v_hi = _row_lookup(rows[hi_kp])
    if v_hi is None:
        return v_lo
    return v_lo + t_kp * (v_hi - v_lo)


# ---------------------------------------------------------------------------
# Skew correction
# ---------------------------------------------------------------------------
# When the runtime batch has heterogeneous decode kv lengths, the
# profiled 4D grid (which carries one kv_decode value per shot) can
# only tell us the uniform-mean latency. Empirically that's faster
# than a truly skewed batch, because FlashAttention's varlen kernel
# suffers tile padding + SM-imbalance costs that the uniform-mean
# measurement misses.
#
# The skew profile (profiler/.../tp<N>/skew.csv + the fitted
# ``skew_fit`` block in meta.yaml, with the bucket alpha table spilled
# to ``tp<N>/skew_fit.csv``) captures this as a 5-axis lookup table of
# alpha values where
#
#     t_skew = t_mean + alpha * (t_max - t_mean)
#
# With alpha=0 (the pre-correction behaviour) the simulator
# systematically under-predicts attention latency by 5-10%, which
# compounds across every batch in a session into noticeable TTFT /
# TPOT drift vs. vLLM.
#
# Lookup is resolved per-batch via ``_skew_alpha``. The bin edges and
# labels come from ``meta.yaml::skew_fit.bucket_axes`` so the profiler
# can widen any axis (e.g. raise ``max_num_seqs`` above 128) without a
# coordinated code change here. The ``_DEFAULT_SKEW_AXES`` block below
# is used as a fallback only when the meta predates that field (which
# is why its shape still matches the original hard-coded scheme).
_ATTN_SKEW_ALPHA_FALLBACK: float = 0.093

_DEFAULT_SKEW_AXES: dict = {
    "n_bins": (0, 2, 4, 8, 16, 32, 64, 128, 1_000_000),
    "n_labels": (
        "n<=2", "n<=4", "n<=8", "n<=16", "n<=32", "n<=64", "n<=128", "n>128",
    ),
    "kv_big_bins": (0, 1024, 4096, 16384, 1_000_000_000),
    "kv_big_labels": ("kvB<=1k", "kvB<=4k", "kvB<=16k", "kvB>16k"),
    "skew_rate_bins": (-0.01, 0.05, 0.15, 0.40, 0.70, 1.01),
    "skew_rate_labels": ("sr<=5%", "sr<=15%", "sr<=40%", "sr<=70%", "sr>70%"),
    "kp_bins": (-1, 0, 2048, 1_000_000_000),
    "kp_labels": ("kp=0", "kp<=2k", "kp>2k"),
}


def _bucket_label(bins, labels, val) -> str:
    # Bucketing is (bins[i], bins[i+1]] — inclusive on the right so
    # the label matches its intuitive reading (``n<=8`` includes 8).
    for i in range(len(labels)):
        if val <= bins[i + 1]:
            return labels[i]
    return labels[-1]


def _resolve_skew_axes(fit_block, tp_entry):
    """Return the (bins, labels) axes used for key construction.

    Priority: per-TP entry > block top-level > module defaults. The
    per-TP override is primarily a transition path — the writer
    promotes ``bucket_axes`` to the top of the block when it's
    identical across TPs, which is the common case.
    """
    axes = None
    if isinstance(tp_entry, dict):
        axes = tp_entry.get("bucket_axes")
    if not axes and isinstance(fit_block, dict):
        axes = fit_block.get("bucket_axes")
    if not axes:
        return _DEFAULT_SKEW_AXES
    return axes


def _skew_alpha(
    perf_db,
    tp: int,
    pc: int,
    n: int,
    skew_rate: float,
    kv_big: int,
    kp: int,
    memory_scenario=_LEGACY_MEMORY_SCENARIO,
) -> float:
    """Resolve alpha for a specific batch from the profile's
    ``skew_fit`` meta block.

    Lookup order:
        1. meta.yaml::skew_fit.per_tp[tp].alpha_by_bucket[bucket_key]
           (hydrated from ``tp<N>/skew_fit.csv`` when the meta points
           at a CSV instead of inlining the mapping). The bucket_key
           is ``pc={pc}|{n_label}|{sr_label}|{kvb_label}|{kp_label}``,
           built against ``skew_fit.bucket_axes`` if present — which
           lets the profiler widen axes (more n bins, finer kp bins)
           without a simulator-side code change.
        2. meta.yaml::skew_fit.per_tp[tp].alpha_default (pooled WLS).
        3. Module-level fallback constant (``_ATTN_SKEW_ALPHA_FALLBACK``).

    v1 在配置缺失或禁用时沿用历史 fallback；v2 禁用时返回 0，
    避免把同一个 alpha 跨场景复用。
    """
    _resolve_memory_scenario(perf_db, memory_scenario)
    meta = perf_db.get("meta") if isinstance(perf_db, dict) else None
    if _is_profile_v2(perf_db):
        if not _v2_skew_enabled(meta):
            return 0.0
        raise ProfileV2RuntimeNotReadyError(
            "Profile v2 skew 尚未按 memory_scenario 隔离，拒绝复用 v1 alpha"
        )
    if not meta:
        return _ATTN_SKEW_ALPHA_FALLBACK
    fit_block = meta.get("skew_fit")
    if not fit_block or not fit_block.get("enabled"):
        return _ATTN_SKEW_ALPHA_FALLBACK
    per_tp = fit_block.get("per_tp") or {}
    entry = per_tp.get(tp) or per_tp.get(int(tp)) or per_tp.get(str(tp))
    if not entry:
        return float(fit_block.get("alpha_default", _ATTN_SKEW_ALPHA_FALLBACK))
    axes = _resolve_skew_axes(fit_block, entry)
    sr = max(0.0, min(1.0, float(skew_rate)))
    n_label = _bucket_label(axes["n_bins"], axes["n_labels"], int(n))
    sr_label = _bucket_label(
        axes["skew_rate_bins"], axes["skew_rate_labels"], sr,
    )
    kvb_label = _bucket_label(
        axes["kv_big_bins"], axes["kv_big_labels"], int(kv_big),
    )
    kp_label = _bucket_label(axes["kp_bins"], axes["kp_labels"], int(kp))
    key = f"pc={int(pc)}|{n_label}|{sr_label}|{kvb_label}|{kp_label}"
    alphas = entry.get("alpha_by_bucket") or {}
    if key in alphas:
        return float(alphas[key])
    return float(entry.get("alpha_default", _ATTN_SKEW_ALPHA_FALLBACK))


def _lookup_attention_with_skew(
    perf_db, tp, prefill_chunk, kv_prefill,
    n_decode, kv_decode_mean, kv_decode_max, kv_decode_min,
    memory_scenario=_LEGACY_MEMORY_SCENARIO,
):
    """Attention lookup with skew correction applied.

    Two 4D interpolations — at kv_decode_mean (the canonical point
    the profiler measured) and kv_decode_max (the per-batch longest
    decode sequence) — combined by the bucket-specific alpha resolved
    from ``meta.yaml::skew_fit``. Skipped entirely when there's no
    skew to correct (single decode or all decodes at the same length).

    Returns an integer nanosecond count. The raw bilinear interp in
    ``_lookup_attention`` produces a float, and the skew formula
    compounds that; the Chakra trace converter requires integer
    ``comp_time`` so we round here.
    """
    t_mean = _lookup_attention(
        perf_db, tp, prefill_chunk, kv_prefill, n_decode, kv_decode_mean,
        memory_scenario=memory_scenario,
    )
    # No skew → no correction (also saves a redundant lookup).
    if n_decode <= 1 or kv_decode_max == kv_decode_mean:
        return max(1, int(round(t_mean)))
    # skew_rate ∈ [0, 1]; = nb / n exactly for a bimodal batch.
    # Fallback to 0.5 (balanced) when kv_max == kv_min (shouldn't
    # reach here due to the short-circuit above, but defensive).
    kv_gap = kv_decode_max - kv_decode_min
    skew_rate = (kv_decode_mean - kv_decode_min) / kv_gap if kv_gap > 0 else 0.5
    alpha = _skew_alpha(
        perf_db, tp, prefill_chunk, n_decode, skew_rate, kv_decode_max,
        kv_prefill, memory_scenario=memory_scenario,
    )
    if alpha == 0.0:
        return max(1, int(round(t_mean)))
    t_max = _lookup_attention(
        perf_db, tp, prefill_chunk, kv_prefill, n_decode, kv_decode_max,
        memory_scenario=memory_scenario,
    )
    # Guard against interpolation producing t_max < t_mean (can happen
    # at the axis boundary); in that case the formula would produce a
    # negative correction, which isn't physical.
    if t_max <= t_mean:
        return max(1, int(round(t_mean)))
    return max(1, int(round(t_mean + alpha * (t_max - t_mean))))


def _lookup_attention(
    perf_db,
    tp,
    prefill_chunk,
    kv_prefill,
    n_decode,
    kv_decode,
    memory_scenario=_LEGACY_MEMORY_SCENARIO,
):
    return _lookup_attention_sample(
        perf_db,
        tp,
        prefill_chunk,
        kv_prefill,
        n_decode,
        kv_decode,
        memory_scenario=memory_scenario,
    ).latency_ns


def _lookup_attention_sample(
    perf_db,
    tp,
    prefill_chunk,
    kv_prefill,
    n_decode,
    kv_decode,
    memory_scenario=_LEGACY_MEMORY_SCENARIO,
):
    """4D log-linear interpolation on (prefill_chunk, kv_prefill,
    n_decode, kv_decode). Every axis is doubled by the profiler, so we
    bracket each axis's two nearest profiled values and blend linearly
    in log-space.
    """
    _validate_v2_attention_query(
        perf_db,
        prefill_chunk=prefill_chunk,
        kv_prefill=kv_prefill,
        n_decode=n_decode,
        kv_decode=kv_decode,
    )
    scenario_tables = _tp_tables(perf_db, tp).get("attention")
    if scenario_tables is None:
        raise KeyError(f"Missing attention profile for tp={tp}.")
    tbl = _select_scenario_table(
        perf_db,
        scenario_tables,
        memory_scenario,
        category="attention",
    )
    if not tbl["pc_nd_pairs"]:
        raise KeyError(f"Missing attention profile for tp={tp}.")

    pcq, ndq = max(int(prefill_chunk), 0), max(int(n_decode), 0)
    latency_ns = _lookup_attention_table_value(
        tbl,
        pcq,
        ndq,
        kv_prefill,
        kv_decode,
    )
    audit_values = {
        column: _lookup_attention_table_value(
            tbl,
            pcq,
            ndq,
            kv_prefill,
            kv_decode,
            audit_column=column,
        )
        for column in _FOUR_WAY_AUDIT_COLUMNS
    }
    return _sample_from_components(latency_ns, audit_values)


def _validate_v2_attention_query(
    perf_db,
    *,
    prefill_chunk,
    kv_prefill,
    n_decode,
    kv_decode,
):
    """拒绝 v2 Attention 查询越过已验证扫描范围。"""

    if not _is_profile_v2(perf_db):
        return
    engine = perf_db.get("engine_effective") or {}
    grid = perf_db.get("attention_grid") or {}
    limits = (
        ("prefill_chunk", int(prefill_chunk), engine.get("max_num_batched_tokens")),
        ("n_decode", int(n_decode), engine.get("max_num_seqs")),
        ("kv_prefill", int(kv_prefill), grid.get("max_kv")),
        ("kv_decode", int(kv_decode), grid.get("max_kv")),
    )
    for field, value, limit in limits:
        if not isinstance(limit, int) or limit <= 0:
            raise ProfileV2RuntimeNotReadyError(
                f"Profile v2 缺少 {field} 的有效扫描上限"
            )
        if value > limit:
            raise ProfileV2RuntimeNotReadyError(
                f"Attention {field}={value} 超过 Profile 验证范围 {limit}"
            )


def _lookup_attention_table_value(
    tbl,
    pcq,
    ndq,
    kv_prefill,
    kv_decode,
    *,
    audit_column=None,
):
    """在同一 Attention 网格上插值延迟或一个审计方向。"""

    pc_vals, nd_vals = tbl["pc_vals"], tbl["nd_vals"]
    lo_pc, hi_pc, t_pc = _axis_bracket(pc_vals, pcq)
    lo_nd, hi_nd, t_nd = _axis_bracket(nd_vals, ndq)

    # Grab the four corners; missing corners fall back to the closest
    # available (pc, nd) pair.
    def _corner(pc, nd):
        v = _attn_slice_lookup(
            tbl,
            pc,
            nd,
            kv_prefill,
            kv_decode,
            audit_column=audit_column,
        )
        if v is not None:
            return v
        nearest = min(tbl["pc_nd_pairs"],
                      key=lambda p: (p[0] - pc) ** 2 + (p[1] - nd) ** 2)
        return _attn_slice_lookup(
            tbl,
            nearest[0],
            nearest[1],
            kv_prefill,
            kv_decode,
            audit_column=audit_column,
        ) or 0.0

    c00 = _corner(pc_vals[lo_pc], nd_vals[lo_nd])
    c01 = _corner(pc_vals[lo_pc], nd_vals[hi_nd])
    c10 = _corner(pc_vals[hi_pc], nd_vals[lo_nd])
    c11 = _corner(pc_vals[hi_pc], nd_vals[hi_nd])

    v0 = c00 + t_nd * (c01 - c00)
    v1 = c10 + t_nd * (c11 - c10)
    return v0 + t_pc * (v1 - v0)


def _lookup_moe(
    perf_db,
    tokens,
    activated_experts,
    memory_scenario=_LEGACY_MEMORY_SCENARIO,
):
    return _lookup_moe_sample(
        perf_db,
        tokens,
        activated_experts,
        memory_scenario=memory_scenario,
    ).latency_ns


def _lookup_moe_sample(
    perf_db,
    tokens,
    activated_experts,
    memory_scenario=_LEGACY_MEMORY_SCENARIO,
):
    """MoE is profiled once at tp=1 (single-rank view); the simulator
    looks up per EP-rank token counts.
    """
    tp_eff = 1 if 1 in perf_db["available_tps"] else perf_db["available_tps"][0]
    scenario_tables = _tp_tables(perf_db, tp_eff).get("moe")
    if scenario_tables is None:
        raise KeyError(
            f"Missing moe profile. Check that moe.csv exists under "
            f"perf/{perf_db['hardware']}/{perf_db['model']}/{perf_db['variant']}/tp{tp_eff}/."
        )
    tbl = _select_scenario_table(
        perf_db,
        scenario_tables,
        memory_scenario,
        category="moe",
    )
    ae_vals = tbl["activated_experts_vals"]
    rows = tbl["rows"]
    aeq = max(int(activated_experts), 1)
    tokq = max(int(tokens), 1)
    lo, hi = _lookup_bounds(ae_vals, aeq)
    sample_lo = _lookup_1d_sample(rows[lo], tokq)
    if lo == hi:
        return sample_lo
    sample_hi = _lookup_1d_sample(rows[hi], tokq)
    latency_ns = _linear_interpolate(
        ae_vals[lo],
        sample_lo.latency_ns,
        ae_vals[hi],
        sample_hi.latency_ns,
        aeq,
    )
    audit_values = {
        column: _linear_interpolate(
            ae_vals[lo],
            getattr(sample_lo, column),
            ae_vals[hi],
            getattr(sample_hi, column),
            aeq,
        )
        for column in _FOUR_WAY_AUDIT_COLUMNS
    }
    return _sample_from_components(latency_ns, audit_values)


def _catalog_has(perf_db, category, name):
    section = perf_db["architecture"]["catalog"].get(category) or {}
    return name in section


# ======================================================================
# Context builders
# ======================================================================

def _validate_policy_against_profile(policy, perf_db):
    """在创建 TraceCtx 前验证静态策略引用的场景与层。"""
    if not isinstance(policy, MemoryScenarioPolicy):
        raise TypeError("memory_scenario_policy 必须是 MemoryScenarioPolicy")
    if not policy.is_v2:
        return
    if not _is_profile_v2(perf_db):
        raise RuntimeError("memory_scenario_v2 策略必须加载 Profile v2 bundle")

    scenario_catalog = perf_db.get("scenario_catalog") or {}
    referenced_scenarios = {
        policy.default_scenario,
        *policy.layer_scenarios.values(),
        *policy.block_scenarios.values(),
    }
    unknown_scenarios = sorted(referenced_scenarios - set(scenario_catalog))
    if unknown_scenarios:
        raise KeyError(
            f"scenario_policy 引用了未知 memory_scenario：{unknown_scenarios}；"
            f"可用场景：{sorted(scenario_catalog)}"
        )

    architecture_catalog = perf_db["architecture"].get("catalog") or {}
    canonical_layers = {
        layer_name
        for category in architecture_catalog.values()
        for layer_name in (category or {})
    }
    unknown_layers = sorted(set(policy.layer_scenarios) - canonical_layers)
    if unknown_layers:
        raise KeyError(
            f"scenario_policy.layers 不是当前 architecture catalog 的精确成员："
            f"{unknown_layers}"
        )


def _scenario_for(ctx, layer_name, layer_num):
    """统一解析每次 Profile 查询使用的静态场景。"""
    policy = ctx.memory_scenario_policy
    if policy is None:
        return None
    return policy.scenario_for(layer_name, layer_num)

def _build_trace_ctx(hardware, model, config, tp_size, pp_size, local_ep, ep_total, node_id, fp,
                     placement, gate, enable_attn_offloading, power_model, pim_model, pd_type,
                     variant, kv_cache_dtype='auto',
                     runtime_max_num_batched_tokens=None, runtime_max_num_seqs=None,
                     runtime_block_size=None,
                     tp_dim=None, ep_dim=None, dp_sum_total_len=0,
                     memory_scenario_policy=None):
    model_type = config.get('model_type')
    if not model_type:
        raise KeyError(
            f"Model config for {model!r} has no 'model_type'; cannot locate "
            f"profiler/models/<model_type>.yaml"
        )
    if memory_scenario_policy is None:
        memory_scenario_policy = parse_instance_performance_profile(
            {},
            config["num_hidden_layers"],
        )
    elif not isinstance(memory_scenario_policy, MemoryScenarioPolicy):
        raise TypeError("memory_scenario_policy 必须是 MemoryScenarioPolicy")
    if memory_scenario_policy.num_hidden_layers != config["num_hidden_layers"]:
        raise ValueError(
            "memory_scenario_policy.num_hidden_layers 与模型配置不一致"
        )

    tp_needed = {max(int(tp_size), 1)}
    if memory_scenario_policy.is_v2:
        perf_db = _load_perf_db(
            hardware,
            model,
            variant,
            tp_needed,
            model_type,
            memory_profile_id=memory_scenario_policy.memory_profile_id,
            model_config=config,
            runtime_max_num_batched_tokens=runtime_max_num_batched_tokens,
            runtime_max_num_seqs=runtime_max_num_seqs,
            runtime_block_size=runtime_block_size,
        )
    else:
        perf_db = _load_perf_db(
            hardware,
            model,
            variant,
            tp_needed,
            model_type,
        )
    _validate_policy_against_profile(memory_scenario_policy, perf_db)
    warn_if_runtime_exceeds_profiled(
        perf_db, runtime_max_num_batched_tokens, runtime_max_num_seqs)

    n_embd = config['hidden_size']
    n_head = config['num_attention_heads']
    kv_head = config.get('num_key_value_heads', n_head)
    head_dim = config.get('head_dim', n_embd // n_head)
    is_moe = gate is not None

    pim_channels = 0
    if enable_attn_offloading and pim_model is not None:
        pim_config = pim_model.get_config()
        pim_channels = int(pim_config["mem_size"] // pim_config["dimm_size"])

    return TraceCtx(
        hardware=hardware, model=model, config=config, perf_db=perf_db,
        node_id=node_id,
        fp=fp, placement=placement, gate=gate,
        enable_attn_offloading=enable_attn_offloading,
        power_model=power_model, pim_model=pim_model, pim_channels=pim_channels,
        n_head=n_head, kv_head=kv_head, head_dim=head_dim, is_moe=is_moe,
        pd_type=pd_type,
        tp_size=tp_size, pp_size=pp_size, local_ep=local_ep, ep_total=ep_total,
        tp_dim=tp_dim, ep_dim=ep_dim, dp_sum_total_len=dp_sum_total_len,
        memory_scenario_policy=memory_scenario_policy,
    )


def _build_batch_ctx(batch, ctx):
    # batch.total_len is the number of tokens actually computed this iteration:
    # the scheduler builds it from chunk_size = original_input - num_computed_tokens,
    # and num_computed_tokens already absorbs any prefix-cache hit, so no further
    # subtraction is needed even when prefix caching is on.
    total_len = batch.total_len
    # DP padding (see serving.__main__._pad_batch_to_max) adds dummy decodes without
    # touching batch.requests. vLLM keeps lm_head's output shape pinned to
    # num_tokens_after_padding for CUDA-graph replay, so each padded decode
    # also contributes a logit. Track it via num_prefill + num_decode.
    lm_head_len = max(len(batch.requests), batch.num_prefill + batch.num_decode)

    # 4D attention keys: profiler sweeps (prefill_chunk, kv_prefill,
    # n_decode, kv_decode). The kv_decode axis carries a single value
    # per shot, so we collapse multi-decode requests to their mean
    # AND capture the per-batch max/min for the skew correction below.
    prefill_chunk = sum(batch.prefill_q_list)
    kv_prefill = sum(batch.prefill_k_list)
    n_decode = len(batch.decode_k_list)
    kv_decode_mean = (sum(batch.decode_k_list) // n_decode) if n_decode > 0 else 0
    kv_decode_max = max(batch.decode_k_list) if n_decode > 0 else 0
    kv_decode_min = min(batch.decode_k_list) if n_decode > 0 else 0

    # PIM offloading: NPU sees only the prefill portion.
    decode_lens = None
    channel_split = 0
    if ctx.enable_attn_offloading and ctx.pim_model is not None:
        channel_split = min(ctx.pim_channels, ctx.kv_head)
        _, decode_lens = _attn_load_balancer(batch.requests, ctx.tp_size, ctx.pim_channels, channel_split)
        n_decode = 0
        kv_decode_mean = 0
        kv_decode_max = 0
        kv_decode_min = 0
        total_len = max(1, total_len)  # preserve for size calcs

    return BatchCtx(batch, total_len, prefill_chunk, kv_prefill, n_decode,
                    kv_decode_mean, kv_decode_max, kv_decode_min,
                    lm_head_len, decode_lens, channel_split)


# ======================================================================
# Layer emission helpers
# ======================================================================

def _layer_category(perf_db, layer_name):
    """Return which catalog category (dense/per_sequence/attention/moe)
    a canonical layer belongs to for this architecture, or None if the
    catalog doesn't include it.
    """
    for cat in ("per_sequence", "attention", "moe", "dense"):
        if _catalog_has(perf_db, cat, layer_name):
            return cat
    return None


def _emit_layer(ctx, bctx, layer_name, lines, power_acc, batch_tag='NONE', layer_num=None,
                comm_type='NONE', comm_size=0, input_loc='LOCAL', output_loc='LOCAL'):
    """Emit a single trace layer: lookup latency, compute sizes, format, track power."""
    category = _layer_category(ctx.perf_db, layer_name)
    if category is None:
        raise KeyError(
            f"Layer {layer_name!r} is not declared in the architecture yaml "
            f"catalog for {ctx.perf_db['variant']}. Add it to "
            f"profiler/models/<model_type>.yaml or remove it from the sequence."
        )

    memory_scenario = _scenario_for(ctx, layer_name, layer_num)
    if category == "per_sequence":
        latency_ns = _lookup_per_sequence(
            ctx.perf_db,
            layer_name,
            ctx.tp_size,
            bctx.lm_head_len,
            memory_scenario=memory_scenario,
        )
    elif category == "attention":
        latency_ns = _lookup_attention_with_skew(
            ctx.perf_db, ctx.tp_size,
            bctx.prefill_chunk, bctx.kv_prefill,
            bctx.n_decode, bctx.kv_decode_mean, bctx.kv_decode_max,
            bctx.kv_decode_min,
            memory_scenario=memory_scenario,
        )
    else:  # dense
        latency_ns = _lookup_dense(
            ctx.perf_db,
            layer_name,
            ctx.tp_size,
            bctx.total_len,
            memory_scenario=memory_scenario,
        )

    # Size calculation uses the same canonical layer names.
    if layer_name == 'attention':
        kv_len_for_sizes = bctx.kv_prefill + bctx.n_decode * bctx.kv_decode_mean
        inp, wt, out = calculate_sizes(ctx.model, layer_name, bctx.total_len,
                                       kv_len=kv_len_for_sizes,
                                       parallel=ctx.tp_size, fp=ctx.fp)
    else:
        inp, wt, out = calculate_sizes(ctx.model, layer_name, bctx.total_len,
                                       parallel=ctx.tp_size, fp=ctx.fp)

    wt_loc = get_device(ctx.placement, layer_num, layer_name, "weights")

    lines.append(formatter(layer_name, str(latency_ns), input_loc, str(inp), wt_loc, str(wt), output_loc, str(out), comm_type, str(comm_size), batch_tag))

    if power_acc is not None:
        power_acc.npu_latencies_ns.append(latency_ns)
        if wt_loc != 'LOCAL':
            power_acc.dram_weight_bytes += wt
        if comm_size > 0:
            collective = comm_type.split(':', 1)[0].lower()
            power_acc.link_data_bytes += total_ring_data(comm_size, ctx.tp_size, collective=collective)

    return latency_ns


def _tp_comm(ctx, layer_name, total_len, collective='ALLREDUCE'):
    """Compute TP communication size. Returns (comm_size, comm_type)."""
    if ctx.tp_size <= 1:
        return 0, 'NONE'
    _, _, out = calculate_sizes(ctx.model, layer_name, total_len, parallel=ctx.tp_size, fp=ctx.fp)
    return out, collective


def _with_dim(comm_type, involved_dim):
    """Encode involved_dim into comm_type string: 'ALLREDUCE' + [T,F] -> 'ALLREDUCE:1,0'."""
    if involved_dim is None or comm_type == 'NONE':
        return comm_type
    dim_str = ','.join('1' if d else '0' for d in involved_dim)
    return f"{comm_type}:{dim_str}"


def _emit_pim_attention(ctx, bctx, lines, power_acc, layer_num, batch_tag='NONE'):
    """Emit PIM attention for decode requests across PIM channels."""
    for ch in range(ctx.pim_channels):
        lines.append(f"PIM {ch}\n")
        for L in bctx.decode_lens[ch]:
            inp, _, out = calculate_sizes(ctx.model, "attention", L, pim=True, parallel=ctx.tp_size, fp=ctx.fp)
            inp //= bctx.channel_split
            out //= bctx.channel_split
            pim_lat = int(ctx.pim_model.get_pim_latency(ctx.n_head, ctx.kv_head, ctx.head_dim, L, bctx.channel_split))
            lines.append(formatter("attention", str(pim_lat),
                f'REMOTE:{ctx.node_id}.{ch}', str(inp),
                get_device(ctx.placement, layer_num, "attention", "weights"), '0',
                f'REMOTE:{ctx.node_id}.{ch}', str(out),
                'NONE', '0', batch_tag))
            if power_acc is not None and pim_lat > 0:
                power_acc.pim_latencies_ns.append(pim_lat)
                power_acc.dram_weight_bytes += inp + out
    lines.append("PIM END\n")


def _emit_npu_attention(ctx, bctx, lines, power_acc, layer_num, batch_tag='NONE'):
    """Emit NPU attention (unified prefill+decode lookup)."""
    if bctx.prefill_chunk == 0 and bctx.n_decode == 0:
        return
    _emit_layer(ctx, bctx, "attention", lines, power_acc, batch_tag, layer_num)


def _emit_moe_block(ctx, bctx, lines, power_acc, layer_num, batch_id_str, batch_tag='NONE'):
    """Emit MoE block: dispatch ALLTOALL + per-EP-rank expert compute + combine ALLTOALL.

    Each EP rank receives a different number of tokens based on expert routing.
    Per-rank latency is looked up independently from profiled data at tp=1.
    ALLTOALL is handled by ASTRA-Sim with involved_dim scoping for DP groups,
    or as a simple ALLTOALL for local EP groups.
    """
    ep_total = ctx.ep_total

    # MoE compute uses ``bctx.total_len`` (= per-rank padded count after
    # ``_pad_batch_to_max``), matching how the real vLLM kernel runs on the
    # full padded forward shape. Routing / ``_lookup_moe`` therefore see
    # the same per-rank padded value as every other dense layer.
    effective_total_len_compute = bctx.total_len
    routing = ctx.gate.route_ep(layer_num, batch_id_str, effective_total_len_compute, ep_total)

    # AG/RS comm sizes are anchored to ``dp_sum_total_len``, which
    # ``serving/__main__.py`` sets to ``max_total_len`` (NOT ``max × dp_group_size``)
    # for DP groups; this calibrates the AG/RS bandwidth model against the same
    # ``link_bw`` that already matches AllReduce. Falls back to this rank's own
    # ``total_len`` when DP is inactive.
    effective_total_len_comm = ctx.dp_sum_total_len if ctx.dp_sum_total_len > 0 else bctx.total_len

    # vLLM default ``allgather_reducescatter`` backend: dispatch = AllGather
    # (hidden + router_logits), combine = ReduceScatter (hidden only).
    # ASTRA-Sim AG ``data_size`` is per-rank local chunk (sum / ep_total);
    # RS ``data_size`` is the pre-scatter total buffer.
    n_embd = ctx.config['hidden_size']
    num_experts = ctx.config.get('num_local_experts', ctx.config.get('num_experts', 0))
    dispatch_per_token = (n_embd + num_experts) * ctx.fp
    combine_per_token = n_embd * ctx.fp
    ag_per_rank_tokens = max(1, effective_total_len_comm // max(ep_total, 1))
    dispatch_comm_size = ag_per_rank_tokens * dispatch_per_token
    combine_comm_size = effective_total_len_comm * combine_per_token

    if ep_total > 1:
        dispatch_comm_type = _with_dim('ALLGATHER', ctx.ep_dim)
        combine_comm_type = _with_dim('REDUCESCATTER', ctx.ep_dim)
    else:
        dispatch_comm_type = 'NONE'
        combine_comm_type = 'NONE'
        dispatch_comm_size = 0
        combine_comm_size = 0

    wt_loc = get_device(ctx.placement, layer_num, "moe", "weights")
    memory_scenario = _scenario_for(ctx, "moe", layer_num)

    # Each local GPU handles exactly one EP rank. The routing result already
    # accounts for cross-instance token redistribution (ALLTOALL), so
    # local_tokens[rank] reflects the post-dispatch workload for that rank.
    emit_ep = max(ctx.local_ep, 1)
    max_rank_latency_ns = 0

    # Pre-expert AllGather power (dispatch)
    if power_acc is not None and ep_total > 1:
        power_acc.link_data_bytes += total_ring_data(dispatch_comm_size, ep_total, collective="allgather")

    for i in range(emit_ep):
        if i == 0:
            lines.append(f"EXPERT {i} {dispatch_comm_type} {dispatch_comm_size}\n")
        else:
            lines.append(f"EXPERT {i} NONE 0\n")

        # ``local_tokens`` here is the per-rank workload after dispatch
        # — already scaled to this rank's real tokens (no DP-padding sum).
        # We feed it straight into the MoE profile lookup.
        local_tokens = routing.local_tokens[i]
        activated_experts = routing.activated_experts[i]

        if local_tokens > 0:
            rank_latency_ns = _lookup_moe(
                ctx.perf_db,
                local_tokens,
                max(activated_experts, 1),
                memory_scenario=memory_scenario,
            )
            rank_inp, rank_wt, rank_out = calculate_sizes(
                ctx.model, "moe", local_tokens, parallel=ep_total, fp=ctx.fp)
            max_rank_latency_ns = max(max_rank_latency_ns, rank_latency_ns)

            lines.append(formatter("expert", str(rank_latency_ns), 'LOCAL', str(rank_inp),
                wt_loc, str(rank_wt), 'LOCAL', str(rank_out), 'NONE', '0', batch_tag))

            if power_acc is not None and wt_loc != 'LOCAL':
                power_acc.dram_weight_bytes += rank_wt

    # Power: all local GPUs are active for the duration of the slowest rank
    if power_acc is not None and max_rank_latency_ns > 0:
        power_acc.npu_latencies_ns.append(max_rank_latency_ns)

    lines.append(f"EXPERT END {combine_comm_type} {combine_comm_size}\n")

    # Post-expert ReduceScatter power (combine)
    if power_acc is not None and ep_total > 1:
        power_acc.link_data_bytes += total_ring_data(combine_comm_size, ep_total, collective="reducescatter")


# ======================================================================
# Block builders (split for interleaving)
# ======================================================================

def _sequence(perf_db, section):
    """Fetch a sequence list from the architecture yaml, defaulting to
    an empty list when the section is absent (e.g. mlp_moe for dense
    architectures).
    """
    seq = perf_db["architecture"].get("sequence") or {}
    return list(seq.get(section) or [])


_skipped_layer_warned = set()


def _layer_available(
    perf_db,
    tp,
    layer_name,
    memory_scenario=_LEGACY_MEMORY_SCENARIO,
):
    """Return True when the CSV-backed table has data for this layer at
    the TP the simulator is about to query. Attention/MoE are always
    present when their category CSVs exist.
    """
    if _is_profile_v2(perf_db):
        _resolve_memory_scenario(perf_db, memory_scenario)
    category = _layer_category(perf_db, layer_name)
    if category is None:
        return False
    tp_eff = _effective_tp(
        perf_db,
        category,
        layer_name,
        tp,
        memory_scenario,
    )
    tables = perf_db["tables"].get(tp_eff, {})
    if category == "dense":
        scenario_tables = tables.get("dense", {}).get(layer_name)
        if scenario_tables is None:
            if _is_profile_v2(perf_db):
                raise KeyError(
                    f"Missing dense profile for layer={layer_name!r} "
                    f"on tp={tp_eff}."
                )
            return False
        _select_scenario_table(
            perf_db,
            scenario_tables,
            memory_scenario,
            category=category,
            layer_name=layer_name,
        )
        return True
    if category == "per_sequence":
        scenario_tables = tables.get("per_sequence", {}).get(layer_name)
        if scenario_tables is None:
            if _is_profile_v2(perf_db):
                raise KeyError(
                    f"Missing per_sequence profile for layer={layer_name!r} "
                    f"on tp={tp_eff}."
                )
            return False
        _select_scenario_table(
            perf_db,
            scenario_tables,
            memory_scenario,
            category=category,
            layer_name=layer_name,
        )
        return True
    if category == "attention":
        scenario_tables = tables.get("attention")
        if not scenario_tables:
            if _is_profile_v2(perf_db):
                raise KeyError(f"Missing attention profile on tp={tp_eff}.")
            return False
        _select_scenario_table(
            perf_db,
            scenario_tables,
            memory_scenario,
            category=category,
        )
        return True
    if category == "moe":
        scenario_tables = tables.get("moe")
        if not scenario_tables:
            if _is_profile_v2(perf_db):
                raise KeyError(f"Missing moe profile on tp={tp_eff}.")
            return False
        _select_scenario_table(
            perf_db,
            scenario_tables,
            memory_scenario,
            category=category,
        )
        return True
    return False


def _emit_sequence(ctx, bctx, layer_num, layers, lines, power_acc, batch_tag):
    """Walk a flat list of canonical layer names from the architecture
    yaml, emitting each. ``attention`` triggers PIM attention before the
    NPU kernel when attn offloading is enabled; layers in
    ``_TP_ALLREDUCE_AFTER`` get an ALLREDUCE attached. When a layer is
    declared in the sequence but the profile CSV lacks data for it
    (e.g., an older profile run that predates a yaml addition), the
    emission is skipped with a single warning per (variant, layer).
    """
    for layer_name in layers:
        if layer_name == "attention":
            if ctx.enable_attn_offloading:
                _emit_pim_attention(ctx, bctx, lines, power_acc, layer_num, batch_tag)
            _emit_npu_attention(ctx, bctx, lines, power_acc, layer_num, batch_tag)
            continue
        if layer_name == "rotary_emb" and "TPU" in ctx.hardware:
            continue
        memory_scenario = _scenario_for(ctx, layer_name, layer_num)
        if not _layer_available(
            ctx.perf_db,
            ctx.tp_size,
            layer_name,
            memory_scenario=memory_scenario,
        ):
            key = (ctx.perf_db["variant"], ctx.perf_db["model"], layer_name)
            if ctx.perf_db.get("memory_profile_id") is not None:
                key = (
                    ctx.perf_db["variant"],
                    ctx.perf_db["model"],
                    ctx.perf_db["memory_profile_id"],
                    layer_name,
                    memory_scenario,
                )
            if key not in _skipped_layer_warned:
                _skipped_layer_warned.add(key)
                logger.warning(
                    "Layer %r is in the architecture yaml sequence but missing from "
                    "the profile CSVs for %s/%s/%s — skipping. Re-profile to include it.",
                    layer_name, ctx.perf_db["hardware"],
                    ctx.perf_db["model"], ctx.perf_db["variant"],
                )
            continue
        if layer_name in _TP_ALLREDUCE_AFTER:
            comm_size, comm_type = _tp_comm(ctx, layer_name, bctx.total_len)
            _emit_layer(ctx, bctx, layer_name, lines, power_acc, batch_tag, layer_num,
                        comm_type=_with_dim(comm_type, ctx.tp_dim), comm_size=comm_size)
        else:
            _emit_layer(ctx, bctx, layer_name, lines, power_acc, batch_tag, layer_num)


def _emit_pre_attn_layers(ctx, bctx, layer_num, lines, power_acc, batch_tag='NONE'):
    _emit_sequence(ctx, bctx, layer_num, _sequence(ctx.perf_db, "pre_attn"),
                   lines, power_acc, batch_tag)


def _emit_post_attn_layers(ctx, bctx, layer_num, lines, power_acc, batch_id_str, batch_tag='NONE'):
    # Attention post-processing common to dense and MoE.
    _emit_sequence(ctx, bctx, layer_num, _sequence(ctx.perf_db, "post_attn"),
                   lines, power_acc, batch_tag)
    # MLP: either the dense FFN stack or a single MoE block.
    if ctx.is_moe:
        moe_seq = _sequence(ctx.perf_db, "mlp_moe")
        for layer_name in moe_seq:
            if layer_name == "moe":
                _emit_moe_block(ctx, bctx, lines, power_acc, layer_num, batch_id_str, batch_tag)
            else:
                _emit_sequence(ctx, bctx, layer_num, [layer_name], lines, power_acc, batch_tag)
    else:
        _emit_sequence(ctx, bctx, layer_num, _sequence(ctx.perf_db, "mlp_dense"),
                       lines, power_acc, batch_tag)


def _build_transformer_block(ctx, bctx, layer_num, batch_tag, batch_id_str):
    """Build a complete transformer block. Returns (lines, PowerAccumulator)."""
    lines = []
    power_acc = PowerAccumulator([], [], 0, 0)
    _emit_pre_attn_layers(ctx, bctx, layer_num, lines, power_acc, batch_tag)
    _emit_post_attn_layers(ctx, bctx, layer_num, lines, power_acc, batch_id_str, batch_tag)
    return lines, power_acc


# ======================================================================
# Final layers and power helpers
# ======================================================================

def _layer_latency_for_power(ctx, bctx, layer_name, layer_num=None):
    """Per-layer latency lookup used purely for power accounting; the
    trace writes its own values via _emit_layer and _emit_moe_block.
    """
    category = _layer_category(ctx.perf_db, layer_name)
    memory_scenario = _scenario_for(ctx, layer_name, layer_num)
    if category == "per_sequence":
        return _lookup_per_sequence(
            ctx.perf_db,
            layer_name,
            ctx.tp_size,
            bctx.lm_head_len,
            memory_scenario=memory_scenario,
        )
    if category == "attention":
        return _lookup_attention_with_skew(
            ctx.perf_db, ctx.tp_size,
            bctx.prefill_chunk, bctx.kv_prefill,
            bctx.n_decode, bctx.kv_decode_mean, bctx.kv_decode_max,
            bctx.kv_decode_min,
            memory_scenario=memory_scenario,
        )
    return _lookup_dense(
        ctx.perf_db,
        layer_name,
        ctx.tp_size,
        bctx.total_len,
        memory_scenario=memory_scenario,
    )


def _emit_final_layers(ctx, bctx, f, batch_tag='NONE'):
    """Emit the architecture's head layers (final_layernorm, lm_head,
    sampler — ordered per the yaml) and feed them into the power model.
    The last emitted layer routes its output to REMOTE so the Chakra
    converter places a MEM_STORE node back to CPU.
    """
    head_layers = _sequence(ctx.perf_db, "head")
    lines = []
    for i, layer_name in enumerate(head_layers):
        output_loc = f'REMOTE:{ctx.node_id}' if i == len(head_layers) - 1 else 'LOCAL'
        _emit_layer(ctx, bctx, layer_name, lines, None, batch_tag, output_loc=output_loc)
    f.writelines(lines)

    if ctx.power_model is not None:
        for layer_name in head_layers:
            lat = _layer_latency_for_power(ctx, bctx, layer_name)
            ctx.power_model.add_npu_active_energy_consumption(ctx.hardware, ctx.node_id, lat, num_npus=ctx.tp_size)
            if get_device(ctx.placement, None, layer_name, "weights") != 'LOCAL':
                _, wt, _ = calculate_sizes(ctx.model, layer_name, bctx.total_len, parallel=ctx.tp_size, fp=ctx.fp)
                ctx.power_model.add_dram_energy_consumption(ctx.node_id, wt)


def _emit_pp_pd_power(ctx, bctx):
    """Emit pipeline parallelism and P/D sync power."""
    if ctx.power_model is None:
        return
    if ctx.pp_size > 1:
        pp_comm = bctx.total_len * ctx.config['hidden_size'] * (ctx.pp_size - 1)
        ctx.power_model.add_link_energy_consumption(ctx.node_id, pp_comm)
    if ctx.pd_type == 'prefill':
        kv_comm = bctx.total_len * ctx.config['hidden_size'] * ctx.fp
        out_size = bctx.lm_head_len * ctx.config['hidden_size'] * ctx.fp
        ctx.power_model.add_link_energy_consumption(ctx.node_id, kv_comm + out_size)


# ======================================================================
# _synthesize_trace (non-interleaved)
# ======================================================================

def _emit_prologue(ctx, bctx, f, batch_tag='NONE'):
    """Emit prologue layers (typically just embedding). The first layer's
    input is routed from REMOTE to match the Chakra converter's
    MEM_LOAD node placement.
    """
    prologue_layers = _sequence(ctx.perf_db, "prologue")
    if not prologue_layers:
        return
    lines = []
    for i, layer_name in enumerate(prologue_layers):
        input_loc = f'REMOTE:{ctx.node_id}' if i == 0 else 'LOCAL'
        _emit_layer(ctx, bctx, layer_name, lines, None, batch_tag, input_loc=input_loc)
    f.writelines(lines)
    if ctx.power_model:
        for layer_name in prologue_layers:
            lat = _layer_latency_for_power(ctx, bctx, layer_name)
            ctx.power_model.add_npu_active_energy_consumption(
                ctx.hardware, ctx.node_id, lat, num_npus=ctx.tp_size)
            if get_device(ctx.placement, None, layer_name, "weights") != 'LOCAL':
                _, wt, _ = calculate_sizes(ctx.model, layer_name, bctx.total_len, fp=ctx.fp)
                ctx.power_model.add_dram_energy_consumption(ctx.node_id, wt)


def _synthesize_trace(hardware, model, config, tp_size, pp_size, local_ep, ep_total, pd_type, node_id, instance_id,
                      batch, max_len, output_path, placement, block_mode_on, gate,
                      enable_attn_offloading, power_model, pim_model, fp,
                      variant, kv_cache_dtype='auto',
                      runtime_max_num_batched_tokens=None, runtime_max_num_seqs=None,
                      runtime_block_size=None,
                      tp_dim=None, ep_dim=None, dp_sum_total_len=0,
                      memory_scenario_policy=None):
    ctx = _build_trace_ctx(hardware, model, config, tp_size, pp_size, local_ep, ep_total, node_id, fp,
                           placement, gate, enable_attn_offloading, power_model, pim_model, pd_type,
                           variant=variant, kv_cache_dtype=kv_cache_dtype,
                           runtime_max_num_batched_tokens=runtime_max_num_batched_tokens,
                           runtime_max_num_seqs=runtime_max_num_seqs,
                           runtime_block_size=runtime_block_size,
                           tp_dim=tp_dim, ep_dim=ep_dim, dp_sum_total_len=dp_sum_total_len,
                           memory_scenario_policy=memory_scenario_policy)
    block_mode_on = bool(
        block_mode_on
        or ctx.memory_scenario_policy.requires_per_block_trace
    )
    bctx = _build_batch_ctx(batch, ctx)

    logger.info(
        "Batch #%d: model=%s num_reqs=%d total_len=%d req_ids=%s",
        batch.batch_id, model, len(batch.requests), batch.total_len,
        [r.id for r in batch.requests],
        extra={"node_id": node_id, "instance_id": instance_id},
    )

    with open(output_path, 'w') as f:
        _emit_prologue(ctx, bctx, f)

        # Transformer blocks
        num_layers = config['num_hidden_layers']
        iter_count, copy_count = (num_layers, 1) if block_mode_on else (1, num_layers)

        for layer_num in range(iter_count):
            block_lines, block_power = _build_transformer_block(ctx, bctx, layer_num, 'NONE', str(batch.batch_id))

            # MoE blocks are only safely replayable when the router
            # opts into block copy (BALANCED is deterministic; others
            # carry tiny per-layer variance that block_copy swallows
            # for the sake of trace-generation speed).
            can_copy = (not ctx.is_moe or ctx.gate.block_copy) and not block_mode_on
            if can_copy:
                for _ in range(copy_count):
                    f.writelines(block_lines)
                    block_power.flush(ctx, enable_attn_offloading)
            else:
                f.writelines(block_lines)
                block_power.flush(ctx, enable_attn_offloading)

        # Final layers
        _emit_final_layers(ctx, bctx, f)
        _emit_pp_pd_power(ctx, bctx)


# ======================================================================
# _synthesize_interleaved_trace (two sub-batches)
# ======================================================================

def _synthesize_interleaved_trace(hardware, model, config, tp_size, pp_size, local_ep, ep_total, pd_type, node_id, instance_id,
                                  batches, max_len, output_path, placement, block_mode_on, gate,
                                  enable_attn_offloading, power_model, pim_model, fp,
                                  variant, kv_cache_dtype='auto',
                                  runtime_max_num_batched_tokens=None, runtime_max_num_seqs=None,
                                  runtime_block_size=None,
                                  tp_dim=None, ep_dim=None, dp_sum_total_len=0,
                                  memory_scenario_policy=None):
    ctx = _build_trace_ctx(hardware, model, config, tp_size, pp_size, local_ep, ep_total, node_id, fp,
                           placement, gate, enable_attn_offloading, power_model, pim_model, pd_type,
                           variant=variant, kv_cache_dtype=kv_cache_dtype,
                           runtime_max_num_batched_tokens=runtime_max_num_batched_tokens,
                           runtime_max_num_seqs=runtime_max_num_seqs,
                           runtime_block_size=runtime_block_size,
                           tp_dim=tp_dim, ep_dim=ep_dim, dp_sum_total_len=dp_sum_total_len,
                           memory_scenario_policy=memory_scenario_policy)
    block_mode_on = bool(
        block_mode_on
        or ctx.memory_scenario_policy.requires_per_block_trace
    )
    bctx1 = _build_batch_ctx(batches[0], ctx)
    bctx2 = _build_batch_ctx(batches[1], ctx)

    logger.info(
        "Sub-batch #%s: model=%s num_reqs=%d total_len=%d req_ids=%s",
        f"{batches[0].batch_id}.0", model, len(batches[0].requests), batches[0].total_len,
        [r.id for r in batches[0].requests],
        extra={"node_id": node_id, "instance_id": instance_id},
    )
    logger.info(
        "Sub-batch #%s: model=%s num_reqs=%d total_len=%d req_ids=%s",
        f"{batches[1].batch_id}.1", model, len(batches[1].requests), batches[1].total_len,
        [r.id for r in batches[1].requests],
        extra={"node_id": node_id, "instance_id": instance_id},
    )

    num_layers = config['num_hidden_layers']

    with open(output_path, 'w') as f:
        # PROLOGUE: Batch1 prologue + first pre-attn
        _emit_prologue(ctx, bctx1, f, 'BATCH_1')

        pre_attn1_lines = []
        pre_attn1_power = PowerAccumulator([], [], 0, 0)
        _emit_pre_attn_layers(ctx, bctx1, 0, pre_attn1_lines, pre_attn1_power, 'BATCH_1')
        f.writelines(pre_attn1_lines)
        pre_attn1_power.flush(ctx, enable_attn_offloading)

        # Batch2 prologue + first pre-attn
        _emit_prologue(ctx, bctx2, f, 'BATCH_2')

        pre_attn2_lines = []
        pre_attn2_power = PowerAccumulator([], [], 0, 0)
        _emit_pre_attn_layers(ctx, bctx2, 0, pre_attn2_lines, pre_attn2_power, 'BATCH_2')
        f.writelines(pre_attn2_lines)
        pre_attn2_power.flush(ctx, enable_attn_offloading)

        # MIDDLE LAYERS: interleaved post_attn + pre_attn
        middle_layers = num_layers - 1
        iter_count, copy_count = (middle_layers, 1) if block_mode_on else (1, middle_layers)

        for layer_num in range(iter_count):
            block_lines = []
            block_power = PowerAccumulator([], [], 0, 0)

            # Batch1: post_attn(current) + pre_attn(next)
            _emit_post_attn_layers(ctx, bctx1, layer_num, block_lines, block_power, f"{batches[0].batch_id}.0", 'BATCH_1')
            _emit_pre_attn_layers(ctx, bctx1, layer_num + 1, block_lines, block_power, 'BATCH_1')

            # Batch2: post_attn(current) + pre_attn(next)
            _emit_post_attn_layers(ctx, bctx2, layer_num, block_lines, block_power, f"{batches[1].batch_id}.1", 'BATCH_2')
            _emit_pre_attn_layers(ctx, bctx2, layer_num + 1, block_lines, block_power, 'BATCH_2')

            # MoE blocks are only safely replayable when the router
            # opts into block copy (BALANCED is deterministic; others
            # carry tiny per-layer variance that block_copy swallows
            # for the sake of trace-generation speed).
            can_copy = (not ctx.is_moe or ctx.gate.block_copy) and not block_mode_on
            if can_copy:
                for _ in range(copy_count):
                    f.writelines(block_lines)
                    block_power.flush(ctx, enable_attn_offloading)
            else:
                f.writelines(block_lines)
                block_power.flush(ctx, enable_attn_offloading)

        # EPILOGUE: last layer post_attn + final layers
        last_lines = []
        last_power = PowerAccumulator([], [], 0, 0)
        _emit_post_attn_layers(ctx, bctx1, num_layers - 1, last_lines, last_power, f"{batches[0].batch_id}.0", 'BATCH_1')
        f.writelines(last_lines)
        last_power.flush(ctx, enable_attn_offloading)
        _emit_final_layers(ctx, bctx1, f, 'BATCH_1')

        last_lines2 = []
        last_power2 = PowerAccumulator([], [], 0, 0)
        _emit_post_attn_layers(ctx, bctx2, num_layers - 1, last_lines2, last_power2, f"{batches[1].batch_id}.1", 'BATCH_2')
        f.writelines(last_lines2)
        last_power2.flush(ctx, enable_attn_offloading)
        _emit_final_layers(ctx, bctx2, f, 'BATCH_2')

        _emit_pp_pd_power(ctx, bctx1)


# ======================================================================
# generate_trace() — public entry point
# ======================================================================

def _effective_block_mode(block_mode_on, memory_scenario_policy):
    """block 覆盖存在时必须逐层构建，避免复用第 0 层的场景。"""
    return bool(
        block_mode_on
        or memory_scenario_policy.requires_per_block_trace
    )


# Wrapper function that creates trace for an instance
def generate_trace(batch, hardware, tp_size, pp_size, local_ep, ep_total, pd_type=None, node_id=0, instance_id=0,
                   max_num_batched_tokens=2048, max_num_seqs=None,
                   placement={}, block_mode_on=False, expert_routing_policy="BALANCED",
                   enable_prefix_caching=False, enable_attn_offloading=False, power_model=None, pim_model=None,
                   enable_sub_batch_interleaving=False, fp=16, dtype=None, kv_cache_dtype='auto',
                   tp_dim=None, ep_dim=None, dp_sum_total_len=0, enable_block_copy=True, inputs_root=None,
                   memory_scenario_policy=None, runtime_block_size=None):

    model = batch.model
    config = get_config(model)
    if memory_scenario_policy is None:
        memory_scenario_policy = parse_instance_performance_profile(
            {},
            config["num_hidden_layers"],
        )
    elif not isinstance(memory_scenario_policy, MemoryScenarioPolicy):
        raise TypeError("memory_scenario_policy 必须是 MemoryScenarioPolicy")
    block_mode_on = _effective_block_mode(
        block_mode_on,
        memory_scenario_policy,
    )
    fp = fp // 8  # bit -> byte of floating point
    max_len = min(max_num_batched_tokens, config['max_position_embeddings'])
    variant = resolve_variant(dtype, kv_cache_dtype, config)

    # vllm: add load or eviction in the txt file
    load_size = batch.load
    evict_size = batch.evict

    if inputs_root is None:
        inputs_root = os.path.join(os.getcwd(), "inputs")
    output_path = input_path(
        inputs_root, "trace", hardware, batch.model,
        f"instance{instance_id}_batch{batch.batch_id}.txt",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # make trace — accept either the Mistral-style ``num_local_experts``
    # key or the HF/Qwen3 ``num_experts`` key so both family's configs
    # resolve to a live GateRouter.
    num_experts_cfg = config.get("num_local_experts", config.get("num_experts"))
    if num_experts_cfg:
        gate = GateRouter(
            node_id, instance_id, num_experts_cfg,
            num_experts_per_tok=config.get('num_experts_per_tok', 1),
            routing_policy=expert_routing_policy,
            seed=42,
            block_copy=enable_block_copy,
        )
    else:
        gate = None

    # reset power model logs
    if power_model is not None:
        power_model.reset_log()

    # make trace
    synth_args = (hardware, model, config, tp_size, pp_size, local_ep, ep_total, pd_type, node_id, instance_id)
    # enable_prefix_caching is intentionally not forwarded: with chunked-prefill
    # semantics, the scheduler already encodes prefix hits via num_computed_tokens,
    # so trace synthesis no longer needs the flag.
    del enable_prefix_caching
    synth_kwargs = dict(placement=placement, block_mode_on=block_mode_on, gate=gate,
                        enable_attn_offloading=enable_attn_offloading,
                        power_model=power_model, pim_model=pim_model, fp=fp,
                        variant=variant, kv_cache_dtype=kv_cache_dtype,
                        runtime_max_num_batched_tokens=max_num_batched_tokens,
                        runtime_max_num_seqs=max_num_seqs,
                        runtime_block_size=runtime_block_size,
                        tp_dim=tp_dim, ep_dim=ep_dim, dp_sum_total_len=dp_sum_total_len,
                        memory_scenario_policy=memory_scenario_policy)
    if not enable_sub_batch_interleaving:
        _synthesize_trace(*synth_args, batch, max_len, output_path, **synth_kwargs)
    else:
        batches = _make_sub_batch(batch)
        if len(batches) < 2 or len(batches[0].requests) == 0 or len(batches[1].requests) == 0:
            _synthesize_trace(*synth_args, batch, max_len, output_path, **synth_kwargs)
        else:
            _synthesize_interleaved_trace(*synth_args, batches, max_len, output_path, **synth_kwargs)

    with open(output_path, 'r') as f:
        dic = []
        for line in f.readlines():
            split = re.findall(r'\S+', line)
            dic.append(split)

    # vllm: open output txt file and add load, evict mem
    mem = []
    if load_size != 0:
        load = ["kv_load", '0', 'LOCAL', '0', get_device(placement, None, None, 'kv_evict_loc'), str(load_size), 'LOCAL', '0', 'NONE', '0', 'NONE']
        mem.append(load)
        if power_model is not None:
            power_model.add_dram_energy_consumption(node_id, load_size)
    if evict_size != 0:
        evict = ["kv_evict", '0', 'LOCAL', '0', get_device(placement, None, None, 'kv_evict_loc'), str(evict_size), 'LOCAL', '0', 'NONE', '0', 'NONE']
        mem.append(evict)
        if power_model is not None:
            power_model.add_dram_energy_consumption(node_id, evict_size)

    if power_model is not None:
        power_model.print_log(node_id)

    result = mem + dic

    with open(output_path, 'w') as f:
        # instance type
        if pd_type == None:
            instance_type = 'COLOCATED'
        elif pd_type == 'prefill':
            instance_type = 'PREFILL'
        elif pd_type == 'decode':
            instance_type = 'DECODE'
        else:
            raise ValueError(f"Unknown instance type {pd_type}.")

        f.write(f"{instance_type}\t\tmodel_parallel_NPU_group: {pp_size}\n")
        f.write(str(len(result))+'\n')
        f.write(header())

        # add layer_number at the end of the layer_name
        for i in range(0, len(result)):
            if "EXPERT" not in result[i][0] and "PIM" not in result[i][0]:
                new_string = f'{result[i][0]}_{i}'
                f.write(formatter(new_string, *result[i][1:]))
            else:
                f.write(formatter(' '.join(result[i]),'','','','','','','','','',''))
    return


# ======================================================================
# generate_event() — preserved exactly
# ======================================================================


# generate event for first request arrival
def generate_event(alarm, inputs_root=None):

    # make inputs for text file
    result = []
    fp = 2
    layer_name = f'event_{alarm}ns'
    comp_time = alarm
    input_loc = 'REMOTE'
    input_size = 0
    weight_loc = 'LOCAL'
    weight_size = 0
    output_loc = 'REMOTE'
    output_size = 0
    comm_type = 'NONE'
    comm_size = 0
    misc = 'NONE'
    result.append([layer_name, comp_time, input_loc, input_size, weight_loc, weight_size, output_loc, output_size, comm_type, comm_size, misc])

    # write to the text file
    if inputs_root is None:
        inputs_root = os.path.join(os.getcwd(), "inputs")
    output_path = input_path(inputs_root, "trace", "event_handler.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(f"EVENT\n")
        f.write(f'{len(result)}'+'\n') # length of the text is 1
        f.write(header())
        for i in result:
            f.write(formatter(*i))


# ======================================================================
# Helper Functions for PIM Scheduling
# ======================================================================

# Greedy Min-Load Bin Packing Algorithm for PIM Attention Load Balancing
def _attn_load_balancer(requests, tp_size, pim_channels=0, channel_split=1):

    # Sort all requests by input length in descending order (longest first)
    requests = sorted(requests, key=lambda r: r.input, reverse=True)
    prefill_len = 0
    decode_lens = [[] for _ in range(pim_channels)]
    decode_loads = [0 for _ in range(pim_channels)]

    # Greedy load balancing with separate prefill / decode loads
    for req in requests:

        if req.is_init:
            # For prefill, just accumulate total length
            prefill_len += req.input
        else:
            # For decode with attn offloading, choose the PIM channel with the smallest decode load
            for channel in range(channel_split): # one channel can handle multiple heads if load is still small
                pim_id = min(range(pim_channels), key=lambda i: decode_loads[i])
                decode_lens[pim_id].append(req.input)
                decode_loads[pim_id] += req.input

    return prefill_len, decode_lens


# ======================================================================
# _make_sub_batch() — chunked-prefill aware sub-batch split
# ======================================================================

# spliting one batch into sub-batches to do sub-batch interleaving while using PIM
def _make_sub_batch(batch):
    if len(batch.requests) == 1:
        return [batch]

    # scheduler attaches per-request chunk sizes; honor them so chunked-prefill
    # later chunks (is_init=False but still is_prefill()) and prefix-cached
    # tokens are accounted for correctly (chunk_size already excludes hits via
    # num_computed_tokens).
    sched = getattr(batch, 'scheduled_tokens', {}) or {}

    def compute_tokens(req):
        if req.is_prefill():
            return sched.get(req.id, max(1, req.original_input - req.num_computed_tokens))
        return 1

    # Greedy split: longest per-iteration compute first, assign to lighter side.
    reqs = sorted(batch.requests, key=compute_tokens, reverse=True)
    req_groups = [[], []]
    loads = [0, 0]
    for req in reqs:
        target = 0 if loads[0] <= loads[1] else 1
        loads[target] += compute_tokens(req)
        req_groups[target].append(req)

    sub_batches = []
    for i, sub_reqs in enumerate(req_groups):
        sub_reqs.sort(key=lambda r: r.arrival)

        total_len = 0
        kv_len = 0
        num_prefill = 0
        num_decode = 0
        q_list = []
        k_list = []
        prefill_q_list = []
        prefill_k_list = []
        decode_k_list = []

        for req in sub_reqs:
            if req.is_prefill():
                chunk = compute_tokens(req)
                total_len += chunk
                q_list.append(chunk)
                prefill_q_list.append(chunk)
                # KV already in cache from prior chunks plus any prefix-cache hit.
                prefill_k_list.append(req.num_computed_tokens)
                num_prefill += 1
            else:
                total_len += 1
                q_list.append(1)
                kv_len += req.num_computed_tokens
                decode_k_list.append(req.num_computed_tokens)
                num_decode += 1
            k_list.append(req.num_computed_tokens)

        # evict/load are counted once for the original batch; attach to sub-batch 0 only.
        evict, load = (batch.evict, batch.load) if i == 0 else (0, 0)
        sub = Batch(
            batch.batch_id, batch.model,
            total_len, kv_len,
            q_list, k_list, num_prefill,
            num_decode, prefill_q_list,
            prefill_k_list, decode_k_list,
            0, 0, evict, load,
        )
        sub.requests.extend(sub_reqs)
        sub_batches.append(sub)

    return sub_batches
