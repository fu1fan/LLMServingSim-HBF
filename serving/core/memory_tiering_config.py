"""严格解析 HBF 分层策略配置。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .memory_tiering import MemoryTier


class MemoryTieringConfigError(ValueError):
    """HBF 容量、Profile 或策略配置不满足联合契约。"""


_WEIGHT_POLICIES = {
    "hbm_only",
    "hbf_only",
    "static_map",
    "hbf_backed_hbm_cache",
}
_KV_POLICIES = {
    "hbm_only",
    "hbf_only",
    "length_threshold",
    "watermark_lru",
}
_PREFIX_POLICIES = {
    "hbm_only",
    "hbf_only",
    "hbf_backed_hbm_hot",
    "instance_affinity",
}
_PREFETCH_POLICIES = {"none", "next_layer", "next_batch"}
_FALLBACK_POLICIES = {"reject", "hbf", "cpu", "cxl"}
_BLOCK_TOKEN = re.compile(r"^([0-9]+)(?:\s*-\s*([0-9]+))?$")


@dataclass(frozen=True)
class WeightTieringPolicy:
    policy: str
    default_tier: MemoryTier
    layer_tiers: Mapping[str, MemoryTier]
    block_tiers: Mapping[int, MemoryTier]
    hbm_high_watermark: float
    hbm_low_watermark: float

    def tier_for(self, layer_name: str, block_index: int | None) -> MemoryTier:
        if layer_name in self.layer_tiers:
            return self.layer_tiers[layer_name]
        if block_index is not None and block_index in self.block_tiers:
            return self.block_tiers[block_index]
        return self.default_tier


@dataclass(frozen=True)
class KVTieringPolicy:
    policy: str
    admission_tier: MemoryTier
    threshold_tokens: int | None
    hbm_high_watermark: float
    hbm_low_watermark: float


@dataclass(frozen=True)
class PrefixTieringPolicy:
    policy: str
    promotion_hits: int
    hbm_high_watermark: float
    hbm_low_watermark: float


@dataclass(frozen=True)
class TransferPolicy:
    prefetch: str
    prefetch_distance: int
    capacity_fallback: str


@dataclass(frozen=True)
class CommunicationBufferPolicy:
    tier: MemoryTier
    allow_hbf_staging: bool


@dataclass(frozen=True)
class MemoryTieringConfig:
    """一个 instance 完整、不可变的 HBF 策略集合。"""

    enabled: bool
    hbf_capacity_bytes: int
    weights: WeightTieringPolicy
    kv: KVTieringPolicy
    prefix: PrefixTieringPolicy
    transfer: TransferPolicy
    communication_buffers: CommunicationBufferPolicy


def _reject_unknown_fields(
    value: Mapping,
    allowed: set[str],
    *,
    field: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        rendered = ", ".join(repr(item) for item in unknown)
        raise MemoryTieringConfigError(f"{field} 包含未知字段：{rendered}")


def _mapping(value, field):
    if not isinstance(value, Mapping):
        raise MemoryTieringConfigError(f"{field} 必须是 mapping")
    return value


def _policy(value, allowed, field):
    if not isinstance(value, str) or value not in allowed:
        raise MemoryTieringConfigError(
            f"{field} 必须是以下值之一：{sorted(allowed)}"
        )
    return value


def _tier(value, field, *, hbf_enabled):
    if not isinstance(value, str):
        raise MemoryTieringConfigError(f"{field} 必须是字符串")
    try:
        tier = MemoryTier(value.lower())
    except ValueError as exc:
        raise MemoryTieringConfigError(
            f"{field} 必须是 hbm、hbf、cpu 或 cxl"
        ) from exc
    if tier is MemoryTier.HBF and not hbf_enabled:
        raise MemoryTieringConfigError(f"{field} 使用 HBF，但 instance 未配置 hbf_mem")
    return tier


def _watermarks(value, field):
    high = value.get("hbm_high_watermark", 0.90)
    low = value.get("hbm_low_watermark", 0.75)
    for name, number in (
        ("hbm_high_watermark", high),
        ("hbm_low_watermark", low),
    ):
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not 0 < float(number) <= 1
        ):
            raise MemoryTieringConfigError(
                f"{field}.{name} 必须在 (0, 1] 范围内"
            )
    if float(low) >= float(high):
        raise MemoryTieringConfigError(
            f"{field}.hbm_low_watermark 必须小于 hbm_high_watermark"
        )
    return float(high), float(low)


def _parse_blocks(expression, num_hidden_layers, field):
    if not isinstance(expression, str) or not expression.strip():
        raise MemoryTieringConfigError(f"{field} 必须是非空 blocks 表达式")
    result = set()
    for raw in expression.split(","):
        token = raw.strip()
        match = _BLOCK_TOKEN.fullmatch(token)
        if match is None:
            raise MemoryTieringConfigError(f"{field} 包含无效分段 {token!r}")
        first = int(match.group(1))
        last = int(match.group(2) or first)
        lo, hi = sorted((first, last))
        if lo < 0 or hi >= num_hidden_layers:
            raise MemoryTieringConfigError(
                f"{field} 的分段 {token!r} 越界；"
                f"有效范围是 0-{num_hidden_layers - 1}"
            )
        expanded = set(range(lo, hi + 1))
        duplicate = sorted(result & expanded)
        if duplicate:
            raise MemoryTieringConfigError(f"{field} 重复选择 block：{duplicate}")
        result.update(expanded)
    return tuple(sorted(result))


def _parse_weights(raw, *, hbf_enabled, num_hidden_layers):
    value = _mapping(raw, "memory_tiering.weights")
    _reject_unknown_fields(
        value,
        {
            "policy",
            "default_tier",
            "layers",
            "blocks",
            "hbm_high_watermark",
            "hbm_low_watermark",
        },
        field="memory_tiering.weights",
    )
    policy = _policy(
        value.get("policy", "hbm_only"),
        _WEIGHT_POLICIES,
        "memory_tiering.weights.policy",
    )
    implicit_tier = MemoryTier.HBF if policy in {"hbf_only", "hbf_backed_hbm_cache"} else MemoryTier.HBM
    default_tier = _tier(
        value.get("default_tier", implicit_tier.value),
        "memory_tiering.weights.default_tier",
        hbf_enabled=hbf_enabled,
    )
    if policy == "hbm_only" and default_tier is not MemoryTier.HBM:
        raise MemoryTieringConfigError("hbm_only 权重策略必须使用 HBM")
    if policy == "hbf_only" and default_tier is not MemoryTier.HBF:
        raise MemoryTieringConfigError("hbf_only 权重策略必须使用 HBF")

    layers_raw = _mapping(value.get("layers", {}), "memory_tiering.weights.layers")
    layer_tiers = {}
    for layer_name, tier_name in layers_raw.items():
        if not isinstance(layer_name, str) or not layer_name:
            raise MemoryTieringConfigError("weights.layers 的键必须是非空层名")
        layer_tiers[layer_name] = _tier(
            tier_name,
            f"memory_tiering.weights.layers.{layer_name}",
            hbf_enabled=hbf_enabled,
        )

    blocks_raw = value.get("blocks", [])
    if not isinstance(blocks_raw, list):
        raise MemoryTieringConfigError("memory_tiering.weights.blocks 必须是列表")
    block_tiers = {}
    for index, rule in enumerate(blocks_raw):
        rule = _mapping(rule, f"memory_tiering.weights.blocks[{index}]")
        _reject_unknown_fields(
            rule,
            {"blocks", "tier"},
            field=f"memory_tiering.weights.blocks[{index}]",
        )
        if "blocks" not in rule or "tier" not in rule:
            raise MemoryTieringConfigError(
                f"memory_tiering.weights.blocks[{index}] 必须包含 blocks 和 tier"
            )
        tier = _tier(
            rule["tier"],
            f"memory_tiering.weights.blocks[{index}].tier",
            hbf_enabled=hbf_enabled,
        )
        for block_id in _parse_blocks(
            rule["blocks"],
            num_hidden_layers,
            f"memory_tiering.weights.blocks[{index}].blocks",
        ):
            if block_id in block_tiers:
                raise MemoryTieringConfigError(
                    f"memory_tiering.weights.blocks 重复选择 block {block_id}"
                )
            block_tiers[block_id] = tier

    if policy != "static_map" and (layer_tiers or block_tiers):
        raise MemoryTieringConfigError(
            "只有 static_map 权重策略可以配置 layers 或 blocks"
        )
    high, low = _watermarks(value, "memory_tiering.weights")
    return WeightTieringPolicy(
        policy=policy,
        default_tier=default_tier,
        layer_tiers=MappingProxyType(layer_tiers),
        block_tiers=MappingProxyType(block_tiers),
        hbm_high_watermark=high,
        hbm_low_watermark=low,
    )


def _parse_kv(raw, *, hbf_enabled):
    value = _mapping(raw, "memory_tiering.kv")
    _reject_unknown_fields(
        value,
        {
            "policy",
            "admission_tier",
            "threshold_tokens",
            "hbm_high_watermark",
            "hbm_low_watermark",
        },
        field="memory_tiering.kv",
    )
    policy = _policy(
        value.get("policy", "hbm_only"),
        _KV_POLICIES,
        "memory_tiering.kv.policy",
    )
    implicit_tier = MemoryTier.HBF if policy == "hbf_only" else MemoryTier.HBM
    admission_tier = _tier(
        value.get("admission_tier", implicit_tier.value),
        "memory_tiering.kv.admission_tier",
        hbf_enabled=hbf_enabled,
    )
    if policy == "hbm_only" and admission_tier is not MemoryTier.HBM:
        raise MemoryTieringConfigError("hbm_only KV 策略必须从 HBM 准入")
    if policy == "hbf_only" and admission_tier is not MemoryTier.HBF:
        raise MemoryTieringConfigError("hbf_only KV 策略必须从 HBF 准入")

    threshold = value.get("threshold_tokens")
    if policy == "length_threshold":
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, int)
            or threshold <= 0
        ):
            raise MemoryTieringConfigError(
                "length_threshold KV 策略必须配置正整数 threshold_tokens"
            )
    elif threshold is not None:
        raise MemoryTieringConfigError(
            "只有 length_threshold KV 策略可以配置 threshold_tokens"
        )
    high, low = _watermarks(value, "memory_tiering.kv")
    return KVTieringPolicy(
        policy=policy,
        admission_tier=admission_tier,
        threshold_tokens=threshold,
        hbm_high_watermark=high,
        hbm_low_watermark=low,
    )


def _parse_prefix(raw, *, hbf_enabled):
    value = _mapping(raw, "memory_tiering.prefix")
    _reject_unknown_fields(
        value,
        {
            "policy",
            "promotion_hits",
            "hbm_high_watermark",
            "hbm_low_watermark",
        },
        field="memory_tiering.prefix",
    )
    policy = _policy(
        value.get("policy", "hbm_only"),
        _PREFIX_POLICIES,
        "memory_tiering.prefix.policy",
    )
    if policy != "hbm_only" and not hbf_enabled:
        raise MemoryTieringConfigError(
            f"{policy} Prefix 策略要求配置 hbf_mem"
        )
    promotion_hits = value.get("promotion_hits", 2)
    if (
        isinstance(promotion_hits, bool)
        or not isinstance(promotion_hits, int)
        or promotion_hits <= 0
    ):
        raise MemoryTieringConfigError("prefix.promotion_hits 必须是正整数")
    high, low = _watermarks(value, "memory_tiering.prefix")
    return PrefixTieringPolicy(
        policy=policy,
        promotion_hits=promotion_hits,
        hbm_high_watermark=high,
        hbm_low_watermark=low,
    )


def _parse_transfer(raw, *, hbf_enabled):
    value = _mapping(raw, "memory_tiering.transfer")
    _reject_unknown_fields(
        value,
        {"prefetch", "prefetch_distance", "capacity_fallback"},
        field="memory_tiering.transfer",
    )
    prefetch = _policy(
        value.get("prefetch", "none"),
        _PREFETCH_POLICIES,
        "memory_tiering.transfer.prefetch",
    )
    distance = value.get("prefetch_distance", 1)
    if isinstance(distance, bool) or not isinstance(distance, int) or distance <= 0:
        raise MemoryTieringConfigError(
            "memory_tiering.transfer.prefetch_distance 必须是正整数"
        )
    fallback = _policy(
        value.get("capacity_fallback", "reject"),
        _FALLBACK_POLICIES,
        "memory_tiering.transfer.capacity_fallback",
    )
    if fallback == "hbf" and not hbf_enabled:
        raise MemoryTieringConfigError("capacity_fallback=hbf 要求配置 hbf_mem")
    return TransferPolicy(
        prefetch=prefetch,
        prefetch_distance=distance,
        capacity_fallback=fallback,
    )


def _parse_communication(raw, *, hbf_enabled):
    value = _mapping(raw, "memory_tiering.communication_buffers")
    _reject_unknown_fields(
        value,
        {"tier", "allow_hbf_staging"},
        field="memory_tiering.communication_buffers",
    )
    tier = _tier(
        value.get("tier", "hbm"),
        "memory_tiering.communication_buffers.tier",
        hbf_enabled=hbf_enabled,
    )
    allow_hbf = value.get("allow_hbf_staging", False)
    if not isinstance(allow_hbf, bool):
        raise MemoryTieringConfigError(
            "communication_buffers.allow_hbf_staging 必须是布尔值"
        )
    if tier is MemoryTier.HBF and not allow_hbf:
        raise MemoryTieringConfigError(
            "communication_buffers.tier=hbf 必须显式开启 allow_hbf_staging"
        )
    return CommunicationBufferPolicy(
        tier=tier,
        allow_hbf_staging=allow_hbf,
    )


def _disabled_config():
    high, low = 0.90, 0.75
    return MemoryTieringConfig(
        enabled=False,
        hbf_capacity_bytes=0,
        weights=WeightTieringPolicy(
            "hbm_only",
            MemoryTier.HBM,
            MappingProxyType({}),
            MappingProxyType({}),
            high,
            low,
        ),
        kv=KVTieringPolicy("hbm_only", MemoryTier.HBM, None, high, low),
        prefix=PrefixTieringPolicy("hbm_only", 2, high, low),
        transfer=TransferPolicy("none", 1, "reject"),
        communication_buffers=CommunicationBufferPolicy(
            MemoryTier.HBM,
            False,
        ),
    )


def parse_instance_memory_tiering(instance, num_hidden_layers):
    """解析 instance 的 HBF 容量、Profile 选择和分层策略。"""

    if not isinstance(instance, Mapping):
        raise MemoryTieringConfigError("instance 必须是 mapping")
    if (
        isinstance(num_hidden_layers, bool)
        or not isinstance(num_hidden_layers, int)
        or num_hidden_layers <= 0
    ):
        raise MemoryTieringConfigError("num_hidden_layers 必须是正整数")

    hbf_raw = instance.get("hbf_mem")
    tiering_raw = instance.get("memory_tiering")
    if hbf_raw is None:
        if tiering_raw is not None:
            raise MemoryTieringConfigError(
                "memory_tiering 要求同一 instance 配置 hbf_mem"
            )
        return _disabled_config()

    hbf = _mapping(hbf_raw, "instance.hbf_mem")
    _reject_unknown_fields(hbf, {"mem_size"}, field="instance.hbf_mem")
    mem_size = hbf.get("mem_size")
    if (
        isinstance(mem_size, bool)
        or not isinstance(mem_size, (int, float))
        or mem_size <= 0
    ):
        raise MemoryTieringConfigError("instance.hbf_mem.mem_size 必须是正数")
    hbf_capacity_bytes = int(float(mem_size) * 1024 * 1024 * 1024)

    profile = _mapping(
        instance.get("performance_profile"),
        "instance.performance_profile",
    )
    if profile.get("mode") != "memory_scenario_v2":
        raise MemoryTieringConfigError(
            "HBF instance 必须使用 memory_scenario_v2 Profile"
        )
    if profile.get("scenario_selection") != "residency_derived":
        raise MemoryTieringConfigError(
            "HBF instance 必须设置 scenario_selection=residency_derived"
        )
    if "scenario_policy" in profile:
        raise MemoryTieringConfigError(
            "residency_derived 与调用方 scenario_policy 互斥"
        )

    value = _mapping(tiering_raw or {}, "instance.memory_tiering")
    _reject_unknown_fields(
        value,
        {
            "weights",
            "kv",
            "prefix",
            "transfer",
            "communication_buffers",
        },
        field="instance.memory_tiering",
    )
    return MemoryTieringConfig(
        enabled=True,
        hbf_capacity_bytes=hbf_capacity_bytes,
        weights=_parse_weights(
            value.get("weights", {}),
            hbf_enabled=True,
            num_hidden_layers=num_hidden_layers,
        ),
        kv=_parse_kv(value.get("kv", {}), hbf_enabled=True),
        prefix=_parse_prefix(value.get("prefix", {}), hbf_enabled=True),
        transfer=_parse_transfer(value.get("transfer", {}), hbf_enabled=True),
        communication_buffers=_parse_communication(
            value.get("communication_buffers", {}),
            hbf_enabled=True,
        ),
    )


def validate_homogeneous_hbf_instances(instances) -> bool:
    """拒绝同一次仿真混用普通 GPU 与 HBF GPU instance。"""

    instances = tuple(instances)
    if not instances:
        raise MemoryTieringConfigError("cluster 至少需要一个 instance")
    enabled = ["hbf_mem" in instance for instance in instances]
    if any(enabled) and not all(enabled):
        raise MemoryTieringConfigError(
            "同一次仿真不能混用普通 GPU 与 HBF GPU instance"
        )
    return all(enabled)

