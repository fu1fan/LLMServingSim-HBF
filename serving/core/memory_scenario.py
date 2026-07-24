"""静态 memory scenario 策略解析与兼容门禁。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .profile_contract import (
    ProfileContractError,
    validate_memory_profile_id,
)


LEGACY_HBM_ONLY = "legacy_hbm_only"
MEMORY_SCENARIO_V2 = "memory_scenario_v2"

_PROFILE_FIELDS = {"mode", "memory_profile_id", "scenario_policy"}
_POLICY_FIELDS = {"default", "layers", "blocks"}
_BLOCK_RULE_FIELDS = {"blocks", "scenario"}
_CANONICAL_LAYER_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_BLOCK_TOKEN_RE = re.compile(r"^([0-9]+)(?:\s*-\s*([0-9]+))?$")


class MemoryScenarioConfigError(ValueError):
    """静态 scenario 配置不合法。"""


class MemoryScenarioCompatibilityError(RuntimeError):
    """v2 Profile 与旧运行路径的组合不兼容。"""


@dataclass(frozen=True)
class MemoryScenarioPolicy:
    """一个 instance 的静态 Profile 选择与场景覆盖。"""

    mode: str
    memory_profile_id: str | None
    default_scenario: str | None
    layer_scenarios: Mapping[str, str]
    block_scenarios: Mapping[int, str]
    num_hidden_layers: int

    @property
    def is_v2(self) -> bool:
        return self.mode == MEMORY_SCENARIO_V2

    @property
    def requires_per_block_trace(self) -> bool:
        return self.is_v2 and bool(self.block_scenarios)

    def scenario_for(
        self,
        layer_name: str,
        block_index: int | None,
    ) -> str | None:
        """按 layer > block > default 的优先级返回离散场景。"""

        if not self.is_v2:
            return None
        _validate_layer_name(layer_name, field="layer_name")
        if block_index is not None:
            if isinstance(block_index, bool) or not isinstance(block_index, int):
                raise MemoryScenarioConfigError(
                    "block_index 必须是整数或 None"
                )
            if not 0 <= block_index < self.num_hidden_layers:
                raise MemoryScenarioConfigError(
                    f"block_index={block_index} 越界；有效范围为 "
                    f"0-{self.num_hidden_layers - 1}"
                )

        if layer_name in self.layer_scenarios:
            return self.layer_scenarios[layer_name]
        if block_index is not None and block_index in self.block_scenarios:
            return self.block_scenarios[block_index]
        return self.default_scenario


def _reject_unknown_fields(
    value: Mapping,
    allowed: set[str],
    *,
    field: str,
) -> None:
    unknown = [key for key in value if key not in allowed]
    if unknown:
        rendered = ", ".join(sorted((repr(key) for key in unknown)))
        raise MemoryScenarioConfigError(
            f"{field} 包含未知字段：{rendered}"
        )


def _validate_num_hidden_layers(num_hidden_layers: object) -> int:
    if (
        isinstance(num_hidden_layers, bool)
        or not isinstance(num_hidden_layers, int)
        or num_hidden_layers <= 0
    ):
        raise MemoryScenarioConfigError("num_hidden_layers 必须是正整数")
    return num_hidden_layers


def _validate_identifier(value: object, *, field: str) -> str:
    try:
        return validate_memory_profile_id(value, field=field)
    except ProfileContractError as exc:
        raise MemoryScenarioConfigError(str(exc)) from exc


def _validate_layer_name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _CANONICAL_LAYER_RE.fullmatch(value):
        raise MemoryScenarioConfigError(
            f"{field}={value!r} 必须是精确 canonical layer 名称"
        )
    return value


def _parse_blocks_expression(
    expression: object,
    *,
    num_hidden_layers: int,
    field: str,
) -> tuple[int, ...]:
    if not isinstance(expression, str) or not expression.strip():
        raise MemoryScenarioConfigError(
            f"{field} 必须是非空 blocks 表达式"
        )

    result: set[int] = set()
    parts = expression.split(",")
    if any(not part.strip() for part in parts):
        raise MemoryScenarioConfigError(f"{field} 包含空的逗号分段")

    for part in parts:
        token = part.strip()
        match = _BLOCK_TOKEN_RE.fullmatch(token)
        if match is None:
            raise MemoryScenarioConfigError(
                f"{field} 包含无效分段 {token!r}"
            )
        first = int(match.group(1))
        second = int(match.group(2)) if match.group(2) is not None else first
        lo, hi = sorted((first, second))
        if lo < 0 or hi >= num_hidden_layers:
            raise MemoryScenarioConfigError(
                f"{field} 的分段 {token!r} 越界；有效范围为 "
                f"0-{num_hidden_layers - 1}"
            )
        expanded = set(range(lo, hi + 1))
        duplicate = sorted(result & expanded)
        if duplicate:
            raise MemoryScenarioConfigError(
                f"{field} 重复选择 block：{duplicate}"
            )
        result.update(expanded)
    return tuple(sorted(result))


def _legacy_policy(num_hidden_layers: int) -> MemoryScenarioPolicy:
    return MemoryScenarioPolicy(
        mode=LEGACY_HBM_ONLY,
        memory_profile_id=None,
        default_scenario=None,
        layer_scenarios=MappingProxyType({}),
        block_scenarios=MappingProxyType({}),
        num_hidden_layers=num_hidden_layers,
    )


def parse_instance_performance_profile(
    instance: Mapping,
    num_hidden_layers: int,
) -> MemoryScenarioPolicy:
    """解析 instance.performance_profile，不读取 Profile bundle。"""

    num_layers = _validate_num_hidden_layers(num_hidden_layers)
    if not isinstance(instance, Mapping):
        raise MemoryScenarioConfigError("instance 必须是 mapping")
    if "performance_profile" not in instance:
        return _legacy_policy(num_layers)

    raw = instance["performance_profile"]
    if not isinstance(raw, Mapping):
        raise MemoryScenarioConfigError(
            "instance.performance_profile 必须是 mapping"
        )
    mode = raw.get("mode")
    if mode == LEGACY_HBM_ONLY:
        _reject_unknown_fields(
            raw,
            {"mode"},
            field="instance.performance_profile",
        )
        return _legacy_policy(num_layers)
    if mode != MEMORY_SCENARIO_V2:
        raise MemoryScenarioConfigError(
            "instance.performance_profile.mode 必须是 "
            f"{LEGACY_HBM_ONLY!r} 或 {MEMORY_SCENARIO_V2!r}"
        )

    _reject_unknown_fields(
        raw,
        _PROFILE_FIELDS,
        field="instance.performance_profile",
    )
    profile_id = _validate_identifier(
        raw.get("memory_profile_id"),
        field="instance.performance_profile.memory_profile_id",
    )
    policy_raw = raw.get("scenario_policy")
    if not isinstance(policy_raw, Mapping):
        raise MemoryScenarioConfigError(
            "instance.performance_profile.scenario_policy 必须是 mapping"
        )
    _reject_unknown_fields(
        policy_raw,
        _POLICY_FIELDS,
        field="instance.performance_profile.scenario_policy",
    )
    if "default" not in policy_raw:
        raise MemoryScenarioConfigError(
            "instance.performance_profile.scenario_policy.default 为必填项"
        )
    default_scenario = _validate_identifier(
        policy_raw["default"],
        field="scenario_policy.default",
    )

    layers_raw = policy_raw.get("layers", {})
    if not isinstance(layers_raw, Mapping):
        raise MemoryScenarioConfigError("scenario_policy.layers 必须是 mapping")
    layer_scenarios: dict[str, str] = {}
    for layer_name, scenario in layers_raw.items():
        layer = _validate_layer_name(
            layer_name,
            field="scenario_policy.layers key",
        )
        layer_scenarios[layer] = _validate_identifier(
            scenario,
            field=f"scenario_policy.layers.{layer}",
        )

    blocks_raw = policy_raw.get("blocks", [])
    if not isinstance(blocks_raw, list):
        raise MemoryScenarioConfigError("scenario_policy.blocks 必须是列表")
    block_scenarios: dict[int, str] = {}
    block_owners: dict[int, int] = {}
    for rule_index, rule in enumerate(blocks_raw):
        field = f"scenario_policy.blocks[{rule_index}]"
        if not isinstance(rule, Mapping):
            raise MemoryScenarioConfigError(f"{field} 必须是 mapping")
        _reject_unknown_fields(rule, _BLOCK_RULE_FIELDS, field=field)
        if "blocks" not in rule or "scenario" not in rule:
            raise MemoryScenarioConfigError(
                f"{field} 必须同时包含 blocks 和 scenario"
            )
        scenario = _validate_identifier(
            rule["scenario"],
            field=f"{field}.scenario",
        )
        block_ids = _parse_blocks_expression(
            rule["blocks"],
            num_hidden_layers=num_layers,
            field=f"{field}.blocks",
        )
        for block_id in block_ids:
            if block_id in block_scenarios:
                raise MemoryScenarioConfigError(
                    f"{field} 与 scenario_policy.blocks"
                    f"[{block_owners[block_id]}] 重叠选择 block {block_id}"
                )
            block_scenarios[block_id] = scenario
            block_owners[block_id] = rule_index

    return MemoryScenarioPolicy(
        mode=MEMORY_SCENARIO_V2,
        memory_profile_id=profile_id,
        default_scenario=default_scenario,
        layer_scenarios=MappingProxyType(layer_scenarios),
        block_scenarios=MappingProxyType(block_scenarios),
        num_hidden_layers=num_layers,
    )


def _is_local_location(value: object) -> bool:
    if not isinstance(value, str):
        return False
    prefix = value.strip().upper().split(":", 1)[0]
    return prefix == "LOCAL"


def _validate_placement_entry(
    entry: Mapping,
    default_entry: Mapping,
    *,
    field: str,
) -> None:
    if not isinstance(entry, Mapping):
        raise MemoryScenarioCompatibilityError(f"{field} 必须是 mapping")
    for kind in ("weights", "kv_loc"):
        location = entry.get(kind, default_entry.get(kind))
        if not _is_local_location(location):
            raise MemoryScenarioCompatibilityError(
                f"Profile v2 要求 {field}.{kind} 为 LOCAL，实际为 "
                f"{location!r}"
            )


def validate_memory_scenario_compatibility(
    policy: MemoryScenarioPolicy,
    *,
    enable_local_offloading: bool,
    enable_attn_offloading: bool,
    enable_sub_batch_interleaving: bool,
    placement: Mapping,
) -> None:
    """对最终 runtime flags 与旧 placement 执行 v2 fail-closed 门禁。"""

    if not isinstance(policy, MemoryScenarioPolicy):
        raise TypeError("policy 必须是 MemoryScenarioPolicy")
    if not policy.is_v2:
        return

    flags = {
        "enable_local_offloading": enable_local_offloading,
        "enable_attn_offloading": enable_attn_offloading,
        "enable_sub_batch_interleaving": enable_sub_batch_interleaving,
    }
    for name, enabled in flags.items():
        if not isinstance(enabled, bool):
            raise MemoryScenarioCompatibilityError(
                f"{name} 必须是最终解析后的布尔值"
            )
    enabled_flags = sorted(name for name, enabled in flags.items() if enabled)
    if enabled_flags:
        raise MemoryScenarioCompatibilityError(
            "Profile v2 不兼容以下旧运行路径："
            + ", ".join(enabled_flags)
        )

    if not isinstance(placement, Mapping):
        raise MemoryScenarioCompatibilityError("placement 必须是 normalized mapping")
    default_entry = placement.get("default")
    if not isinstance(default_entry, Mapping):
        raise MemoryScenarioCompatibilityError(
            "placement.default 必须是 normalized mapping"
        )
    _validate_placement_entry(
        default_entry,
        default_entry,
        field="placement.default",
    )

    blocks = placement.get("block", [])
    if not isinstance(blocks, list):
        raise MemoryScenarioCompatibilityError("placement.block 必须是列表")
    for block_index, entry in enumerate(blocks):
        _validate_placement_entry(
            entry,
            default_entry,
            field=f"placement.block[{block_index}]",
        )

    layers = placement.get("layer", {})
    if not isinstance(layers, Mapping):
        raise MemoryScenarioCompatibilityError("placement.layer 必须是 mapping")
    for layer_name, entry in layers.items():
        _validate_placement_entry(
            entry,
            default_entry,
            field=f"placement.layer[{layer_name!r}]",
        )
