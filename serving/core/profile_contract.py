"""Serving profile bundle 的版本化身份与静态契约校验。"""

from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

import yaml


PROFILE_SCHEMA_V1 = 1
PROFILE_SCHEMA_V2 = 2

V2_AUDIT_COLUMNS = (
    "hbm_read_bytes",
    "hbm_write_bytes",
    "hbf_read_bytes",
    "hbf_write_bytes",
)

_CATEGORY_SHAPE_COLUMNS = {
    "dense.csv": ("layer", "tokens"),
    "per_sequence.csv": ("layer", "sequences"),
    "attention.csv": (
        "prefill_chunk",
        "kv_prefill",
        "n_decode",
        "kv_decode",
    ),
    "moe.csv": ("tokens", "activated_experts"),
}

_POSITIVE_SHAPE_COLUMNS = {
    "tokens",
    "sequences",
    "activated_experts",
}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_REQUIRED_ACCOUNTING_INCLUDES = {
    "compute",
    "hbm_demand_access",
    "hbf_demand_access",
}

_REQUIRED_ACCOUNTING_EXCLUDES = {
    "migration",
    "prefetch",
    "eviction",
    "network_collective",
}

_SCENARIO_MAPPING_KEYS = ("accesses", "access_tiers", "placements")

PROFILE_V2_PARTIAL = "partial"
PROFILE_V2_RUNTIME_READY = "runtime_ready"

_SCENARIO_BINDING_BY_READINESS = {
    PROFILE_V2_PARTIAL: "caller_asserted",
    PROFILE_V2_RUNTIME_READY: "producer_verified_v1",
}

_ARCHITECTURE_REQUIREMENT_FIELDS = {
    "model_type",
    "tp_degrees",
    "scenario_ids",
    "dense_layers",
    "per_sequence_layers",
    "attention_required",
    "moe_required",
}


class ProfileContractError(ValueError):
    """Profile bundle 不满足静态契约。"""


class ProfileV2RuntimeNotReadyError(RuntimeError):
    """Profile v2 已通过静态校验，但不满足运行时完整性门禁。"""


@dataclass(frozen=True)
class ProfileContract:
    """已校验的 bundle 身份和 manifest。"""

    schema_version: int
    memory_profile_id: str | None
    scenario_catalog: Mapping[str, Mapping[str, Any]]
    latency_accounting: Mapping[str, Any]
    bundle_readiness: str | None
    runtime_compatible: bool | None
    scenario_binding: str | None
    tp_degrees: tuple[int, ...]
    architecture_requirements: Mapping[str, Any] | None
    meta: Mapping[str, Any]

    @property
    def is_v2(self) -> bool:
        return self.schema_version == PROFILE_SCHEMA_V2


def validate_memory_profile_id(memory_profile_id: object, *, field: str) -> str:
    """校验可安全作为单层目录名的稳定标识。"""

    if not isinstance(memory_profile_id, str) or not memory_profile_id:
        raise ProfileContractError(f"{field} 必须是非空字符串")
    if (
        not _IDENTIFIER_RE.fullmatch(memory_profile_id)
        or ".." in memory_profile_id
        or "/" in memory_profile_id
        or "\\" in memory_profile_id
    ):
        raise ProfileContractError(
            f"{field}={memory_profile_id!r} 不是合法标识；"
            "仅允许字母、数字、点、下划线和连字符，且不得包含 '..'"
        )
    return memory_profile_id


def _read_meta(profile_root: str) -> Mapping[str, Any]:
    path = os.path.join(profile_root, "meta.yaml")
    if not os.path.isfile(path):
        raise ProfileContractError(f"Profile manifest 不存在：{path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ProfileContractError(f"无法解析 Profile manifest {path}：{exc}") from exc
    except OSError as exc:
        raise ProfileContractError(f"无法读取 Profile manifest {path}：{exc}") from exc
    if not isinstance(meta, dict):
        raise ProfileContractError(f"Profile manifest {path} 的顶层必须是 mapping")
    return meta


def _schema_version(meta: Mapping[str, Any], source: str) -> int:
    if "profile_schema_version" not in meta:
        return PROFILE_SCHEMA_V1
    version = meta["profile_schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ProfileContractError(
            f"{source}: profile_schema_version 必须是整数 1 或 2"
        )
    if version not in (PROFILE_SCHEMA_V1, PROFILE_SCHEMA_V2):
        raise ProfileContractError(
            f"{source}: 不支持 profile_schema_version={version!r}；仅支持 1 或 2"
        )
    return version


def _string_set(value: object, *, field: str) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
        raise ProfileContractError(f"{field} 必须是非空字符串列表")
    if len(value) != len(set(value)):
        raise ProfileContractError(f"{field} 不得包含重复项")
    return set(value)


def _strict_string_list(
    value: object,
    *,
    field: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, list) or (
        not allow_empty and not value
    ):
        suffix = "" if allow_empty else "非空"
        raise ProfileContractError(f"{field} 必须是{suffix}字符串列表")
    result = []
    for item in value:
        if not isinstance(item, str) or not item or item != item.strip():
            raise ProfileContractError(
                f"{field} 的每一项必须是无首尾空白的非空字符串"
            )
        result.append(item)
    if len(result) != len(set(result)):
        raise ProfileContractError(f"{field} 不得包含重复项")
    return tuple(result)


def _positive_integer_list(
    value: object,
    *,
    field: str,
) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ProfileContractError(f"{field} 必须是非空正整数列表")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ProfileContractError(f"{field} 必须是非空正整数列表")
        result.append(item)
    if len(result) != len(set(result)):
        raise ProfileContractError(f"{field} 不得包含重复项")
    return tuple(result)


def _validate_architecture_requirements(
    value: object,
    *,
    source: str,
    tp_degrees: tuple[int, ...],
    scenario_ids: set[str],
) -> Mapping[str, Any]:
    field = f"{source}: architecture_requirements"
    if not isinstance(value, dict):
        raise ProfileContractError(f"{field} 必须是 mapping")
    actual_fields = set(value)
    if actual_fields != _ARCHITECTURE_REQUIREMENT_FIELDS:
        missing = sorted(_ARCHITECTURE_REQUIREMENT_FIELDS - actual_fields)
        extra = sorted(actual_fields - _ARCHITECTURE_REQUIREMENT_FIELDS)
        details = []
        if missing:
            details.append(f"缺少 {', '.join(missing)}")
        if extra:
            details.append(f"包含未知字段 {', '.join(extra)}")
        raise ProfileContractError(f"{field} 字段不完整：{'; '.join(details)}")

    model_type = value["model_type"]
    if (
        not isinstance(model_type, str)
        or not model_type
        or model_type != model_type.strip()
    ):
        raise ProfileContractError(
            f"{field}.model_type 必须是无首尾空白的非空字符串"
        )
    requirement_tps = _positive_integer_list(
        value["tp_degrees"],
        field=f"{field}.tp_degrees",
    )
    requirement_scenarios = _strict_string_list(
        value["scenario_ids"],
        field=f"{field}.scenario_ids",
        allow_empty=False,
    )
    dense_layers = _strict_string_list(
        value["dense_layers"],
        field=f"{field}.dense_layers",
        allow_empty=True,
    )
    per_sequence_layers = _strict_string_list(
        value["per_sequence_layers"],
        field=f"{field}.per_sequence_layers",
        allow_empty=True,
    )
    for boolean_field in ("attention_required", "moe_required"):
        if type(value[boolean_field]) is not bool:
            raise ProfileContractError(
                f"{field}.{boolean_field} 必须是布尔值"
            )

    if set(requirement_tps) != set(tp_degrees):
        raise ProfileContractError(
            f"{field}.tp_degrees 必须与顶层 tp_degrees 完全一致"
        )
    if set(requirement_scenarios) != scenario_ids:
        raise ProfileContractError(
            f"{field}.scenario_ids 必须与 scenario_catalog 完全一致"
        )

    return {
        "model_type": model_type,
        "tp_degrees": requirement_tps,
        "scenario_ids": requirement_scenarios,
        "dense_layers": dense_layers,
        "per_sequence_layers": per_sequence_layers,
        "attention_required": value["attention_required"],
        "moe_required": value["moe_required"],
    }


def _validate_latency_accounting(
    value: object,
    *,
    source: str,
) -> Mapping[str, Any]:
    field = f"{source}: latency_accounting"
    if not isinstance(value, dict):
        raise ProfileContractError(f"{field} 必须是 mapping")
    if value.get("demand_access_included") is not True:
        raise ProfileContractError(
            f"{field}.demand_access_included 必须显式为 true"
        )

    includes = _string_set(value.get("includes"), field=f"{field}.includes")
    excludes = _string_set(value.get("excludes"), field=f"{field}.excludes")
    missing_includes = sorted(_REQUIRED_ACCOUNTING_INCLUDES - includes)
    missing_excludes = sorted(_REQUIRED_ACCOUNTING_EXCLUDES - excludes)
    if missing_includes:
        raise ProfileContractError(
            f"{field}.includes 缺少 {', '.join(missing_includes)}"
        )
    if missing_excludes:
        raise ProfileContractError(
            f"{field}.excludes 缺少 {', '.join(missing_excludes)}"
        )
    overlap = sorted(includes & excludes)
    if overlap:
        raise ProfileContractError(
            f"{field} 的 includes/excludes 互相冲突：{', '.join(overlap)}"
        )
    return value


def _validate_scenario_catalog(
    value: object,
    *,
    source: str,
) -> Mapping[str, Mapping[str, Any]]:
    field = f"{source}: scenario_catalog"
    if not isinstance(value, dict) or not value:
        raise ProfileContractError(f"{field} 必须是非空 mapping")

    catalog: dict[str, Mapping[str, Any]] = {}
    for scenario_id, declaration in value.items():
        scenario = validate_memory_profile_id(
            scenario_id,
            field=f"{field} 的 scenario ID",
        )
        if not isinstance(declaration, dict):
            raise ProfileContractError(
                f"{field}.{scenario} 必须是 mapping"
            )
        declared_id = declaration.get("scenario_id")
        if declared_id is not None and declared_id != scenario:
            raise ProfileContractError(
                f"{field}.{scenario}.scenario_id={declared_id!r} "
                "必须与 catalog key 一致"
            )

        mapping_keys = [key for key in _SCENARIO_MAPPING_KEYS if key in declaration]
        if len(mapping_keys) != 1:
            choices = "/".join(_SCENARIO_MAPPING_KEYS)
            raise ProfileContractError(
                f"{field}.{scenario} 必须且只能声明一个 {choices} mapping"
            )
        mapping_key = mapping_keys[0]
        access_mapping = declaration[mapping_key]
        if not isinstance(access_mapping, dict) or not access_mapping:
            raise ProfileContractError(
                f"{field}.{scenario}.{mapping_key} 必须是非空 mapping"
            )
        for access_key, tier in access_mapping.items():
            if not isinstance(access_key, str) or not access_key:
                raise ProfileContractError(
                    f"{field}.{scenario}.{mapping_key} 的 access key 必须是非空字符串"
                )
            if tier not in ("hbm", "hbf"):
                raise ProfileContractError(
                    f"{field}.{scenario}.{mapping_key}.{access_key} "
                    f"必须是 'hbm' 或 'hbf'，实际为 {tier!r}"
                )
        catalog[scenario] = declaration
    return catalog


def validate_profile_meta(
    meta: Mapping[str, Any],
    *,
    requested_memory_profile_id: str | None = None,
    source: str = "meta.yaml",
) -> ProfileContract:
    """校验 manifest；缺版本字段时按兼容规则识别为 v1。"""

    if not isinstance(meta, dict):
        raise ProfileContractError(f"{source} 的顶层必须是 mapping")
    if requested_memory_profile_id is not None:
        requested_memory_profile_id = validate_memory_profile_id(
            requested_memory_profile_id,
            field="请求的 memory_profile_id",
        )

    version = _schema_version(meta, source)
    if version == PROFILE_SCHEMA_V1:
        if requested_memory_profile_id is not None:
            raise ProfileContractError(
                f"{source}: 请求了 memory_profile_id={requested_memory_profile_id!r}，"
                "但 bundle 是 Profile v1"
            )
        mixed_fields = sorted(
            key
            for key in (
                "memory_profile_id",
                "scenario_catalog",
                "latency_accounting",
                "bundle_readiness",
                "runtime_compatible",
                "scenario_binding",
                "architecture_requirements",
            )
            if key in meta
        )
        if mixed_fields:
            raise ProfileContractError(
                f"{source}: Profile v1 不得混入 v2 字段：{', '.join(mixed_fields)}"
            )
        return ProfileContract(
            schema_version=version,
            memory_profile_id=None,
            scenario_catalog={},
            latency_accounting={},
            bundle_readiness=None,
            runtime_compatible=None,
            scenario_binding=None,
            tp_degrees=(),
            architecture_requirements=None,
            meta=meta,
        )

    if requested_memory_profile_id is None:
        raise ProfileContractError(
            f"{source}: Profile v2 必须通过 memory_profile_id 显式选择"
        )
    manifest_id = validate_memory_profile_id(
        meta.get("memory_profile_id"),
        field=f"{source}: memory_profile_id",
    )
    if manifest_id != requested_memory_profile_id:
        raise ProfileContractError(
            f"{source}: memory_profile_id={manifest_id!r} 与请求的 "
            f"{requested_memory_profile_id!r} 不一致"
        )

    catalog = _validate_scenario_catalog(
        meta.get("scenario_catalog"),
        source=source,
    )
    accounting = _validate_latency_accounting(
        meta.get("latency_accounting"),
        source=source,
    )
    readiness = meta.get("bundle_readiness")
    if readiness not in _SCENARIO_BINDING_BY_READINESS:
        raise ProfileContractError(
            f"{source}: bundle_readiness 必须是 "
            f"{PROFILE_V2_PARTIAL!r} 或 {PROFILE_V2_RUNTIME_READY!r}"
        )
    runtime_compatible = meta.get("runtime_compatible")
    if type(runtime_compatible) is not bool:
        raise ProfileContractError(
            f"{source}: runtime_compatible 必须是布尔值"
        )
    expected_compatible = readiness == PROFILE_V2_RUNTIME_READY
    if runtime_compatible is not expected_compatible:
        raise ProfileContractError(
            f"{source}: bundle_readiness={readiness!r} 时 "
            f"runtime_compatible 必须为 {str(expected_compatible).lower()}"
        )

    scenario_binding = meta.get("scenario_binding")
    if not isinstance(scenario_binding, str) or not scenario_binding:
        raise ProfileContractError(
            f"{source}: scenario_binding 必须是非空字符串"
        )
    expected_binding = _SCENARIO_BINDING_BY_READINESS[readiness]
    if scenario_binding != expected_binding:
        raise ProfileContractError(
            f"{source}: bundle_readiness={readiness!r} 时 "
            f"scenario_binding 必须是 {expected_binding!r}"
        )

    tp_degrees = _positive_integer_list(
        meta.get("tp_degrees"),
        field=f"{source}: tp_degrees",
    )
    requirements_value = meta.get("architecture_requirements")
    if readiness == PROFILE_V2_PARTIAL:
        if requirements_value is not None:
            raise ProfileContractError(
                f"{source}: partial bundle 不得声明 architecture_requirements"
            )
        architecture_requirements = None
    else:
        architecture_requirements = _validate_architecture_requirements(
            requirements_value,
            source=source,
            tp_degrees=tp_degrees,
            scenario_ids=set(catalog),
        )

    return ProfileContract(
        schema_version=version,
        memory_profile_id=manifest_id,
        scenario_catalog=catalog,
        latency_accounting=accounting,
        bundle_readiness=readiness,
        runtime_compatible=runtime_compatible,
        scenario_binding=scenario_binding,
        tp_degrees=tp_degrees,
        architecture_requirements=architecture_requirements,
        meta=meta,
    )


def _parse_nonnegative_integer(
    value: object,
    *,
    field: str,
    positive: bool = False,
) -> int:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ProfileContractError(f"{field} 必须是整数，实际为 {value!r}") from None
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        raise ProfileContractError(f"{field} 必须是整数，实际为 {value!r}")
    result = int(parsed)
    lower_bound = 1 if positive else 0
    if result < lower_bound:
        relation = "正整数" if positive else "非负整数"
        raise ProfileContractError(f"{field} 必须是{relation}，实际为 {value!r}")
    return result


def _normalise_shape_value(
    column: str,
    value: object,
    *,
    field: str,
) -> str | int:
    if column == "layer":
        if not isinstance(value, str) or not value.strip():
            raise ProfileContractError(f"{field} 必须是非空字符串")
        return value.strip()
    return _parse_nonnegative_integer(
        value,
        field=field,
        positive=column in _POSITIVE_SHAPE_COLUMNS,
    )


def _validate_performance_csv(
    path: str,
    *,
    shape_columns: tuple[str, ...],
    scenarios: set[str],
) -> None:
    required = set(shape_columns) | {
        "memory_scenario",
        "time_us",
        *V2_AUDIT_COLUMNS,
    }
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            if not headers:
                raise ProfileContractError(f"{path}: CSV 缺少表头")
            if len(headers) != len(set(headers)):
                raise ProfileContractError(f"{path}: CSV 表头不得包含重复列")
            missing = sorted(required - set(headers))
            if missing:
                raise ProfileContractError(
                    f"{path}: Profile v2 CSV 缺少列：{', '.join(missing)}"
                )

            seen: set[tuple[object, ...]] = set()
            row_count = 0
            for row_number, row in enumerate(reader, start=2):
                row_count += 1
                field_prefix = f"{path}:{row_number}"
                if None in row:
                    raise ProfileContractError(
                        f"{field_prefix}: 字段数量超过表头定义"
                    )

                scenario = (row.get("memory_scenario") or "").strip()
                if not scenario:
                    raise ProfileContractError(
                        f"{field_prefix}: memory_scenario 不能为空"
                    )
                if scenario not in scenarios:
                    raise ProfileContractError(
                        f"{field_prefix}: memory_scenario={scenario!r} "
                        "未在 scenario_catalog 声明"
                    )

                try:
                    time_us = float(row.get("time_us", ""))
                except (TypeError, ValueError):
                    raise ProfileContractError(
                        f"{field_prefix}: time_us 必须是有限正数"
                    ) from None
                if not math.isfinite(time_us) or time_us <= 0:
                    raise ProfileContractError(
                        f"{field_prefix}: time_us 必须是有限正数，实际为 "
                        f"{row.get('time_us')!r}"
                    )

                for column in V2_AUDIT_COLUMNS:
                    _parse_nonnegative_integer(
                        row.get(column),
                        field=f"{field_prefix}: {column}",
                    )

                shape = tuple(
                    _normalise_shape_value(
                        column,
                        row.get(column),
                        field=f"{field_prefix}: {column}",
                    )
                    for column in shape_columns
                )
                unique_key = (*shape, scenario)
                if unique_key in seen:
                    raise ProfileContractError(
                        f"{field_prefix}: 同一 shape 与 memory_scenario "
                        f"重复：{unique_key!r}"
                    )
                seen.add(unique_key)
            if row_count == 0:
                raise ProfileContractError(f"{path}: Profile v2 CSV 不得为空")
    except ProfileContractError:
        raise
    except csv.Error as exc:
        raise ProfileContractError(f"{path}: CSV 解析失败：{exc}") from exc
    except OSError as exc:
        raise ProfileContractError(f"{path}: CSV 读取失败：{exc}") from exc


def validate_v2_performance_tables(
    profile_root: str,
    contract: ProfileContract,
) -> None:
    """校验 v2 bundle 中所有已存在的类别性能表。"""

    if not contract.is_v2:
        raise ProfileContractError("仅 Profile v2 可以执行 v2 CSV 契约校验")

    table_count = 0
    try:
        entries = sorted(os.listdir(profile_root))
    except OSError as exc:
        raise ProfileContractError(
            f"无法读取 Profile bundle 目录 {profile_root}：{exc}"
        ) from exc

    for entry in entries:
        if not re.fullmatch(r"tp[0-9]+", entry):
            continue
        tp_dir = os.path.join(profile_root, entry)
        if not os.path.isdir(tp_dir):
            continue
        for filename, shape_columns in _CATEGORY_SHAPE_COLUMNS.items():
            path = os.path.join(tp_dir, filename)
            if not os.path.isfile(path):
                continue
            table_count += 1
            _validate_performance_csv(
                path,
                shape_columns=shape_columns,
                scenarios=set(contract.scenario_catalog),
            )
    if table_count == 0:
        raise ProfileContractError(
            f"{profile_root}: Profile v2 至少需要一个 tp<N>/ 类别性能 CSV"
        )


def load_profile_contract(
    profile_root: str,
    *,
    requested_memory_profile_id: str | None = None,
) -> ProfileContract:
    """读取并完整校验 bundle 的 manifest 与版本相关 CSV。"""

    meta_path = os.path.join(profile_root, "meta.yaml")
    meta = _read_meta(profile_root)
    contract = validate_profile_meta(
        meta,
        requested_memory_profile_id=requested_memory_profile_id,
        source=meta_path,
    )
    if contract.is_v2:
        validate_v2_performance_tables(profile_root, contract)
    return contract
