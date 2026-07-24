"""由已验证 Profile 生成 ASTRA-Sim HBM/HBF 迁移资源。"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Mapping

from .profile_contract import load_profile_contract


class HbfMemoryConfigError(ValueError):
    """Profile 四向参数不能安全映射到 ASTRA-Sim。"""


@dataclass(frozen=True)
class HbfAstraMemorySpec:
    integration_mode: str
    local_mem: Mapping[str, object]
    hbf_mem: Mapping[str, object]


def _positive_number(value, field):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise HbfMemoryConfigError(f"{field} 必须是有限正数")
    return float(value)


def _direction(tier, name, field):
    if not isinstance(tier, Mapping):
        raise HbfMemoryConfigError(f"{field} 必须是 mapping")
    value = tier.get(name)
    if not isinstance(value, Mapping):
        raise HbfMemoryConfigError(f"{field}.{name} 必须是 mapping")
    bandwidth = _positive_number(
        value.get("bandwidth_byte_per_second"),
        f"{field}.{name}.bandwidth_byte_per_second",
    )
    latency = _positive_number(
        value.get("fixed_latency_second"),
        f"{field}.{name}.fixed_latency_second",
    )
    # ASTRA analytical backend 的单位分别是十进制 GB/s 与 ns。
    return max(1, int(bandwidth / 1e9)), max(1, math.ceil(latency * 1e9))


def astra_memory_spec_from_integration(memory_integration):
    """把 LLMCompass memory_integration 转为显式迁移资源。"""

    if not isinstance(memory_integration, Mapping):
        raise HbfMemoryConfigError("memory_integration 必须是 mapping")
    mode = memory_integration.get("mode")
    if mode not in {"cli", "csi"}:
        raise HbfMemoryConfigError("memory_integration.mode 必须是 cli 或 csi")
    parameters = memory_integration.get("parameters")
    if not isinstance(parameters, Mapping):
        raise HbfMemoryConfigError("memory_integration.parameters 必须是 mapping")
    if parameters.get("integration_mode") != mode:
        raise HbfMemoryConfigError(
            "memory_integration.mode 与 parameters.integration_mode 不一致"
        )
    tiers = parameters.get("tiers")
    if not isinstance(tiers, Mapping):
        raise HbfMemoryConfigError("memory_integration.parameters.tiers 必须是 mapping")

    entries = {}
    for tier_name, location in (
        ("hbm", "LOCAL_MEMORY"),
        ("hbf", "HBF_MEMORY"),
    ):
        tier = tiers.get(tier_name)
        read_bw, read_latency = _direction(
            tier,
            "read",
            f"memory_integration.parameters.tiers.{tier_name}",
        )
        write_bw, write_latency = _direction(
            tier,
            "write",
            f"memory_integration.parameters.tiers.{tier_name}",
        )
        entries[tier_name] = {
            "memory-type": "PER_NPU_MEMORY_EXPANSION",
            "memory-location": location,
            "read-mem-bw": read_bw,
            "read-mem-latency": read_latency,
            "write-mem-bw": write_bw,
            "write-mem-latency": write_latency,
        }

    if mode == "cli":
        entries["hbm"]["service-group"] = "hbm-data"
        entries["hbf"]["service-group"] = "hbf-data"
    else:
        # CSI 仅共享显式迁移的数据传输资源；Profile 内 demand access 不会进入 ASTRA。
        entries["hbm"]["service-group"] = "gpu-memory-data"
        entries["hbf"]["service-group"] = "gpu-memory-data"
        fabric_bw = parameters.get(
            "gpu_memory_fabric_bandwidth_byte_per_second"
        )
        if fabric_bw is not None:
            shared_bw = max(
                1,
                int(
                    _positive_number(
                        fabric_bw,
                        "gpu_memory_fabric_bandwidth_byte_per_second",
                    )
                    / 1e9
                ),
            )
            entries["hbm"]["service-group-bw"] = shared_bw
            entries["hbf"]["service-group-bw"] = shared_bw

    return HbfAstraMemorySpec(
        integration_mode=mode,
        local_mem=entries["hbm"],
        hbf_mem=entries["hbf"],
    )


def install_hbf_memory_resources(
    memory_config_path,
    instances,
    runtime_configs,
    *,
    profiler_root,
    variant_resolver,
):
    """校验所有 instance 的性能身份一致后更新 ASTRA memory JSON。"""

    if len(instances) != len(runtime_configs):
        raise HbfMemoryConfigError("instances 与 runtime_configs 数量不一致")
    specs = []
    for instance, runtime in zip(instances, runtime_configs):
        tiering = runtime["memory_tiering"]
        if not tiering.enabled:
            continue
        profile_id = runtime["memory_scenario_policy"].memory_profile_id
        variant = variant_resolver(
            runtime["dtype"],
            runtime["kv_cache_dtype"],
            None,
        )
        profile_root = os.path.join(
            profiler_root,
            instance["hardware"],
            instance["model_name"],
            variant,
            profile_id,
        )
        contract = load_profile_contract(
            profile_root,
            requested_memory_profile_id=profile_id,
        )
        specs.append(
            (
                astra_memory_spec_from_integration(
                    contract.memory_integration
                ),
                contract.performance_identity,
            )
        )
    if not specs:
        return None

    first_spec, first_identity = specs[0]
    for spec, identity in specs[1:]:
        if spec != first_spec or identity != first_identity:
            raise HbfMemoryConfigError(
                "同一次仿真的 HBF instance 必须使用相同性能身份与内存集成参数"
            )

    with open(memory_config_path, "r", encoding="utf-8") as f:
        memory_config = json.load(f)
    memory_config["local_mem"] = dict(first_spec.local_mem)
    memory_config["hbf_mem"] = dict(first_spec.hbf_mem)
    with open(memory_config_path, "w", encoding="utf-8") as f:
        json.dump(memory_config, f, ensure_ascii=False, indent=2)
    return first_spec
