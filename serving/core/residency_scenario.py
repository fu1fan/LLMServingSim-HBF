"""由实际对象驻留派生 Profile v2 memory scenario。"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .memory_tiering import MemoryTier


class ResidencyScenarioError(RuntimeError):
    """Profile 场景目录不能表示当前实际驻留。"""


@dataclass(frozen=True)
class AccessDescriptor:
    """Profile 中一个稳定 logical access 的语义。"""

    operator_id: str
    access_id: str
    semantic: str
    access_type: str
    lifetime: str

    @property
    def key(self) -> str:
        return f"{self.operator_id}/{self.access_id}"


@dataclass(frozen=True)
class RuntimeScenarioBinding:
    """一次算子 lookup 的真实 placement 与匹配场景。"""

    operator_id: str
    scenario_id: str
    access_tiers: Mapping[str, MemoryTier]


@dataclass(frozen=True)
class BatchMemoryView:
    """一个 batch 生成 Trace 时不可变的对象驻留视图。"""

    snapshot_version: int
    weight_tiers: Mapping[tuple[str, int | None], MemoryTier]
    kv_tiers: Mapping[tuple[str, int], MemoryTier]

    def __post_init__(self):
        if (
            isinstance(self.snapshot_version, bool)
            or not isinstance(self.snapshot_version, int)
            or self.snapshot_version < 0
        ):
            raise ValueError("snapshot_version 必须是非负整数")
        for field, mapping in (
            ("weight_tiers", self.weight_tiers),
            ("kv_tiers", self.kv_tiers),
        ):
            if not isinstance(mapping, Mapping):
                raise TypeError(f"{field} 必须是 mapping")
            for tier in mapping.values():
                if tier not in {MemoryTier.HBM, MemoryTier.HBF}:
                    raise ResidencyScenarioError(
                        f"{field} 只允许 HBM/HBF 驻留"
                    )
        object.__setattr__(
            self,
            "weight_tiers",
            MappingProxyType(dict(self.weight_tiers)),
        )
        object.__setattr__(
            self,
            "kv_tiers",
            MappingProxyType(dict(self.kv_tiers)),
        )

    def weight_tier(
        self,
        layer_name: str,
        block_index: int | None,
    ) -> MemoryTier:
        """按精确 block、共享层定义的顺序读取权重驻留。"""

        exact = (layer_name, block_index)
        shared = (layer_name, None)
        if exact in self.weight_tiers:
            return self.weight_tiers[exact]
        if shared in self.weight_tiers:
            return self.weight_tiers[shared]
        raise ResidencyScenarioError(
            f"驻留视图缺少 weight {layer_name!r} block={block_index}"
        )

    def kv_tier(self, request_id: object, layer_index: int) -> MemoryTier:
        key = (str(request_id), layer_index)
        try:
            return self.kv_tiers[key]
        except KeyError as exc:
            raise ResidencyScenarioError(
                f"驻留视图缺少 request={request_id!r} layer={layer_index} 的 KV"
            ) from exc

    def kv_groups(
        self,
        request_ids,
        layer_index: int,
    ) -> Mapping[MemoryTier, tuple[str, ...]]:
        groups = {}
        for request_id in request_ids:
            tier = self.kv_tier(request_id, layer_index)
            groups.setdefault(tier, []).append(str(request_id))
        return MappingProxyType(
            {
                tier: tuple(ids)
                for tier, ids in sorted(
                    groups.items(),
                    key=lambda item: item[0].value,
                )
            }
        )


_WEIGHT_SEMANTICS = {"weight"}
_KV_SEMANTICS = {"kv_cache"}
_KNOWN_ACCESS_TYPES = {"read", "write", "read_write"}
_KNOWN_LIFETIMES = {"model", "request", "iteration", "operator"}


def _identifier(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ResidencyScenarioError(f"{field} 必须是非空字符串")
    return value.strip()


def _access_descriptor(key, value):
    if not isinstance(key, str) or key.count("/") != 1:
        raise ResidencyScenarioError(
            f"access_catalog key={key!r} 必须是 operator_id/access_id"
        )
    operator_id, access_id = (
        _identifier(part, f"access_catalog[{key!r}]")
        for part in key.split("/", 1)
    )
    if not isinstance(value, Mapping):
        raise ResidencyScenarioError(f"access_catalog[{key!r}] 必须是 mapping")
    expected = {"semantic", "access_type", "lifetime"}
    if set(value) != expected:
        raise ResidencyScenarioError(
            f"access_catalog[{key!r}] 必须且只能包含 {sorted(expected)}"
        )
    semantic = _identifier(value["semantic"], f"{key}.semantic").lower()
    access_type = _identifier(value["access_type"], f"{key}.access_type").lower()
    lifetime = _identifier(value["lifetime"], f"{key}.lifetime").lower()
    if access_type not in _KNOWN_ACCESS_TYPES:
        raise ResidencyScenarioError(f"{key}.access_type={access_type!r} 不受支持")
    if lifetime not in _KNOWN_LIFETIMES:
        raise ResidencyScenarioError(f"{key}.lifetime={lifetime!r} 不受支持")
    return AccessDescriptor(
        operator_id=operator_id,
        access_id=access_id,
        semantic=semantic,
        access_type=access_type,
        lifetime=lifetime,
    )


def _scenario_accesses(scenario_id, value, expected_keys):
    if not isinstance(value, Mapping):
        raise ResidencyScenarioError(f"scenario {scenario_id!r} 必须是 mapping")
    accesses = value.get("accesses")
    if not isinstance(accesses, Mapping):
        raise ResidencyScenarioError(
            f"scenario {scenario_id!r} 缺少完整 accesses mapping"
        )
    if set(accesses) != expected_keys:
        missing = sorted(expected_keys - set(accesses))
        extra = sorted(set(accesses) - expected_keys)
        raise ResidencyScenarioError(
            f"scenario {scenario_id!r} access 覆盖不完整："
            f"missing={missing}, extra={extra}"
        )
    parsed = {}
    for key, raw_tier in accesses.items():
        try:
            parsed[key] = MemoryTier(str(raw_tier).lower())
        except ValueError as exc:
            raise ResidencyScenarioError(
                f"scenario {scenario_id!r} 的 {key} 使用未知 tier={raw_tier!r}"
            ) from exc
        if parsed[key] not in {MemoryTier.HBM, MemoryTier.HBF}:
            raise ResidencyScenarioError(
                f"Profile demand access 只允许 HBM/HBF，实际为 {parsed[key].value}"
            )
    return MappingProxyType(parsed)


class ResidencyScenarioResolver:
    """把 weight/KV 的运行时驻留绑定到严格 Profile 场景。"""

    def __init__(self, access_catalog, scenario_catalog):
        if not isinstance(access_catalog, Mapping) or not access_catalog:
            raise ResidencyScenarioError("runtime-ready HBF Profile 缺少 access_catalog")
        if not isinstance(scenario_catalog, Mapping) or not scenario_catalog:
            raise ResidencyScenarioError("Profile 缺少 scenario_catalog")

        descriptors = {
            key: _access_descriptor(key, value)
            for key, value in access_catalog.items()
        }
        expected_keys = set(descriptors)
        scenarios = {
            _identifier(scenario_id, "scenario_id"): _scenario_accesses(
                scenario_id,
                value,
                expected_keys,
            )
            for scenario_id, value in scenario_catalog.items()
        }
        by_operator = {}
        for descriptor in descriptors.values():
            by_operator.setdefault(descriptor.operator_id, []).append(descriptor)
        for operator_id in by_operator:
            by_operator[operator_id] = tuple(
                sorted(by_operator[operator_id], key=lambda item: item.access_id)
            )

        self.access_catalog = MappingProxyType(descriptors)
        self.scenario_catalog = MappingProxyType(scenarios)
        self._by_operator = MappingProxyType(by_operator)
        self._cache = {}

    @property
    def operator_ids(self):
        return frozenset(self._by_operator)

    def _runtime_access_tiers(
        self,
        operator_id,
        *,
        weight_tier,
        kv_tier,
    ):
        try:
            descriptors = self._by_operator[operator_id]
        except KeyError as exc:
            raise ResidencyScenarioError(
                f"Profile access_catalog 不包含 operator {operator_id!r}"
            ) from exc
        if not isinstance(weight_tier, MemoryTier):
            raise TypeError("weight_tier 必须是 MemoryTier")
        if not isinstance(kv_tier, MemoryTier):
            raise TypeError("kv_tier 必须是 MemoryTier")
        if weight_tier not in {MemoryTier.HBM, MemoryTier.HBF}:
            raise ResidencyScenarioError("算子 weight demand 只允许 HBM/HBF")
        if kv_tier not in {MemoryTier.HBM, MemoryTier.HBF}:
            raise ResidencyScenarioError("算子 KV demand 只允许 HBM/HBF")

        result = {}
        for descriptor in descriptors:
            if descriptor.semantic in _WEIGHT_SEMANTICS:
                tier = weight_tier
            elif descriptor.semantic in _KV_SEMANTICS:
                tier = kv_tier
            else:
                # 本轮只允许持久对象进入 HBF，激活和临时张量保持 HBM。
                tier = MemoryTier.HBM
            result[descriptor.key] = tier
        return result

    def resolve(
        self,
        operator_id,
        *,
        weight_tier=MemoryTier.HBM,
        kv_tier=MemoryTier.HBM,
    ):
        cache_key = (operator_id, weight_tier, kv_tier)
        if cache_key in self._cache:
            return self._cache[cache_key]

        actual = self._runtime_access_tiers(
            operator_id,
            weight_tier=weight_tier,
            kv_tier=kv_tier,
        )
        operator_keys = set(actual)
        matches = []
        for scenario_id, accesses in self.scenario_catalog.items():
            if all(accesses[key] is actual[key] for key in operator_keys):
                matches.append(scenario_id)
        if not matches:
            rendered = {
                key: tier.value
                for key, tier in sorted(actual.items())
            }
            raise ResidencyScenarioError(
                f"operator {operator_id!r} 的实际驻留没有 Profile 场景：{rendered}"
            )

        # 全局场景可能对当前算子等价；稳定排序避免配置顺序改变 lookup。
        scenario_id = sorted(matches)[0]
        binding = RuntimeScenarioBinding(
            operator_id=operator_id,
            scenario_id=scenario_id,
            access_tiers=MappingProxyType(actual),
        )
        self._cache[cache_key] = binding
        return binding

    def preflight(
        self,
        *,
        allow_weight_hbf,
        allow_kv_hbf,
    ):
        """启动时枚举策略可达驻留，避免运行中才发现缺场景。"""

        weight_tiers = [MemoryTier.HBM]
        kv_tiers = [MemoryTier.HBM]
        if allow_weight_hbf:
            weight_tiers.append(MemoryTier.HBF)
        if allow_kv_hbf:
            kv_tiers.append(MemoryTier.HBF)
        bindings = []
        for operator_id in sorted(self.operator_ids):
            descriptors = self._by_operator[operator_id]
            has_weight = any(
                item.semantic in _WEIGHT_SEMANTICS for item in descriptors
            )
            has_kv = any(item.semantic in _KV_SEMANTICS for item in descriptors)
            for weight_tier in weight_tiers if has_weight else [MemoryTier.HBM]:
                for kv_tier in kv_tiers if has_kv else [MemoryTier.HBM]:
                    bindings.append(
                        self.resolve(
                            operator_id,
                            weight_tier=weight_tier,
                            kv_tier=kv_tier,
                        )
                    )
        return tuple(bindings)
