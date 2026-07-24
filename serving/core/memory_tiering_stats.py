"""HBF 分层驻留与显式迁移统计。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from .memory_tiering import (
    MemoryObjectKind,
    MemoryTier,
    ResidencySnapshot,
    TransferOperation,
)


_CAPACITY_TIERS = (MemoryTier.HBM, MemoryTier.HBF)


@dataclass(frozen=True)
class CountedBytes:
    """一组事件的逻辑次数和各 rank 物理字节。"""

    operations: int
    bytes_per_rank: tuple[int, ...]

    @property
    def total_bytes(self) -> int:
        return sum(self.bytes_per_rank)

    def to_dict(self) -> dict:
        return {
            "operations": self.operations,
            "total_bytes": self.total_bytes,
            "bytes_per_rank": list(self.bytes_per_rank),
        }


@dataclass(frozen=True)
class MemoryTieringStatsSnapshot:
    """可跨批次保存的不可变分层统计快照。"""

    num_ranks: int
    resident_high_water_bytes: Mapping[MemoryTier, tuple[int, ...]]
    capacity_high_water_bytes: Mapping[MemoryTier, tuple[int, ...]]
    transfer_directions: Mapping[
        tuple[MemoryTier, MemoryTier],
        CountedBytes,
    ]
    transfers_by_reason: Mapping[str, CountedBytes]
    transfers_by_object_kind: Mapping[MemoryObjectKind, CountedBytes]
    transfers_by_layer: Mapping[int | None, CountedBytes]
    policy_actions: Mapping[str, int]
    residency_batches: int
    residency_hit_batches: int
    attention_group_observations: int
    attention_hbm_groups: int
    attention_hbf_groups: int

    def __post_init__(self) -> None:
        for field in (
            "resident_high_water_bytes",
            "capacity_high_water_bytes",
            "transfer_directions",
            "transfers_by_reason",
            "transfers_by_object_kind",
            "transfers_by_layer",
            "policy_actions",
        ):
            object.__setattr__(
                self,
                field,
                MappingProxyType(dict(getattr(self, field))),
            )

    def to_dict(self) -> dict:
        """转换为不含 Enum 键和 tuple 的 JSON 友好结构。"""

        return {
            "schema": "llmservingsim_memory_tiering_stats_v1",
            "num_ranks": self.num_ranks,
            "resident_high_water_bytes": {
                tier.value: list(values)
                for tier, values in self.resident_high_water_bytes.items()
            },
            "capacity_high_water_bytes": {
                tier.value: list(values)
                for tier, values in self.capacity_high_water_bytes.items()
            },
            "explicit_transfers": {
                "directions": {
                    f"{source.value}->{target.value}": counter.to_dict()
                    for (source, target), counter
                    in self.transfer_directions.items()
                },
                "by_reason": {
                    reason: counter.to_dict()
                    for reason, counter in self.transfers_by_reason.items()
                },
                "by_object_kind": {
                    kind.value: counter.to_dict()
                    for kind, counter in self.transfers_by_object_kind.items()
                },
                "by_layer": {
                    "unscoped" if layer is None else str(layer): counter.to_dict()
                    for layer, counter in self.transfers_by_layer.items()
                },
            },
            "policy_actions": dict(self.policy_actions),
            "residency_batches": {
                "observed": self.residency_batches,
                "hits": self.residency_hit_batches,
                "misses": self.residency_batches - self.residency_hit_batches,
            },
            "attention_groups": {
                "observations": self.attention_group_observations,
                "hbm": self.attention_hbm_groups,
                "hbf": self.attention_hbf_groups,
            },
        }


def _validate_rank_values(values, num_ranks, field):
    result = tuple(values)
    if len(result) != num_ranks:
        raise ValueError(f"{field} 必须包含 {num_ranks} 个 rank")
    for value in result:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} 必须只包含非负整数")
    return result


def _positive_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} 必须是正整数")
    return value


def _nonnegative_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} 必须是非负整数")
    return value


def _name(value, field):
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须是非空字符串或字符串 Enum")
    return value.strip()


class MemoryTieringStats:
    """累积分层容量和 Serving 显式完成的搬运。

    Profile 内部的 HBM/HBF demand 四向字节已经包含在算子性能中，
    本类没有接收该类字节的入口，避免再次计为迁移。
    """

    def __init__(self, *, num_ranks: int) -> None:
        if (
            isinstance(num_ranks, bool)
            or not isinstance(num_ranks, int)
            or num_ranks <= 0
        ):
            raise ValueError("num_ranks 必须是正整数")
        self.num_ranks = num_ranks
        self._resident_high_water = {
            tier: [0] * num_ranks for tier in _CAPACITY_TIERS
        }
        self._capacity_high_water = {
            tier: [0] * num_ranks for tier in _CAPACITY_TIERS
        }
        self._transfer_directions = {}
        self._transfers_by_reason = {}
        self._transfers_by_object_kind = {}
        self._transfers_by_layer = {}
        self._policy_actions = {}
        self._residency_batches = 0
        self._residency_hit_batches = 0
        self._attention_group_observations = 0
        self._attention_hbm_groups = 0
        self._attention_hbf_groups = 0

    def _increment_bytes(self, counters, key, sizes):
        current = counters.get(key)
        if current is None:
            current = [0, [0] * self.num_ranks]
            counters[key] = current
        current[0] += 1
        for rank, amount in enumerate(sizes):
            current[1][rank] += amount

    def observe_residency(self, snapshot: ResidencySnapshot) -> None:
        """观察一次账本状态；容量高水位包含目标侧预留。"""

        if not isinstance(snapshot, ResidencySnapshot):
            raise TypeError("snapshot 必须是 ResidencySnapshot")
        for tier in _CAPACITY_TIERS:
            used = _validate_rank_values(
                snapshot.used_bytes.get(tier, (0,) * self.num_ranks),
                self.num_ranks,
                f"{tier.value} used_bytes",
            )
            reserved = _validate_rank_values(
                snapshot.reserved_bytes.get(tier, (0,) * self.num_ranks),
                self.num_ranks,
                f"{tier.value} reserved_bytes",
            )
            for rank in range(self.num_ranks):
                self._resident_high_water[tier][rank] = max(
                    self._resident_high_water[tier][rank],
                    used[rank],
                )
                self._capacity_high_water[tier][rank] = max(
                    self._capacity_high_water[tier][rank],
                    used[rank] + reserved[rank],
                )

    def observe_usage(self, used_bytes, reserved_bytes=None) -> None:
        """观察未使用 TieredResidencyManager 的运行时容量账本。"""

        if not isinstance(used_bytes, Mapping):
            raise TypeError("used_bytes 必须是 mapping")
        reserved_bytes = reserved_bytes or {}
        if not isinstance(reserved_bytes, Mapping):
            raise TypeError("reserved_bytes 必须是 mapping")
        for tier in _CAPACITY_TIERS:
            used = _validate_rank_values(
                used_bytes.get(tier, (0,) * self.num_ranks),
                self.num_ranks,
                f"{tier.value} used_bytes",
            )
            reserved = _validate_rank_values(
                reserved_bytes.get(tier, (0,) * self.num_ranks),
                self.num_ranks,
                f"{tier.value} reserved_bytes",
            )
            for rank in range(self.num_ranks):
                self._resident_high_water[tier][rank] = max(
                    self._resident_high_water[tier][rank],
                    used[rank],
                )
                self._capacity_high_water[tier][rank] = max(
                    self._capacity_high_water[tier][rank],
                    used[rank] + reserved[rank],
                )

    def record_explicit_transfer(
        self,
        *,
        source,
        target,
        bytes_per_rank,
        reason,
        object_kind,
        layer_index=None,
    ) -> None:
        """记录一个已由 ASTRA 完成的通用显式搬运。"""

        if not isinstance(source, MemoryTier) or not isinstance(
            target,
            MemoryTier,
        ):
            raise TypeError("source 和 target 必须是 MemoryTier")
        if source is target:
            raise ValueError("显式迁移的 source 与 target 不能相同")
        if not isinstance(object_kind, MemoryObjectKind):
            raise TypeError("object_kind 必须是 MemoryObjectKind")
        sizes = _validate_rank_values(
            bytes_per_rank,
            self.num_ranks,
            "bytes_per_rank",
        )
        reason = _name(reason, "reason")
        self._increment_bytes(
            self._transfer_directions,
            (source, target),
            sizes,
        )
        self._increment_bytes(self._transfers_by_reason, reason, sizes)
        self._increment_bytes(
            self._transfers_by_object_kind,
            object_kind,
            sizes,
        )
        self._increment_bytes(
            self._transfers_by_layer,
            layer_index,
            sizes,
        )

    def record_completed_transfer(self, operation: TransferOperation) -> None:
        """记录 ASTRA 已完成的显式搬运，而非 Profile demand 访存。"""

        if not isinstance(operation, TransferOperation):
            raise TypeError("operation 必须是 TransferOperation")
        self.record_explicit_transfer(
            source=operation.source,
            target=operation.target,
            bytes_per_rank=operation.bytes_per_rank,
            reason=operation.reason,
            object_kind=operation.object_key.kind,
            layer_index=operation.object_key.layer_index,
        )

    def record_policy_action(self, action, *, count: int = 1) -> None:
        """记录策略引擎的决定；KEEP 等动作不会伪造迁移流量。"""

        normalized = _name(action, "action")
        count = _positive_int(count, "count")
        self._policy_actions[normalized] = (
            self._policy_actions.get(normalized, 0) + count
        )

    def record_residency_batch(self, *, hit: bool) -> None:
        """记录一个 batch 是否无需额外迁移即可满足目标驻留。"""

        if not isinstance(hit, bool):
            raise TypeError("hit 必须是 bool")
        self._residency_batches += 1
        if hit:
            self._residency_hit_batches += 1

    def record_attention_groups(
        self,
        *,
        hbm_groups: int,
        hbf_groups: int,
    ) -> None:
        """累计一次 Attention lookup 的真实驻留分组数。"""

        hbm_groups = _nonnegative_int(hbm_groups, "hbm_groups")
        hbf_groups = _nonnegative_int(hbf_groups, "hbf_groups")
        self._attention_group_observations += 1
        self._attention_hbm_groups += hbm_groups
        self._attention_hbf_groups += hbf_groups

    def _freeze_bytes(self, counters):
        return {
            key: CountedBytes(
                operations=value[0],
                bytes_per_rank=tuple(value[1]),
            )
            for key, value in counters.items()
        }

    def snapshot(self) -> MemoryTieringStatsSnapshot:
        return MemoryTieringStatsSnapshot(
            num_ranks=self.num_ranks,
            resident_high_water_bytes={
                tier: tuple(values)
                for tier, values in self._resident_high_water.items()
            },
            capacity_high_water_bytes={
                tier: tuple(values)
                for tier, values in self._capacity_high_water.items()
            },
            transfer_directions=self._freeze_bytes(
                self._transfer_directions
            ),
            transfers_by_reason=self._freeze_bytes(
                self._transfers_by_reason
            ),
            transfers_by_object_kind=self._freeze_bytes(
                self._transfers_by_object_kind
            ),
            transfers_by_layer=self._freeze_bytes(
                self._transfers_by_layer
            ),
            policy_actions=dict(self._policy_actions),
            residency_batches=self._residency_batches,
            residency_hit_batches=self._residency_hit_batches,
            attention_group_observations=self._attention_group_observations,
            attention_hbm_groups=self._attention_hbm_groups,
            attention_hbf_groups=self._attention_hbf_groups,
        )
