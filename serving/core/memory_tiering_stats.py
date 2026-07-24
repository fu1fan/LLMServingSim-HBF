"""HBF 分层驻留与显式迁移统计。"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .memory_tiering import (
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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resident_high_water_bytes",
            MappingProxyType(dict(self.resident_high_water_bytes)),
        )
        object.__setattr__(
            self,
            "capacity_high_water_bytes",
            MappingProxyType(dict(self.capacity_high_water_bytes)),
        )
        object.__setattr__(
            self,
            "transfer_directions",
            MappingProxyType(dict(self.transfer_directions)),
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

    def record_completed_transfer(self, operation: TransferOperation) -> None:
        """记录 ASTRA 已完成的显式搬运，而非 Profile demand 访存。"""

        if not isinstance(operation, TransferOperation):
            raise TypeError("operation 必须是 TransferOperation")
        if operation.source is operation.target:
            raise ValueError("显式迁移的 source 与 target 不能相同")
        sizes = _validate_rank_values(
            operation.bytes_per_rank,
            self.num_ranks,
            "operation.bytes_per_rank",
        )
        key = (operation.source, operation.target)
        current = self._transfer_directions.get(key)
        if current is None:
            current = [0, [0] * self.num_ranks]
            self._transfer_directions[key] = current
        current[0] += 1
        for rank, amount in enumerate(sizes):
            current[1][rank] += amount

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
            transfer_directions={
                key: CountedBytes(
                    operations=value[0],
                    bytes_per_rank=tuple(value[1]),
                )
                for key, value in self._transfer_directions.items()
            },
        )
