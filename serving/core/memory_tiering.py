"""HBM/HBF 分层驻留、容量预留与迁移事务。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping


class TieringError(RuntimeError):
    """分层内存状态或容量约束不成立。"""


class MemoryTier(str, Enum):
    """Serving 侧可追踪的物理内存层级。"""

    HBM = "hbm"
    HBF = "hbf"
    CPU = "cpu"
    CXL = "cxl"


class MemoryObjectKind(str, Enum):
    """需要长期驻留或显式搬运的对象类别。"""

    WEIGHT = "weight"
    KV = "kv"
    PREFIX = "prefix"
    STAGING = "staging"


class ResidencyState(str, Enum):
    """对象在驻留账本中的稳定或迁移状态。"""

    RESIDENT = "resident"
    TRANSFERRING = "transferring"


@dataclass(frozen=True, order=True)
class MemoryObjectKey:
    """跨批次稳定标识一个分层内存对象。"""

    kind: MemoryObjectKind
    object_id: str
    layer_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MemoryObjectKind):
            raise TypeError("kind 必须是 MemoryObjectKind")
        if not isinstance(self.object_id, str) or not self.object_id.strip():
            raise ValueError("object_id 必须是非空字符串")
        if self.layer_index is not None and (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise ValueError("layer_index 必须是非负整数或 None")


@dataclass(frozen=True)
class TransferOperation:
    """一次尚未完成的源读、目标写迁移。"""

    transfer_id: int
    object_key: MemoryObjectKey
    source: MemoryTier
    target: MemoryTier
    bytes_per_rank: tuple[int, ...]
    reason: str
    earliest_after: str = "batch_start"
    required_before: str = "batch_end"

    @property
    def total_bytes(self) -> int:
        return sum(self.bytes_per_rank)


@dataclass(frozen=True)
class ResidencySnapshot:
    """生成一个 batch 时使用的不可变驻留快照。"""

    version: int
    records: Mapping[MemoryObjectKey, "ResidencyRecord"]
    used_bytes: Mapping[MemoryTier, tuple[int, ...]]
    reserved_bytes: Mapping[MemoryTier, tuple[int, ...]]

    def tier_of(self, key: MemoryObjectKey) -> MemoryTier:
        try:
            return self.records[key].tier
        except KeyError as exc:
            raise TieringError(f"对象 {key} 尚未登记驻留信息") from exc


@dataclass(frozen=True)
class ResidencyRecord:
    """一个对象当前的稳定驻留及迁移预留。"""

    key: MemoryObjectKey
    bytes_per_rank: tuple[int, ...]
    tier: MemoryTier
    state: ResidencyState = ResidencyState.RESIDENT
    pending_target: MemoryTier | None = None
    transfer_id: int | None = None
    lock_count: int = 0
    last_access: int = 0


def _rank_bytes(values: Iterable[int], num_ranks: int, field: str) -> tuple[int, ...]:
    result = tuple(values)
    if len(result) != num_ranks:
        raise ValueError(f"{field} 必须包含 {num_ranks} 个 rank 的字节数")
    for value in result:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} 必须只包含非负整数")
    if not any(result):
        raise ValueError(f"{field} 不能全部为零")
    return result


class TieredResidencyManager:
    """按 rank 维护容量，并以事务方式规划对象迁移。"""

    def __init__(
        self,
        capacities: Mapping[MemoryTier, Iterable[int]],
        *,
        num_ranks: int,
    ) -> None:
        if isinstance(num_ranks, bool) or not isinstance(num_ranks, int):
            raise TypeError("num_ranks 必须是整数")
        if num_ranks <= 0:
            raise ValueError("num_ranks 必须大于零")
        if not isinstance(capacities, Mapping) or not capacities:
            raise ValueError("capacities 必须是非空 mapping")

        parsed = {}
        for tier, values in capacities.items():
            if not isinstance(tier, MemoryTier):
                raise TypeError("capacities 的键必须是 MemoryTier")
            parsed[tier] = _rank_bytes(values, num_ranks, f"{tier.value} capacity")

        self.num_ranks = num_ranks
        self.capacities = MappingProxyType(parsed)
        self._used = {
            tier: [0] * num_ranks
            for tier in self.capacities
        }
        self._reserved = {
            tier: [0] * num_ranks
            for tier in self.capacities
        }
        self._records: dict[MemoryObjectKey, ResidencyRecord] = {}
        self._pending: dict[int, TransferOperation] = {}
        self._next_transfer_id = 1
        self._clock = 0
        self._version = 0

    def _require_tier(self, tier: MemoryTier) -> None:
        if not isinstance(tier, MemoryTier):
            raise TypeError("tier 必须是 MemoryTier")
        if tier not in self.capacities:
            raise TieringError(f"未配置 {tier.value} 容量")

    def _check_capacity(
        self,
        tier: MemoryTier,
        bytes_per_rank: tuple[int, ...],
        *,
        include_reserved: bool,
    ) -> None:
        self._require_tier(tier)
        for rank, amount in enumerate(bytes_per_rank):
            occupied = self._used[tier][rank]
            if include_reserved:
                occupied += self._reserved[tier][rank]
            available = self.capacities[tier][rank] - occupied
            if amount > available:
                raise TieringError(
                    f"{tier.value} rank {rank} 容量不足："
                    f"需要 {amount} 字节，可用 {available} 字节"
                )

    def register(
        self,
        key: MemoryObjectKey,
        bytes_per_rank: Iterable[int],
        tier: MemoryTier,
    ) -> ResidencyRecord:
        """登记一个已经完成物理放置的对象。"""

        if not isinstance(key, MemoryObjectKey):
            raise TypeError("key 必须是 MemoryObjectKey")
        if key in self._records:
            raise TieringError(f"对象 {key} 已经登记")
        sizes = _rank_bytes(bytes_per_rank, self.num_ranks, "bytes_per_rank")
        self._check_capacity(tier, sizes, include_reserved=True)
        for rank, amount in enumerate(sizes):
            self._used[tier][rank] += amount
        record = ResidencyRecord(key=key, bytes_per_rank=sizes, tier=tier)
        self._records[key] = record
        self._version += 1
        return record

    def remove(self, key: MemoryObjectKey) -> None:
        """移除未锁定且没有迁移中的对象。"""

        record = self.record(key)
        if record.lock_count:
            raise TieringError(f"对象 {key} 仍被锁定")
        if record.state is not ResidencyState.RESIDENT:
            raise TieringError(f"对象 {key} 正在迁移")
        for rank, amount in enumerate(record.bytes_per_rank):
            self._used[record.tier][rank] -= amount
        del self._records[key]
        self._version += 1

    def record(self, key: MemoryObjectKey) -> ResidencyRecord:
        try:
            return self._records[key]
        except KeyError as exc:
            raise TieringError(f"对象 {key} 尚未登记") from exc

    def touch(self, key: MemoryObjectKey) -> ResidencyRecord:
        """更新对象的稳定 LRU 序号。"""

        record = self.record(key)
        self._clock += 1
        updated = ResidencyRecord(
            key=record.key,
            bytes_per_rank=record.bytes_per_rank,
            tier=record.tier,
            state=record.state,
            pending_target=record.pending_target,
            transfer_id=record.transfer_id,
            lock_count=record.lock_count,
            last_access=self._clock,
        )
        self._records[key] = updated
        return updated

    def lock(self, key: MemoryObjectKey) -> ResidencyRecord:
        record = self.record(key)
        updated = ResidencyRecord(
            **{
                **record.__dict__,
                "lock_count": record.lock_count + 1,
            }
        )
        self._records[key] = updated
        return updated

    def unlock(self, key: MemoryObjectKey) -> ResidencyRecord:
        record = self.record(key)
        if record.lock_count == 0:
            raise TieringError(f"对象 {key} 未被锁定")
        updated = ResidencyRecord(
            **{
                **record.__dict__,
                "lock_count": record.lock_count - 1,
            }
        )
        self._records[key] = updated
        return updated

    def plan_transfers(
        self,
        requests: Iterable[
            tuple[MemoryObjectKey, MemoryTier, str, str, str]
        ],
    ) -> tuple[TransferOperation, ...]:
        """原子预留一组迁移；任一失败时不留下部分状态。"""

        normalized = tuple(requests)
        if not normalized:
            return ()

        reserved_delta = {
            tier: [0] * self.num_ranks
            for tier in self.capacities
        }
        operations = []
        seen = set()
        next_id = self._next_transfer_id

        for key, target, reason, earliest_after, required_before in normalized:
            if key in seen:
                raise TieringError(f"同一事务重复迁移对象 {key}")
            seen.add(key)
            record = self.record(key)
            self._require_tier(target)
            if record.state is not ResidencyState.RESIDENT:
                raise TieringError(f"对象 {key} 已在迁移")
            if record.lock_count:
                raise TieringError(f"对象 {key} 被锁定，不能迁移")
            if target is record.tier:
                continue
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("迁移 reason 必须是非空字符串")
            if not isinstance(earliest_after, str) or not earliest_after:
                raise ValueError("earliest_after 必须是非空字符串")
            if not isinstance(required_before, str) or not required_before:
                raise ValueError("required_before 必须是非空字符串")

            for rank, amount in enumerate(record.bytes_per_rank):
                occupied = (
                    self._used[target][rank]
                    + self._reserved[target][rank]
                    + reserved_delta[target][rank]
                )
                available = self.capacities[target][rank] - occupied
                if amount > available:
                    raise TieringError(
                        f"{target.value} rank {rank} 容量不足："
                        f"需要 {amount} 字节，可用 {available} 字节"
                    )
                reserved_delta[target][rank] += amount

            operations.append(
                TransferOperation(
                    transfer_id=next_id,
                    object_key=key,
                    source=record.tier,
                    target=target,
                    bytes_per_rank=record.bytes_per_rank,
                    reason=reason.strip(),
                    earliest_after=earliest_after,
                    required_before=required_before,
                )
            )
            next_id += 1

        for tier, values in reserved_delta.items():
            for rank, amount in enumerate(values):
                self._reserved[tier][rank] += amount

        for operation in operations:
            record = self._records[operation.object_key]
            self._records[operation.object_key] = ResidencyRecord(
                key=record.key,
                bytes_per_rank=record.bytes_per_rank,
                tier=record.tier,
                state=ResidencyState.TRANSFERRING,
                pending_target=operation.target,
                transfer_id=operation.transfer_id,
                lock_count=record.lock_count,
                last_access=record.last_access,
            )
            self._pending[operation.transfer_id] = operation

        if operations:
            self._next_transfer_id = next_id
            self._version += 1
        return tuple(operations)

    def commit_transfer(self, transfer_id: int) -> ResidencyRecord:
        """在 ASTRA 传输完成后提交目标驻留并释放源容量。"""

        try:
            operation = self._pending.pop(transfer_id)
        except KeyError as exc:
            raise TieringError(f"未知 transfer_id={transfer_id}") from exc
        record = self.record(operation.object_key)
        if (
            record.state is not ResidencyState.TRANSFERRING
            or record.transfer_id != transfer_id
        ):
            raise TieringError(f"对象 {record.key} 的迁移状态不一致")

        for rank, amount in enumerate(record.bytes_per_rank):
            self._reserved[operation.target][rank] -= amount
            self._used[operation.target][rank] += amount
            self._used[operation.source][rank] -= amount
        updated = ResidencyRecord(
            key=record.key,
            bytes_per_rank=record.bytes_per_rank,
            tier=operation.target,
            last_access=record.last_access,
        )
        self._records[record.key] = updated
        self._version += 1
        return updated

    def abort_transfer(self, transfer_id: int) -> ResidencyRecord:
        """取消尚未提交的迁移并释放目标容量预留。"""

        try:
            operation = self._pending.pop(transfer_id)
        except KeyError as exc:
            raise TieringError(f"未知 transfer_id={transfer_id}") from exc
        record = self.record(operation.object_key)
        for rank, amount in enumerate(record.bytes_per_rank):
            self._reserved[operation.target][rank] -= amount
        updated = ResidencyRecord(
            key=record.key,
            bytes_per_rank=record.bytes_per_rank,
            tier=record.tier,
            lock_count=record.lock_count,
            last_access=record.last_access,
        )
        self._records[record.key] = updated
        self._version += 1
        return updated

    def usage(self, tier: MemoryTier) -> tuple[int, ...]:
        self._require_tier(tier)
        return tuple(self._used[tier])

    def reserved(self, tier: MemoryTier) -> tuple[int, ...]:
        self._require_tier(tier)
        return tuple(self._reserved[tier])

    def available(self, tier: MemoryTier) -> tuple[int, ...]:
        self._require_tier(tier)
        return tuple(
            self.capacities[tier][rank]
            - self._used[tier][rank]
            - self._reserved[tier][rank]
            for rank in range(self.num_ranks)
        )

    def snapshot(self) -> ResidencySnapshot:
        return ResidencySnapshot(
            version=self._version,
            records=MappingProxyType(dict(self._records)),
            used_bytes=MappingProxyType(
                {tier: tuple(values) for tier, values in self._used.items()}
            ),
            reserved_bytes=MappingProxyType(
                {tier: tuple(values) for tier, values in self._reserved.items()}
            ),
        )

