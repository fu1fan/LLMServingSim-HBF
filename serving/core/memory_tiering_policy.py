"""基于真实驻留状态生成确定性的分层存储策略决策。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .memory_tiering import (
    MemoryObjectKey,
    MemoryObjectKind,
    MemoryTier,
    TransferOperation,
)


class TieringPolicyError(RuntimeError):
    """策略输入、容量或状态不允许形成完整决策。"""


class PlacementMode(str, Enum):
    """策略解析后的通用放置行为。"""

    STATIC_HBM = "static_hbm"
    STATIC_HBF = "static_hbf"
    HBM_FIRST = "hbm_first"
    HBF_FIRST = "hbf_first"
    ADAPTIVE = "adaptive"


class PolicyAction(str, Enum):
    """策略引擎对一个对象产生的状态变更。"""

    REGISTER = "register"
    KEEP = "keep"
    MIGRATE = "migrate"
    DEFERRED_REGISTER = "deferred_register"


def _nonempty_string(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须是非空字符串")
    return value.strip()


def _validate_bytes_per_rank(values, field):
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{field} 必须是非空 tuple")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} 必须只包含非负整数")
    if not any(values):
        raise ValueError(f"{field} 不能全部为零")


@dataclass(frozen=True)
class WeightPlacementRequest:
    """一个 canonical 权重对象的静态放置请求。"""

    key: MemoryObjectKey
    bytes_per_rank: tuple[int, ...]
    layer_name: str
    block_index: int | None = None

    def __post_init__(self):
        if not isinstance(self.key, MemoryObjectKey):
            raise TypeError("key 必须是 MemoryObjectKey")
        if self.key.kind is not MemoryObjectKind.WEIGHT:
            raise ValueError("权重放置请求必须使用 WEIGHT 对象")
        _validate_bytes_per_rank(self.bytes_per_rank, "bytes_per_rank")
        object.__setattr__(
            self,
            "layer_name",
            _nonempty_string(self.layer_name, "layer_name"),
        )
        if self.block_index is not None and (
            isinstance(self.block_index, bool)
            or not isinstance(self.block_index, int)
            or self.block_index < 0
        ):
            raise ValueError("block_index 必须是非负整数或 None")
        if (
            self.key.layer_index is not None
            and self.key.layer_index != self.block_index
        ):
            raise ValueError("key.layer_index 必须与 block_index 一致")


@dataclass(frozen=True)
class CachePlacementRequest:
    """一个 KV 或 Prefix 对象的放置与访问上下文。"""

    key: MemoryObjectKey
    bytes_per_rank: tuple[int, ...]
    token_count: int
    hit_count: int = 0
    earliest_after: str = "batch_start"
    required_before: str = "batch_end"

    def __post_init__(self):
        if not isinstance(self.key, MemoryObjectKey):
            raise TypeError("key 必须是 MemoryObjectKey")
        if self.key.kind not in (MemoryObjectKind.KV, MemoryObjectKind.PREFIX):
            raise ValueError("缓存请求必须使用 KV 或 PREFIX 对象")
        _validate_bytes_per_rank(self.bytes_per_rank, "bytes_per_rank")
        if (
            isinstance(self.token_count, bool)
            or not isinstance(self.token_count, int)
            or self.token_count <= 0
        ):
            raise ValueError("token_count 必须是正整数")
        if (
            isinstance(self.hit_count, bool)
            or not isinstance(self.hit_count, int)
            or self.hit_count < 0
        ):
            raise ValueError("hit_count 必须是非负整数")
        object.__setattr__(
            self,
            "earliest_after",
            _nonempty_string(self.earliest_after, "earliest_after"),
        )
        object.__setattr__(
            self,
            "required_before",
            _nonempty_string(self.required_before, "required_before"),
        )


@dataclass(frozen=True)
class PolicyDecision:
    """策略对单个对象的确定性决策，不包含 Profile demand 访存。"""

    key: MemoryObjectKey
    bytes_per_rank: tuple[int, ...]
    mode: PlacementMode
    action: PolicyAction
    source: MemoryTier | None
    target: MemoryTier
    reason: str

    def __post_init__(self):
        if not isinstance(self.key, MemoryObjectKey):
            raise TypeError("key 必须是 MemoryObjectKey")
        _validate_bytes_per_rank(self.bytes_per_rank, "bytes_per_rank")
        if not isinstance(self.mode, PlacementMode):
            raise TypeError("mode 必须是 PlacementMode")
        if not isinstance(self.action, PolicyAction):
            raise TypeError("action 必须是 PolicyAction")
        if self.source is not None and not isinstance(self.source, MemoryTier):
            raise TypeError("source 必须是 MemoryTier 或 None")
        if not isinstance(self.target, MemoryTier):
            raise TypeError("target 必须是 MemoryTier")
        object.__setattr__(
            self,
            "reason",
            _nonempty_string(self.reason, "reason"),
        )
        if self.action is PolicyAction.REGISTER and self.source is not None:
            raise ValueError("REGISTER 决策不能包含 source")
        if self.action is PolicyAction.DEFERRED_REGISTER and self.source is not None:
            raise ValueError("DEFERRED_REGISTER 决策不能包含 source")
        if self.action in (PolicyAction.KEEP, PolicyAction.MIGRATE):
            if self.source is None:
                raise ValueError(f"{self.action.value} 决策必须包含 source")
        if self.action is PolicyAction.KEEP and self.source is not self.target:
            raise ValueError("KEEP 决策的 source 和 target 必须相同")
        if self.action is PolicyAction.MIGRATE and self.source is self.target:
            raise ValueError("MIGRATE 决策必须改变内存层级")


@dataclass(frozen=True)
class TieringPolicyPlan:
    """已经容量校验的放置决策及显式迁移操作。"""

    decisions: tuple[PolicyDecision, ...]
    transfers: tuple[TransferOperation, ...] = ()

    def __post_init__(self):
        if not isinstance(self.decisions, tuple):
            raise TypeError("decisions 必须是 tuple")
        if not isinstance(self.transfers, tuple):
            raise TypeError("transfers 必须是 tuple")
        if not all(
            isinstance(item, PolicyDecision)
            for item in self.decisions
        ):
            raise TypeError("decisions 必须只包含 PolicyDecision")
        if not all(
            isinstance(item, TransferOperation)
            for item in self.transfers
        ):
            raise TypeError("transfers 必须只包含 TransferOperation")

    @property
    def explicit_transfer_bytes(self):
        """只统计策略显式搬运，不包含算子 Profile 已计入的 demand access。"""

        return sum(item.total_bytes for item in self.transfers)
