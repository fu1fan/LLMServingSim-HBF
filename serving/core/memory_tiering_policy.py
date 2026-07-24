"""基于真实驻留状态生成确定性的分层存储策略决策。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .memory_tiering import (
    MemoryObjectKey,
    MemoryObjectKind,
    MemoryTier,
    ResidencyState,
    TieredResidencyManager,
    TieringError,
    TransferOperation,
)
from .memory_tiering_config import MemoryTieringConfig


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


def _key_order(key):
    return (
        key.kind.value,
        key.object_id,
        -1 if key.layer_index is None else key.layer_index,
    )


class TieringPolicyEngine:
    """把静态与自适应策略落实到容量账本和迁移事务。"""

    def __init__(self, config, residency):
        if not isinstance(config, MemoryTieringConfig):
            raise TypeError("config 必须是 MemoryTieringConfig")
        if not isinstance(residency, TieredResidencyManager):
            raise TypeError("residency 必须是 TieredResidencyManager")
        if MemoryTier.HBM not in residency.capacities:
            raise TieringPolicyError("策略引擎要求配置 HBM 容量")
        if config.enabled and MemoryTier.HBF not in residency.capacities:
            raise TieringPolicyError("启用 HBF 策略时必须配置 HBF 容量")
        self.config = config
        self.residency = residency

    def _validate_sizes(self, values):
        _validate_bytes_per_rank(values, "bytes_per_rank")
        if len(values) != self.residency.num_ranks:
            raise TieringPolicyError(
                f"bytes_per_rank 必须包含 {self.residency.num_ranks} 个 rank"
            )

    def _fits(self, tier, values, *, extra_by_rank=None):
        self._validate_sizes(values)
        if tier not in self.residency.capacities:
            return False
        extra = extra_by_rank or (0,) * self.residency.num_ranks
        return all(
            values[rank] + extra[rank]
            <= self.residency.available(tier)[rank]
            for rank in range(self.residency.num_ranks)
        )

    def _projected_ratio(self, tier, extra_by_rank=None):
        extra = extra_by_rank or (0,) * self.residency.num_ranks
        used = self.residency.usage(tier)
        reserved = self.residency.reserved(tier)
        return tuple(
            (used[rank] + reserved[rank] + extra[rank])
            / self.residency.capacities[tier][rank]
            for rank in range(self.residency.num_ranks)
        )

    def _register(self, decision):
        try:
            self.residency.register(
                decision.key,
                decision.bytes_per_rank,
                decision.target,
            )
        except TieringError as exc:
            raise TieringPolicyError(str(exc)) from exc

    def _register_group(self, decisions):
        totals = {
            tier: [0] * self.residency.num_ranks
            for tier in self.residency.capacities
        }
        seen = set()
        records = self.residency.snapshot().records
        for decision in decisions:
            if decision.key in seen:
                raise TieringPolicyError(f"重复放置对象 {decision.key}")
            seen.add(decision.key)
            if decision.key in records:
                raise TieringPolicyError(f"对象 {decision.key} 已经登记")
            if decision.target not in totals:
                raise TieringPolicyError(
                    f"未配置 {decision.target.value} 容量"
                )
            self._validate_sizes(decision.bytes_per_rank)
            for rank, amount in enumerate(decision.bytes_per_rank):
                totals[decision.target][rank] += amount
        for tier, values in totals.items():
            available = self.residency.available(tier)
            for rank, amount in enumerate(values):
                if amount > available[rank]:
                    raise TieringPolicyError(
                        f"{tier.value} rank {rank} 容量不足："
                        f"需要 {amount} 字节，可用 {available[rank]} 字节"
                    )
        for decision in decisions:
            self._register(decision)

    @staticmethod
    def _static_mode(tier):
        if tier is MemoryTier.HBM:
            return PlacementMode.STATIC_HBM
        if tier is MemoryTier.HBF:
            return PlacementMode.STATIC_HBF
        raise TieringPolicyError("运行就绪 Profile 的静态放置只支持 HBM/HBF")

    def place_weights(self, requests):
        """按 canonical layer/block 规则原子预检并登记静态权重。"""

        requests = tuple(requests)
        if not requests:
            raise TieringPolicyError("权重放置请求不能为空")
        if not all(isinstance(item, WeightPlacementRequest) for item in requests):
            raise TypeError("权重放置请求必须是 WeightPlacementRequest")
        policy = self.config.weights
        decisions = []
        for request in sorted(requests, key=lambda item: _key_order(item.key)):
            tier = policy.tier_for(
                request.layer_name,
                request.block_index,
            )
            if policy.policy == "hbf_backed_hbm_cache":
                if tier is not MemoryTier.HBF:
                    raise TieringPolicyError(
                        "hbf_backed_hbm_cache 权重必须先登记到 HBF"
                    )
                mode = PlacementMode.HBF_FIRST
            else:
                mode = self._static_mode(tier)
            decisions.append(
                PolicyDecision(
                    request.key,
                    request.bytes_per_rank,
                    mode,
                    PolicyAction.REGISTER,
                    None,
                    tier,
                    f"weight_{policy.policy}",
                )
            )
        decisions = tuple(decisions)
        self._register_group(decisions)
        return TieringPolicyPlan(decisions)

    def _plan_migrations(
        self,
        keys,
        target,
        *,
        mode,
        reason,
        earliest_after,
        required_before,
    ):
        keys = tuple(keys)
        if not all(isinstance(key, MemoryObjectKey) for key in keys):
            raise TypeError("keys 必须只包含 MemoryObjectKey")
        if len(keys) != len(set(keys)):
            raise TieringPolicyError("同一迁移计划不能重复包含对象")
        keys = tuple(sorted(keys, key=_key_order))
        if not keys:
            return TieringPolicyPlan(())
        requests = []
        decisions = []
        target_total = [0] * self.residency.num_ranks
        for key in keys:
            record = self.residency.record(key)
            if record.state is not ResidencyState.RESIDENT:
                raise TieringPolicyError(f"对象 {key} 已在迁移")
            if record.lock_count:
                raise TieringPolicyError(f"对象 {key} 被锁定，不能迁移")
            if record.tier is target:
                decisions.append(
                    PolicyDecision(
                        key,
                        record.bytes_per_rank,
                        mode,
                        PolicyAction.KEEP,
                        target,
                        target,
                        f"{reason}_already_resident",
                    )
                )
                continue
            for rank, amount in enumerate(record.bytes_per_rank):
                target_total[rank] += amount
            requests.append(
                (
                    key,
                    target,
                    reason,
                    earliest_after,
                    required_before,
                )
            )
        if requests and not self._fits(target, tuple(target_total)):
            raise TieringPolicyError(
                f"{target.value} 容量不足，无法规划 {reason}"
            )
        try:
            transfers = self.residency.plan_transfers(requests)
        except (TieringError, ValueError) as exc:
            raise TieringPolicyError(str(exc)) from exc
        transfer_by_key = {
            operation.object_key: operation
            for operation in transfers
        }
        for key in keys:
            operation = transfer_by_key.get(key)
            if operation is None:
                continue
            decisions.append(
                PolicyDecision(
                    key,
                    operation.bytes_per_rank,
                    mode,
                    PolicyAction.MIGRATE,
                    operation.source,
                    operation.target,
                    reason,
                )
            )
        decisions.sort(key=lambda item: _key_order(item.key))
        return TieringPolicyPlan(tuple(decisions), transfers)

    def plan_weight_promotions(
        self,
        keys,
        *,
        earliest_after="batch_start",
        required_before="batch_end",
    ):
        """为 HBF-backed 权重生成显式 HBF→HBM 预取。"""

        policy = self.config.weights
        if policy.policy != "hbf_backed_hbm_cache":
            raise TieringPolicyError(
                "只有 hbf_backed_hbm_cache 支持权重提升"
            )
        keys = tuple(keys)
        for key in keys:
            if not isinstance(key, MemoryObjectKey):
                raise TypeError("keys 必须只包含 MemoryObjectKey")
            if key.kind is not MemoryObjectKind.WEIGHT:
                raise TieringPolicyError("权重提升只能处理 WEIGHT 对象")
        plan = self._plan_migrations(
            keys,
            MemoryTier.HBM,
            mode=PlacementMode.HBF_FIRST,
            reason="weight_prefetch",
            earliest_after=earliest_after,
            required_before=required_before,
        )
        # plan_transfers 已经把目标容量计入 reserved，不能再次叠加。
        ratios = self._projected_ratio(MemoryTier.HBM)
        if any(value > policy.hbm_high_watermark for value in ratios):
            for operation in plan.transfers:
                self.residency.abort_transfer(operation.transfer_id)
            raise TieringPolicyError("权重提升会超过 HBM high watermark")
        return plan

    def _lru_candidates(self, kind, *, exclude=()):
        excluded = set(exclude)
        candidates = []
        for record in self.residency.snapshot().records.values():
            if (
                record.key.kind is kind
                and record.key not in excluded
                and record.tier is MemoryTier.HBM
                and record.state is ResidencyState.RESIDENT
                and not record.lock_count
            ):
                candidates.append(record)
        candidates.sort(
            key=lambda record: (
                record.last_access,
                _key_order(record.key),
            )
        )
        return candidates

    def _plan_lru_demotions(
        self,
        kind,
        *,
        high,
        low,
        extra_hbm,
        reason,
        exclude=(),
        earliest_after="batch_start",
        required_before="batch_end",
    ):
        if MemoryTier.HBF not in self.residency.capacities:
            raise TieringPolicyError("LRU 降级要求配置 HBF 容量")
        projected = [
            self.residency.usage(MemoryTier.HBM)[rank]
            + self.residency.reserved(MemoryTier.HBM)[rank]
            + extra_hbm[rank]
            for rank in range(self.residency.num_ranks)
        ]
        capacities = self.residency.capacities[MemoryTier.HBM]
        if all(
            projected[rank] / capacities[rank] <= high
            for rank in range(self.residency.num_ranks)
        ):
            return TieringPolicyPlan(())
        low_limits = [
            int(capacity * low)
            for capacity in capacities
        ]
        hbf_available = list(self.residency.available(MemoryTier.HBF))
        selected = []
        for record in self._lru_candidates(kind, exclude=exclude):
            if all(
                projected[rank] <= low_limits[rank]
                for rank in range(self.residency.num_ranks)
            ):
                break
            if any(
                amount > hbf_available[rank]
                for rank, amount in enumerate(record.bytes_per_rank)
            ):
                continue
            selected.append(record.key)
            for rank, amount in enumerate(record.bytes_per_rank):
                projected[rank] -= amount
                hbf_available[rank] -= amount
        if any(
            projected[rank] > low_limits[rank]
            for rank in range(self.residency.num_ranks)
        ):
            raise TieringPolicyError(
                f"{reason} 无法把 HBM 使用量降到 low watermark"
            )
        return self._plan_migrations(
            selected,
            MemoryTier.HBF,
            mode=PlacementMode.ADAPTIVE,
            reason=reason,
            earliest_after=earliest_after,
            required_before=required_before,
        )

    def rebalance_weights(self):
        policy = self.config.weights
        if policy.policy != "hbf_backed_hbm_cache":
            raise TieringPolicyError(
                "只有 hbf_backed_hbm_cache 支持权重 LRU 降级"
            )
        return self._plan_lru_demotions(
            MemoryObjectKind.WEIGHT,
            high=policy.hbm_high_watermark,
            low=policy.hbm_low_watermark,
            extra_hbm=(0,) * self.residency.num_ranks,
            reason="weight_watermark_lru",
        )

    def _register_cache(self, request, tier, mode, reason):
        decision = PolicyDecision(
            request.key,
            request.bytes_per_rank,
            mode,
            PolicyAction.REGISTER,
            None,
            tier,
            reason,
        )
        self._register_group((decision,))
        self.residency.touch(request.key)
        return TieringPolicyPlan((decision,))

    def _fallback_tier(self, preferred):
        fallback = self.config.transfer.capacity_fallback
        if fallback == "reject":
            return None
        tier = MemoryTier(fallback)
        if tier not in (MemoryTier.HBM, MemoryTier.HBF):
            raise TieringPolicyError(
                "HBF Profile 策略不能把 demand 对象直接放到 CPU/CXL；"
                "应由显式 offload/staging 路径处理"
            )
        if tier is preferred:
            return None
        return tier

    def admit_kv(self, request):
        """按 KV 策略登记新对象，或返回先降级后登记的两阶段计划。"""

        if not isinstance(request, CachePlacementRequest):
            raise TypeError("request 必须是 CachePlacementRequest")
        if request.key.kind is not MemoryObjectKind.KV:
            raise TieringPolicyError("admit_kv 只能处理 KV 对象")
        if request.key in self.residency.snapshot().records:
            record = self.residency.record(request.key)
            if record.state is not ResidencyState.RESIDENT:
                raise TieringPolicyError("迁移中的 KV 对象不能进入 demand 访问")
            if record.bytes_per_rank != request.bytes_per_rank:
                raise TieringPolicyError("已登记 KV 对象的字节数不能改变")
            if (
                self.config.kv.policy == "hbm_only"
                and record.tier is not MemoryTier.HBM
            ):
                raise TieringPolicyError("hbm_only KV 对象实际不在 HBM")
            if (
                self.config.kv.policy == "hbf_only"
                and record.tier is not MemoryTier.HBF
            ):
                raise TieringPolicyError("hbf_only KV 对象实际不在 HBF")
            self.residency.touch(request.key)
            if self.config.kv.policy == "hbm_only":
                mode = PlacementMode.STATIC_HBM
            elif self.config.kv.policy == "hbf_only":
                mode = PlacementMode.STATIC_HBF
            else:
                mode = PlacementMode.ADAPTIVE
            return TieringPolicyPlan(
                (
                    PolicyDecision(
                        request.key,
                        request.bytes_per_rank,
                        mode,
                        PolicyAction.KEEP,
                        record.tier,
                        record.tier,
                        "kv_existing",
                    ),
                )
            )

        policy = self.config.kv
        if policy.policy == "hbm_only":
            return self._register_cache(
                request,
                MemoryTier.HBM,
                PlacementMode.STATIC_HBM,
                "kv_hbm_only",
            )
        if policy.policy == "hbf_only":
            return self._register_cache(
                request,
                MemoryTier.HBF,
                PlacementMode.STATIC_HBF,
                "kv_hbf_only",
            )
        if policy.policy == "length_threshold":
            target = (
                MemoryTier.HBF
                if request.token_count >= policy.threshold_tokens
                else policy.admission_tier
            )
            if self._fits(target, request.bytes_per_rank):
                return self._register_cache(
                    request,
                    target,
                    PlacementMode.ADAPTIVE,
                    "kv_length_threshold",
                )
            fallback = self._fallback_tier(target)
            if fallback is not None and self._fits(
                fallback,
                request.bytes_per_rank,
            ):
                return self._register_cache(
                    request,
                    fallback,
                    PlacementMode.ADAPTIVE,
                    "kv_capacity_fallback",
                )
            raise TieringPolicyError(
                f"{target.value} 容量不足，无法登记 KV 对象"
            )
        if policy.policy != "watermark_lru":
            raise TieringPolicyError(f"未知 KV 策略 {policy.policy!r}")

        target = policy.admission_tier
        if target is not MemoryTier.HBM:
            return self._register_cache(
                request,
                target,
                PlacementMode.ADAPTIVE,
                "kv_watermark_admission",
            )
        projected = self._projected_ratio(
            MemoryTier.HBM,
            request.bytes_per_rank,
        )
        if (
            self._fits(MemoryTier.HBM, request.bytes_per_rank)
            and all(
                value <= policy.hbm_high_watermark
                for value in projected
            )
        ):
            return self._register_cache(
                request,
                MemoryTier.HBM,
                PlacementMode.HBM_FIRST,
                "kv_hbm_first",
            )
        try:
            demotions = self._plan_lru_demotions(
                MemoryObjectKind.KV,
                high=policy.hbm_high_watermark,
                low=policy.hbm_low_watermark,
                extra_hbm=request.bytes_per_rank,
                reason="kv_watermark_lru",
                earliest_after=request.earliest_after,
                required_before=request.required_before,
            )
        except TieringPolicyError:
            if self._fits(MemoryTier.HBF, request.bytes_per_rank):
                return self._register_cache(
                    request,
                    MemoryTier.HBF,
                    PlacementMode.HBM_FIRST,
                    "kv_hbm_pressure_spill",
                )
            raise
        deferred = PolicyDecision(
            request.key,
            request.bytes_per_rank,
            PlacementMode.ADAPTIVE,
            PolicyAction.DEFERRED_REGISTER,
            None,
            MemoryTier.HBM,
            "kv_after_lru_demotion",
        )
        return TieringPolicyPlan(
            demotions.decisions + (deferred,),
            demotions.transfers,
        )

    def complete_deferred(self, plan):
        """迁移完成后登记计划中的延迟放置对象。"""

        if not isinstance(plan, TieringPolicyPlan):
            raise TypeError("plan 必须是 TieringPolicyPlan")
        for operation in plan.transfers:
            record = self.residency.record(operation.object_key)
            if (
                record.state is not ResidencyState.RESIDENT
                or record.tier is not operation.target
            ):
                raise TieringPolicyError(
                    "延迟放置要求先提交计划中的全部迁移"
                )
        deferred = tuple(
            item
            for item in plan.decisions
            if item.action is PolicyAction.DEFERRED_REGISTER
        )
        if not deferred:
            raise TieringPolicyError("计划不包含延迟放置对象")
        registered = tuple(
            PolicyDecision(
                item.key,
                item.bytes_per_rank,
                item.mode,
                PolicyAction.REGISTER,
                None,
                item.target,
                f"{item.reason}_completed",
            )
            for item in deferred
        )
        self._register_group(registered)
        for item in registered:
            self.residency.touch(item.key)
        return TieringPolicyPlan(registered)

    def admit_prefix(self, request):
        """按 Prefix 策略首次登记缓存对象。"""

        if not isinstance(request, CachePlacementRequest):
            raise TypeError("request 必须是 CachePlacementRequest")
        if request.key.kind is not MemoryObjectKind.PREFIX:
            raise TieringPolicyError("admit_prefix 只能处理 PREFIX 对象")
        if request.key in self.residency.snapshot().records:
            return self.access_prefix(
                request.key,
                hit_count=request.hit_count,
                earliest_after=request.earliest_after,
                required_before=request.required_before,
            )
        policy = self.config.prefix
        if policy.policy == "hbm_only":
            return self._register_cache(
                request,
                MemoryTier.HBM,
                PlacementMode.STATIC_HBM,
                "prefix_hbm_only",
            )
        if policy.policy == "hbf_only":
            return self._register_cache(
                request,
                MemoryTier.HBF,
                PlacementMode.STATIC_HBF,
                "prefix_hbf_only",
            )
        if policy.policy == "hbf_backed_hbm_hot":
            return self._register_cache(
                request,
                MemoryTier.HBF,
                PlacementMode.HBF_FIRST,
                "prefix_hbf_backing",
            )
        if policy.policy != "instance_affinity":
            raise TieringPolicyError(f"未知 Prefix 策略 {policy.policy!r}")
        hbm_ratio = self._projected_ratio(
            MemoryTier.HBM,
            request.bytes_per_rank,
        )
        if (
            self._fits(MemoryTier.HBM, request.bytes_per_rank)
            and all(
                value <= policy.hbm_high_watermark
                for value in hbm_ratio
            )
        ):
            return self._register_cache(
                request,
                MemoryTier.HBM,
                PlacementMode.HBM_FIRST,
                "prefix_instance_affinity_hbm",
            )
        if self._fits(MemoryTier.HBF, request.bytes_per_rank):
            return self._register_cache(
                request,
                MemoryTier.HBF,
                PlacementMode.HBM_FIRST,
                "prefix_instance_affinity_hbf",
            )
        raise TieringPolicyError("HBM/HBF 均无法容纳 Prefix 对象")

    def access_prefix(
        self,
        key,
        *,
        hit_count,
        earliest_after="batch_start",
        required_before="batch_end",
    ):
        """记录 Prefix 命中，并按 hot threshold 规划可选提升。"""

        if not isinstance(key, MemoryObjectKey):
            raise TypeError("key 必须是 MemoryObjectKey")
        if key.kind is not MemoryObjectKind.PREFIX:
            raise TieringPolicyError("access_prefix 只能处理 PREFIX 对象")
        if (
            isinstance(hit_count, bool)
            or not isinstance(hit_count, int)
            or hit_count < 0
        ):
            raise ValueError("hit_count 必须是非负整数")
        record = self.residency.record(key)
        if record.state is not ResidencyState.RESIDENT:
            raise TieringPolicyError("迁移中的 Prefix 对象不能进入 demand 访问")
        policy = self.config.prefix
        if (
            policy.policy == "hbm_only"
            and record.tier is not MemoryTier.HBM
        ):
            raise TieringPolicyError("hbm_only Prefix 对象实际不在 HBM")
        if (
            policy.policy == "hbf_only"
            and record.tier is not MemoryTier.HBF
        ):
            raise TieringPolicyError("hbf_only Prefix 对象实际不在 HBF")
        record = self.residency.touch(key)
        if (
            policy.policy != "hbf_backed_hbm_hot"
            or record.tier is not MemoryTier.HBF
            or hit_count < policy.promotion_hits
        ):
            mode = (
                PlacementMode.STATIC_HBM
                if record.tier is MemoryTier.HBM
                else PlacementMode.STATIC_HBF
            )
            if policy.policy == "hbf_backed_hbm_hot":
                mode = PlacementMode.HBF_FIRST
            elif policy.policy == "instance_affinity":
                mode = PlacementMode.HBM_FIRST
            return TieringPolicyPlan(
                (
                    PolicyDecision(
                        key,
                        record.bytes_per_rank,
                        mode,
                        PolicyAction.KEEP,
                        record.tier,
                        record.tier,
                        "prefix_keep",
                    ),
                )
            )
        projected = self._projected_ratio(
            MemoryTier.HBM,
            record.bytes_per_rank,
        )
        if (
            not self._fits(MemoryTier.HBM, record.bytes_per_rank)
            or any(
                value > policy.hbm_high_watermark
                for value in projected
            )
        ):
            return TieringPolicyPlan(
                (
                    PolicyDecision(
                        key,
                        record.bytes_per_rank,
                        PlacementMode.HBF_FIRST,
                        PolicyAction.KEEP,
                        record.tier,
                        record.tier,
                        "prefix_promotion_deferred_capacity",
                    ),
                )
            )
        return self._plan_migrations(
            (key,),
            MemoryTier.HBM,
            mode=PlacementMode.HBF_FIRST,
            reason="prefix_hot_promotion",
            earliest_after=earliest_after,
            required_before=required_before,
        )

    def rebalance_prefix(self):
        policy = self.config.prefix
        if policy.policy != "hbf_backed_hbm_hot":
            raise TieringPolicyError(
                "只有 hbf_backed_hbm_hot 支持 Prefix LRU 降级"
            )
        return self._plan_lru_demotions(
            MemoryObjectKind.PREFIX,
            high=policy.hbm_high_watermark,
            low=policy.hbm_low_watermark,
            extra_hbm=(0,) * self.residency.num_ranks,
            reason="prefix_watermark_lru",
        )
