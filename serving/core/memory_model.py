import os, threading
from dataclasses import dataclass
from types import MappingProxyType
from .utils import get_config
from .radix_tree import *
from .memory_tiering import MemoryObjectKind, MemoryTier
from .memory_tiering_stats import MemoryTieringStats
from .residency_scenario import BatchMemoryView
import logging
from enum import Enum

GB_TO_BYTE = 1024 * 1024 * 1024
MB_TO_BYTE = 1024 * 1024
KB_TO_BYTE = 1024

class Device(Enum):
    NPU = 1
    CPU = 2
    CXL = 3
    HBF = 4


@dataclass(frozen=True)
class KVLayerResidency:
    """一个请求在单个 Transformer 层的完整 KV 驻留。"""

    request_id: str
    layer_index: int
    tier: MemoryTier
    allocated_tokens: int
    bytes_per_rank: int


@dataclass(frozen=True)
class KVTransferEvent:
    """交给显式迁移链路计时的 KV 整层搬运事件。"""

    request_id: str
    layer_index: int
    pp_stage: int
    source: MemoryTier
    target: MemoryTier
    bytes_per_rank: int
    reason: str


@dataclass(frozen=True)
class KVAllocationPlan:
    """一次调度候选的只读 KV 容量事务。"""

    base_version: int
    records: tuple[KVLayerResidency, ...]
    removed_keys: tuple[tuple[str, int], ...]
    used_delta: MappingProxyType
    growth_bytes_per_rank: int
    transfers: tuple[KVTransferEvent, ...]

class MemoryModel():
    def __init__(self, model, instance_id, node_id, num_npus, tp_size, npu_mem, cpu_mem, block_size, fp, enable_prefix_caching, enable_prefix_sharing, prefix_pool, prefix_storage, cxl_mem=0, ep_size=1, pp_size=1, kv_cache_dtype='auto', memory_tiering=None, kv_policy_engine=None):
        self.model = model
        self.node_id = node_id
        self.instance_id = instance_id
        self.num_npus = num_npus
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.ep_size = ep_size
        self.npu_mem = npu_mem * GB_TO_BYTE # GB -> Byte
        self.cpu_mem = cpu_mem * GB_TO_BYTE # GB -> Byte
        self.cxl_mem = cxl_mem * GB_TO_BYTE
        self.block_size = block_size
        self.fp = fp // 8 # bit -> byte of floating point
        self.kv_fp = 1 if kv_cache_dtype == 'fp8' else self.fp  # KV cache bytes per element
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_prefix_sharing = enable_prefix_sharing
        self.prefix_storage = prefix_storage
        self.memory_tiering = memory_tiering
        self.tiering_enabled = bool(
            memory_tiering is not None and memory_tiering.enabled
        )
        self.kv_policy_engine = kv_policy_engine
        if self.tiering_enabled and self.num_npus != self.tp_size * self.pp_size:
            raise RuntimeError(
                "HBF 分层分账要求 num_npus == tp_size * pp_size"
            )

        self.config = get_config(model)
        self.n_embd = self.config['hidden_size']
        self.n_layer = self.config['num_hidden_layers']
        self.n_head = self.config['num_attention_heads']
        self.head_dim = self.config.get('head_dim', self.n_embd // self.n_head)
        self.kv_head = self.config.get("num_key_value_heads", self.n_head)  # fallback to n_head if not defined
        self.q_dim = self.n_head * self.head_dim       # total Q projection output dim
        self.kv_dim = self.kv_head * self.head_dim     # total KV projection output dim
        self.vocab_size = self.config['vocab_size']
        # Accept either the Mistral-style ``num_local_experts`` or the
        # HF/Qwen-style ``num_experts`` key — profiler configs track
        # upstream HF naming which varies per family.
        self.is_moe = 'num_local_experts' in self.config or 'num_experts' in self.config

        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)

        # Memory model
        self.weight = self.get_weight() # assume weight is loaded
        self.hbf_mem = (
            memory_tiering.hbf_capacity_bytes
            if self.tiering_enabled
            else 0
        )
        self._weight_tiers = {}
        self._kv_records = {}
        self._kv_version = 0
        self._kv_transfer_events = []

        if self.tiering_enabled:
            self._init_tiered_weight_accounting()
        else:
            self.hbm_weight = self.weight
            self.hbf_weight = 0
            self._hbm_used_by_rank = [self.weight] * self.num_npus
            self._hbf_used_by_rank = [0] * self.num_npus
            self._hbm_weight_by_rank = tuple(self._hbm_used_by_rank)
            self._hbf_weight_by_rank = tuple(self._hbf_used_by_rank)

        self.npu_used = max(self._hbm_used_by_rank, default=0)
        self.hbf_used = max(self._hbf_used_by_rank, default=0)
        self.cpu_used = 0
        if self.npu_used > self.npu_mem:
            raise RuntimeError(f"[MemoryModel] [node={self.node_id},inst={self.instance_id}]: Model size {self.weight*self.num_npus//GB_TO_BYTE}GB exceeds total NPU memory {self.npu_mem*self.num_npus//GB_TO_BYTE}GB")
        if self.hbf_used > self.hbf_mem:
            raise RuntimeError(
                f"[MemoryModel] [node={self.node_id},inst={self.instance_id}]: "
                f"HBF weight {self.hbf_used / GB_TO_BYTE:.2f}GB exceeds "
                f"HBF memory {self.hbf_mem / GB_TO_BYTE:.2f}GB"
            )
        self.tiering_stats = (
            MemoryTieringStats(num_ranks=self.num_npus)
            if self.tiering_enabled
            else None
        )
        if self.tiering_stats is not None:
            self.tiering_stats.record_policy_action(
                "register",
                count=len(self._weight_tiers),
            )
            self._observe_tiering_usage()

        if self.tiering_enabled and enable_prefix_caching:
            if (
                memory_tiering.kv.policy != "hbm_only"
                or memory_tiering.prefix.policy != "hbm_only"
            ):
                raise RuntimeError(
                    "HBF KV/Prefix 策略尚未接入 RadixCache；"
                    "启用 Prefix Caching 时必须使用 hbm_only"
                )

        if enable_prefix_caching:
            one_token_kv_size = self.get_kv(1)
            self.mem_for_kv = self.npu_mem - self.hbm_weight
            self.npu_prefix_cache = RadixCache(device='NPU', 
                                               node_id=self.node_id,
                                               instance_id=self.instance_id,
                                               page_size=self.block_size,
                                               capacity=self.mem_for_kv,
                                               kv_size=one_token_kv_size,
                                               enable_kv_cache_events=True,
                                                )
            if prefix_storage is not None:
                if enable_prefix_sharing and prefix_pool is not None:
                    self.second_tier_prefix_cache = prefix_pool
                else:
                    prefix_cache_capacity = 0
                    if prefix_storage == Device.CPU:
                        device = "CPU"
                        prefix_cache_capacity = self.cpu_mem
                    elif prefix_storage == Device.CXL:
                        device = "CXL"
                        prefix_cache_capacity = self.cxl_mem
                    else:
                        raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}]: Device {prefix_storage} is currently not supported as a second tier prefix cache storage")
                    # print("[instance {}] prefix_cache_capacity : {}".format(instance_id, prefix_cache_capacity // GB_TO_BYTE))
                    self.second_tier_prefix_cache = RadixCache(device=device, 
                                                    node_id=self.node_id,
                                                    instance_id=self.instance_id,
                                                    page_size=1,
                                                    capacity=prefix_cache_capacity,
                                                    kv_size=(one_token_kv_size * self.num_npus),
                                                    enable_kv_cache_events=True,
                                                    )
                
        # Hash id -> token length for corresponding prefix cache block
        self._npu_cache_hashtolen = {}
        self._cpu_cache_hashtolen = {}
        self._bytes_per_token = self.get_kv(1)  # bytes per token for kv cache
    def get_weight(self):
        """Per-GPU model weight in bytes.

        Conservative upper bound across PP ranks: assumes a single rank
        holds embedding + final_layernorm + lm_head along with its share
        of transformer blocks (n_layer // pp_size). In real PP these
        non-block weights live on the first/last rank only, so middle
        ranks are lighter — but using the heaviest-rank value here keeps
        the `weight > npu_mem` check safe.
        """
        tp = self.tp_size
        pp = max(self.pp_size, 1)
        ep = self.ep_size
        fp = self.fp
        weight = 0

        _, embedding, _ = calculate_sizes(self.model, 'embedding', 1, parallel=tp, fp=fp)
        weight += embedding
        weight += self._get_weight_per_block(tp, ep, fp) * (self.n_layer // pp)
        _, ln_f, _ = calculate_sizes(self.model, 'final_layernorm', 1, parallel=tp, fp=fp)
        weight += ln_f
        _, lm_head, _ = calculate_sizes(self.model, 'lm_head', 1, parallel=tp, fp=fp)
        weight += lm_head

        self.logger.info(
            "NPU: model weight %dMB loaded",
            weight * tp // MB_TO_BYTE,
        )
        return weight

    def _get_weight_per_block(self, tp, ep, fp):
        """Per-block weight: dense layers use TP, MoE experts use EP."""
        block_weight = 0
        _, ln_w, _ = calculate_sizes(self.model, 'layernorm', 1, parallel=tp, fp=fp)
        block_weight += ln_w  # input layernorm
        _, qkv_w, _ = calculate_sizes(self.model, 'qkv_proj', 1, parallel=tp, fp=fp)
        block_weight += qkv_w
        _, o_w, _ = calculate_sizes(self.model, 'o_proj', 1, parallel=tp, fp=fp)
        block_weight += o_w
        block_weight += ln_w  # post layernorm (same weight size)
        if self.is_moe:
            _, moe_w, _ = calculate_sizes(self.model, 'moe', 1, parallel=ep, fp=fp)
            block_weight += moe_w
        else:
            _, ffn1_w, _ = calculate_sizes(self.model, 'gate_up_proj', 1, parallel=tp, fp=fp)
            block_weight += ffn1_w
            _, ffn2_w, _ = calculate_sizes(self.model, 'down_proj', 1, parallel=tp, fp=fp)
            block_weight += ffn2_w
        return block_weight

    def _canonical_weight_sizes(self):
        """返回 canonical 层名对应的单块、单 rank 权重字节数。"""

        tp = self.tp_size
        ep = self.ep_size
        fp = self.fp
        sizes = {}
        for name in ("embedding", "final_layernorm", "lm_head"):
            _, sizes[name], _ = calculate_sizes(
                self.model,
                name,
                1,
                parallel=tp,
                fp=fp,
            )

        _, layernorm, _ = calculate_sizes(
            self.model,
            "layernorm",
            1,
            parallel=tp,
            fp=fp,
        )
        sizes["layernorm"] = layernorm * 2
        for name in ("qkv_proj", "o_proj"):
            _, sizes[name], _ = calculate_sizes(
                self.model,
                name,
                1,
                parallel=tp,
                fp=fp,
            )
        if self.config.get("model_type") == "qwen3":
            _, sizes["qk_norm"], _ = calculate_sizes(
                self.model,
                "qk_norm",
                1,
                parallel=tp,
                fp=fp,
            )
        if self.is_moe:
            _, sizes["moe"], _ = calculate_sizes(
                self.model,
                "moe",
                1,
                parallel=ep,
                fp=fp,
            )
        else:
            for name in ("gate_up_proj", "down_proj"):
                _, sizes[name], _ = calculate_sizes(
                    self.model,
                    name,
                    1,
                    parallel=tp,
                    fp=fp,
                )
        return sizes

    def _pipeline_blocks(self, stage):
        """按连续 block 切分 PP stage；余数优先放在靠前 stage。"""

        base, remainder = divmod(self.n_layer, max(self.pp_size, 1))
        start = stage * base + min(stage, remainder)
        count = base + (1 if stage < remainder else 0)
        return range(start, start + count)

    def _pipeline_stage(self, layer_index):
        """返回 transformer layer 实际所属的 PP stage。"""

        for stage in range(max(self.pp_size, 1)):
            if layer_index in self._pipeline_blocks(stage):
                return stage
        raise RuntimeError(f"transformer layer {layer_index} 未分配到 PP stage")

    def _pipeline_rank_ids(self, layer_index):
        """KV 只占用所属 PP stage 内的 TP ranks。"""

        stage = self._pipeline_stage(layer_index)
        first = stage * self.tp_size
        return stage, range(first, first + self.tp_size)

    def _init_tiered_weight_accounting(self):
        """按真实 canonical 层和 PP block 为每个 rank 分账。"""

        policy = self.memory_tiering.weights
        sizes = self._canonical_weight_sizes()
        hbm = [0] * self.num_npus
        hbf = [0] * self.num_npus

        block_layers = (
            "layernorm",
            "qkv_proj",
            "qk_norm",
            "o_proj",
            "gate_up_proj",
            "act_fn",
            "down_proj",
            "moe",
            "rotary_emb",
            "attention",
        )
        shared_layers = (
            "embedding",
            "final_layernorm",
            "lm_head",
            "sampler",
        )
        for name in shared_layers:
            self._weight_tiers[(name, None)] = policy.tier_for(name, None)
        for block_index in range(self.n_layer):
            for name in block_layers:
                self._weight_tiers[(name, block_index)] = policy.tier_for(
                    name,
                    block_index,
                )

        for rank in range(self.num_npus):
            stage = (rank // max(self.tp_size, 1)) % max(self.pp_size, 1)
            rank_sizes = {
                MemoryTier.HBM: 0,
                MemoryTier.HBF: 0,
            }
            if stage == 0:
                tier = self._weight_tiers[("embedding", None)]
                rank_sizes[tier] += sizes["embedding"]
            if stage == self.pp_size - 1:
                for name in ("final_layernorm", "lm_head"):
                    tier = self._weight_tiers[(name, None)]
                    rank_sizes[tier] += sizes[name]

            for block_index in self._pipeline_blocks(stage):
                for name, size in sizes.items():
                    if name in {"embedding", "final_layernorm", "lm_head"}:
                        continue
                    tier = self._weight_tiers[(name, block_index)]
                    rank_sizes[tier] += size
            hbm[rank] = rank_sizes[MemoryTier.HBM]
            hbf[rank] = rank_sizes[MemoryTier.HBF]

        self._hbm_used_by_rank = hbm
        self._hbf_used_by_rank = hbf
        self._hbm_weight_by_rank = tuple(hbm)
        self._hbf_weight_by_rank = tuple(hbf)
        self.hbm_weight = max(hbm, default=0)
        self.hbf_weight = max(hbf, default=0)

    def get_kv(self, seq):
        # shape of kv cache
        # (kv_head, batch_size, n_embd//n_head, seq_len) per layer
        # return batch_size = 1 to caclulate max batch_size in scheduler

        # K & V multiply 2
        return 2 * self.kv_dim * seq * self.n_layer * self.kv_fp // self.num_npus

    def get_layer_kv(self, seq):
        """单请求单层在所属 PP stage 的每个 TP rank 上的 KV 字节。"""

        return 2 * self.kv_dim * seq * self.kv_fp // self.tp_size

    @staticmethod
    def _device_for_tier(tier):
        if tier is MemoryTier.HBM:
            return Device.NPU
        if tier is MemoryTier.HBF:
            return Device.HBF
        raise RuntimeError(f"KV demand residency 不支持 {tier.value}")

    @staticmethod
    def _tier_for_device(device):
        if device is Device.NPU:
            return MemoryTier.HBM
        if device is Device.HBF:
            return MemoryTier.HBF
        raise RuntimeError(f"{device} 不是 GPU demand memory")

    def _select_kv_tier(self, req, projected_tokens):
        if not self.tiering_enabled:
            return MemoryTier.HBM
        policy = self.memory_tiering.kv
        if policy.policy == "hbm_only":
            return MemoryTier.HBM
        if policy.policy == "hbf_only":
            return MemoryTier.HBF
        if policy.policy == "length_threshold":
            if projected_tokens < policy.threshold_tokens:
                return policy.admission_tier
            return MemoryTier.HBF
        if policy.policy == "watermark_lru":
            selector = getattr(self.kv_policy_engine, "select_kv_tier", None)
            if not callable(selector):
                raise RuntimeError(
                    "watermark_lru 需要提供实现 select_kv_tier() 的策略引擎"
                )
            tier = selector(
                request=req,
                projected_tokens=projected_tokens,
                memory_model=self,
            )
            if not isinstance(tier, MemoryTier):
                raise RuntimeError(
                    "select_kv_tier() 必须返回 MemoryTier"
                )
            if tier not in {MemoryTier.HBM, MemoryTier.HBF}:
                raise RuntimeError(
                    "KV 策略只能选择 HBM 或 HBF"
                )
            return tier
        raise RuntimeError(f"未实现的 KV 策略 {policy.policy!r}")

    def plan_kv_allocation(
        self,
        batch_req,
        batch_len=None,
        scheduled_tokens=None,
    ):
        """规划但不修改一次 batch 的 KV 增长和整层迁移。"""

        if not self.tiering_enabled:
            raise RuntimeError("普通 GPU 不需要 KV 分层事务")
        if batch_len is None:
            batch_len = len(batch_req)
        requests = batch_req[:batch_len]
        if len({str(req.id) for req in requests}) != len(requests):
            raise RuntimeError("同一 KV 事务不能重复包含 request")

        delta = {
            Device.NPU: [0] * self.num_npus,
            Device.HBF: [0] * self.num_npus,
        }
        growth = [0] * self.num_npus
        records = []
        transfers = []
        for req in requests:
            scheduled = (
                scheduled_tokens.get(req.id, 0)
                if scheduled_tokens is not None
                else 0
            )
            projected = req.num_computed_tokens + scheduled
            aligned = (
                (projected + self.block_size - 1) // self.block_size
                * self.block_size
                if projected > 0
                else 0
            )
            target = self._select_kv_tier(req, projected)
            target_device = self._device_for_tier(target)
            for layer_index in range(self.n_layer):
                pp_stage, rank_ids = self._pipeline_rank_ids(layer_index)
                rank_ids = tuple(rank_ids)
                key = (str(req.id), layer_index)
                current = self._kv_records.get(key)
                desired_bytes = self.get_layer_kv(aligned)
                if desired_bytes == 0:
                    continue
                if current is None:
                    for rank in rank_ids:
                        delta[target_device][rank] += desired_bytes
                        growth[rank] += desired_bytes
                elif current.tier is target:
                    change = desired_bytes - current.bytes_per_rank
                    for rank in rank_ids:
                        delta[target_device][rank] += change
                        growth[rank] += max(change, 0)
                else:
                    for rank in rank_ids:
                        # 迁移完成前源副本仍占容量，目标侧先做完整预留。
                        delta[target_device][rank] += desired_bytes
                        growth[rank] += max(
                            desired_bytes - current.bytes_per_rank,
                            0,
                        )
                    transfers.append(
                        KVTransferEvent(
                            request_id=str(req.id),
                            layer_index=layer_index,
                            pp_stage=pp_stage,
                            source=current.tier,
                            target=target,
                            bytes_per_rank=current.bytes_per_rank,
                            reason="length_threshold",
                        )
                    )
                records.append(
                    KVLayerResidency(
                        request_id=str(req.id),
                        layer_index=layer_index,
                        tier=target,
                        allocated_tokens=aligned,
                        bytes_per_rank=desired_bytes,
                    )
                )

        return KVAllocationPlan(
            base_version=self._kv_version,
            records=tuple(records),
            removed_keys=(),
            used_delta=MappingProxyType(
                {
                    device: tuple(values)
                    for device, values in delta.items()
                }
            ),
            growth_bytes_per_rank=max(growth, default=0),
            transfers=tuple(transfers),
        )

    def is_kv_plan_avail(self, plan):
        """检查事务应用后每个 rank 的 HBM/HBF 容量。"""

        if plan.base_version != self._kv_version:
            return False
        for device, used, capacity in (
            (Device.NPU, self._hbm_used_by_rank, self.npu_mem),
            (Device.HBF, self._hbf_used_by_rank, self.hbf_mem),
        ):
            changes = plan.used_delta[device]
            if len(changes) != self.num_npus:
                return False
            for rank_used, change in zip(used, changes):
                if rank_used + change < 0 or rank_used + change > capacity:
                    return False
        return True

    def apply_kv_plan(self, plan):
        """原子提交容量事务；迁移耗时由后续显式资源节点负责。"""

        if plan.base_version != self._kv_version:
            raise RuntimeError("KV 容量事务已过期")
        if not self.is_kv_plan_avail(plan):
            raise RuntimeError("KV 容量事务超过 HBM/HBF 可用容量")

        previous = dict(self._kv_records)
        for device, used in (
            (Device.NPU, self._hbm_used_by_rank),
            (Device.HBF, self._hbf_used_by_rank),
        ):
            for rank, change in enumerate(plan.used_delta[device]):
                used[rank] += change
        for key in plan.removed_keys:
            self._kv_records.pop(key, None)
        for record in plan.records:
            self._kv_records[
                (record.request_id, record.layer_index)
            ] = record
        self._kv_transfer_events.extend(plan.transfers)
        self._kv_version += 1
        self._sync_accelerator_usage()
        if self.tiering_stats is not None:
            for record in plan.records:
                old = previous.get((record.request_id, record.layer_index))
                if old is None:
                    action = "register"
                elif old.tier is record.tier:
                    action = "keep"
                else:
                    action = "migrate"
                self.tiering_stats.record_policy_action(action)
            self.tiering_stats.record_residency_batch(
                hit=not bool(plan.transfers)
            )
            self._observe_tiering_usage()
        return plan.transfers

    def release_request_kv(self, req):
        """释放请求全部层 KV，返回各 tier 的最大单 rank 字节数。"""

        released = {
            MemoryTier.HBM: [0] * self.num_npus,
            MemoryTier.HBF: [0] * self.num_npus,
        }
        prefix = str(req.id)
        keys = [
            key
            for key in self._kv_records
            if key[0] == prefix
        ]
        for key in keys:
            record = self._kv_records.pop(key)
            _, rank_ids = self._pipeline_rank_ids(record.layer_index)
            for rank in rank_ids:
                released[record.tier][rank] += record.bytes_per_rank
        for tier, sizes in released.items():
            if not any(sizes):
                continue
            used = (
                self._hbm_used_by_rank
                if tier is MemoryTier.HBM
                else self._hbf_used_by_rank
            )
            for rank, size in enumerate(sizes):
                used[rank] -= size
        if keys:
            self._kv_version += 1
            self._sync_accelerator_usage()
            self._observe_tiering_usage()
        return MappingProxyType(
            {
                tier: max(sizes, default=0)
                for tier, sizes in released.items()
            }
        )

    def take_kv_transfer_events(self):
        events = tuple(self._kv_transfer_events)
        self._kv_transfer_events.clear()
        return events

    def complete_kv_transfer_events(self, events):
        """在 ASTRA batch 完成后提交迁移统计。"""

        if self.tiering_stats is None:
            stats_enabled = False
        else:
            stats_enabled = True
        for event in events:
            rank_bytes = [0] * self.num_npus
            first = event.pp_stage * self.tp_size
            for rank in range(first, first + self.tp_size):
                rank_bytes[rank] = event.bytes_per_rank
            source_used = (
                self._hbm_used_by_rank
                if event.source is MemoryTier.HBM
                else self._hbf_used_by_rank
            )
            source_weight = (
                self._hbm_weight_by_rank
                if event.source is MemoryTier.HBM
                else self._hbf_weight_by_rank
            )
            for rank, size in enumerate(rank_bytes):
                if source_used[rank] - size < source_weight[rank]:
                    raise RuntimeError(
                        "KV 迁移完成时源层容量小于待释放副本"
                    )
                source_used[rank] -= size
            if stats_enabled:
                self.tiering_stats.record_explicit_transfer(
                    source=event.source,
                    target=event.target,
                    bytes_per_rank=tuple(rank_bytes),
                    reason=event.reason,
                    object_kind=MemoryObjectKind.KV,
                    layer_index=event.layer_index,
                )
        if events:
            self._sync_accelerator_usage()
            self._observe_tiering_usage()

    def _observe_tiering_usage(self):
        if self.tiering_stats is None:
            return
        self.tiering_stats.observe_usage(
            {
                MemoryTier.HBM: tuple(self._hbm_used_by_rank),
                MemoryTier.HBF: tuple(self._hbf_used_by_rank),
            }
        )

    def tiering_stats_snapshot(self):
        if self.tiering_stats is None:
            return None
        return self.tiering_stats.snapshot()

    def kv_tier_of(self, request_id, layer_index):
        try:
            return self._kv_records[(str(request_id), layer_index)].tier
        except KeyError as exc:
            raise RuntimeError(
                f"request={request_id} layer={layer_index} 尚无 KV 驻留"
            ) from exc

    def batch_memory_view(self, requests):
        """冻结 Trace lookup 使用的权重和逐请求、逐层 KV 驻留。"""

        kv_tiers = {}
        for req in requests:
            fallback = self._select_kv_tier(
                req,
                req.num_computed_tokens,
            )
            for layer_index in range(self.n_layer):
                record = self._kv_records.get((str(req.id), layer_index))
                kv_tiers[(str(req.id), layer_index)] = (
                    record.tier if record is not None else fallback
                )
        if self.tiering_stats is not None and requests:
            for layer_index in range(self.n_layer):
                tiers = {
                    kv_tiers[(str(req.id), layer_index)]
                    for req in requests
                }
                self.tiering_stats.record_attention_groups(
                    hbm_groups=int(MemoryTier.HBM in tiers),
                    hbf_groups=int(MemoryTier.HBF in tiers),
                )
        return BatchMemoryView(
            snapshot_version=self._kv_version,
            weight_tiers=self._weight_tiers,
            kv_tiers=kv_tiers,
        )
    
    # get the total size of current kv cache for the request
    # used when adding prefilled request to decode instance.
    def get_total_kv(self, req):
        # ceil division: (n + block_size - 1) // block_size
        num_blocks = (req.num_computed_tokens + self.block_size - 1) // self.block_size
        return self.get_kv(num_blocks * self.block_size)

    # get size of kv block that should be 'added'. including new init requests
    # also checks evicted request and include its kv cache
    # scheduled_tokens: dict mapping request id to number of tokens scheduled this step
    # 
    # vLLM-style cumulative allocation:
    #   blocks_after = ceil((computed + scheduled) / block_size)
    #   blocks_before = ceil(computed / block_size) if computed > 0 else 0
    #   new_blocks = blocks_after - blocks_before
    def get_block_kv(self, batch_req, batch_len, scheduled_tokens=None):
        # print("[get_block_kv] current batch_req length : {}".format(batch_len))
        block_kv_size = 0
        for i in range(batch_len):
            req = batch_req[i]
            if req.evict or req.is_prefill():
                # Prefill and reloaded decode requests may allocate newly
                # computed blocks. Existing evicted KV is reloaded separately
                # by Scheduler.load_size.
                hit = req.npu_cache_hit if self.enable_prefix_caching else 0
                
                if scheduled_tokens and req.id in scheduled_tokens:
                    tokens_this_step = scheduled_tokens[req.id]
                else:
                    raise RuntimeError("[MemoryModel] [node_id={self.node_id},inst={self.instance_id}]: scheduled_tokens cannot be None")
                
                # vLLM-style cumulative block allocation
                computed_before = req.num_computed_tokens
                
                total_after = computed_before + tokens_this_step
                
                # Calculate blocks needed (cumulative)
                blocks_after = (total_after + self.block_size - 1) // self.block_size
                blocks_before = (computed_before + self.block_size - 1) // self.block_size if computed_before > 0 else 0
                
                
                new_blocks = max(0, blocks_after - blocks_before)
                block_kv_size += self.get_kv(new_blocks * self.block_size)
                # print("[DEBUG] hit : {} | tokens_this_step : {} | computed_before : {} | total_after : {} | new_blocks : {} | block_kv_size : {}".format(
                #     hit, tokens_this_step, computed_before, total_after, new_blocks, block_kv_size
                # ))
            else:
                # Decode: use num_computed_tokens (or input for backwards compat)
                computed = req.num_computed_tokens
                num_before = (computed + self.block_size - 1) // self.block_size if computed > 0 else 0
                num_after = (computed + 1 + self.block_size - 1) // self.block_size
                if num_after > num_before: # difference of the block is maximum one block
                    block_kv_size += self.get_kv(self.block_size)
        return block_kv_size
    
    # get size of kv cache that should be evicted
    def get_evict_kv(self, req):
        evict_size = 0
        # Use num_computed_tokens if available, fallback to input for backwards compat
        computed = req.num_computed_tokens
        hit = req.npu_cache_hit if self.enable_prefix_caching else 0
        needed = max(0, computed - hit)
        # ceil division: (needed + block_size - 1) // block_size
        num_blocks = (needed + self.block_size - 1) // self.block_size
        evict_size += self.get_kv(num_blocks * self.block_size)
        return evict_size

    def _sync_accelerator_usage(self):
        self.npu_used = max(self._hbm_used_by_rank, default=0)
        self.hbf_used = max(self._hbf_used_by_rank, default=0)

    def free_weight(self):
        if self.tiering_enabled:
            for rank in range(self.num_npus):
                self._hbm_used_by_rank[rank] -= self._hbm_weight_by_rank[rank]
                self._hbf_used_by_rank[rank] -= self._hbf_weight_by_rank[rank]
            self._hbm_weight_by_rank = (0,) * self.num_npus
            self._hbf_weight_by_rank = (0,) * self.num_npus
            self.hbm_weight = 0
            self.hbf_weight = 0
            self._sync_accelerator_usage()
            return
        if self.npu_used - self.weight < 0:
            raise RuntimeError(
                f"[MemoryModel] [node={self.node_id}, inst={self.instance_id}] NPU: tried to free model weight {self.weight / MB_TO_BYTE:.2f}MB "
                f"but only {self.npu_used / MB_TO_BYTE:.2f}MB is used."
            )
        self.logger.info(
            "NPU: used: %.2fMB remove: %.2fMB after: %.2fMB",
            self.npu_used / MB_TO_BYTE,
            self.weight / MB_TO_BYTE,
            (self.npu_used - self.weight) / MB_TO_BYTE,
        )
        self.npu_used -= self.weight

    def is_free(self):
        if not self.tiering_enabled:
            is_free = self.npu_used == 0 and self.cpu_used == 0
            if not is_free:
                self.logger.error(
                    "Memory leak detected: NPU used: %.2fMB, CPU used: %.2fMB",
                    self.npu_used / MB_TO_BYTE,
                    self.cpu_used / MB_TO_BYTE,
                )
            return is_free
        is_free = (
            self.npu_used == 0
            and self.hbf_used == 0
            and self.cpu_used == 0
        )
        if not is_free:
            self.logger.error(
                "Memory leak detected: NPU used: %.2fMB, HBF used: %.2fMB, CPU used: %.2fMB",
                self.npu_used / MB_TO_BYTE,
                self.hbf_used / MB_TO_BYTE,
                self.cpu_used / MB_TO_BYTE,
            )
        return is_free

    # -------------------- Memory Management --------------------
    
    def allocate(self, size, device):
        if device == Device.NPU:
            if self.tiering_enabled:
                if any(
                    used + size > self.npu_mem
                    for used in self._hbm_used_by_rank
                ):
                    raise RuntimeError(
                        f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] "
                        f"NPU: tried to load {size / MB_TO_BYTE:.2f}MB but "
                        f"one or more ranks lack capacity."
                    )
                for rank in range(self.num_npus):
                    self._hbm_used_by_rank[rank] += size
                self._sync_accelerator_usage()
                self._observe_tiering_usage()
                return
            if self.npu_used + size > self.npu_mem:
                raise RuntimeError(
                    f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] NPU: tried to load {size / MB_TO_BYTE:.2f}MB but only {(self.npu_mem - self.npu_used) / MB_TO_BYTE:.2f}MB is available."
                )
            self.logger.info(
                "NPU: used: %.2fMB load: %.2fMB after: %.2fMB",
                self.npu_used / MB_TO_BYTE,
                size / MB_TO_BYTE,
                (self.npu_used + size) / MB_TO_BYTE,
            )
            self.npu_used += size
        elif device == Device.CPU:
            if self.prefix_storage == Device.CPU and self.enable_prefix_sharing:
                self.second_tier_prefix_cache.allocate(size)
            else:
                if self.cpu_used + size > self.cpu_mem:
                    raise RuntimeError(
                        f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] CPU: tried to load {size / MB_TO_BYTE:.2f}MB "
                        f"but only {(self.cpu_mem - self.cpu_used) / MB_TO_BYTE:.2f}MB is available."
                    )
                self.logger.info(
                    "CPU: used: %.2fMB load: %.2fMB after: %.2fMB",
                    self.cpu_used / MB_TO_BYTE,
                    size / MB_TO_BYTE,
                    (self.cpu_used + size) / MB_TO_BYTE,
                )
                self.cpu_used += size
        elif device == Device.CXL:
            self.second_tier_prefix_cache.allocate(size)
        elif device == Device.HBF:
            if not self.tiering_enabled:
                raise RuntimeError(
                    "普通 GPU 未配置 HBF，不能分配 HBF 容量"
                )
            if any(
                used + size > self.hbf_mem
                for used in self._hbf_used_by_rank
            ):
                raise RuntimeError(
                    f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] "
                    f"HBF: tried to load {size / MB_TO_BYTE:.2f}MB but "
                    f"one or more ranks lack capacity."
                )
            for rank in range(self.num_npus):
                self._hbf_used_by_rank[rank] += size
            self._sync_accelerator_usage()
            self._observe_tiering_usage()
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to allocate KV cache in unsupported device {device}")
    
    def free(self, size, device):
        if device == Device.NPU:
            if self.tiering_enabled:
                for rank, used in enumerate(self._hbm_used_by_rank):
                    if used - size < self._hbm_weight_by_rank[rank]:
                        raise RuntimeError(
                            f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] "
                            "NPU: tried to free more KV than the rank owns."
                        )
                for rank in range(self.num_npus):
                    self._hbm_used_by_rank[rank] -= size
                self._sync_accelerator_usage()
                self._observe_tiering_usage()
                return
            if self.npu_used - size < self.weight:
                raise RuntimeError(
                    f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] NPU: tried to free {size / MB_TO_BYTE:.2f}MB but only {(self.npu_used - self.weight) / MB_TO_BYTE:.2f}MB is used."
                )
            self.logger.info(
                "NPU: used: %.2fMB remove: %.2fMB after: %.2fMB",
                self.npu_used / MB_TO_BYTE,
                size / MB_TO_BYTE,
                (self.npu_used - size) / MB_TO_BYTE,
            )
            self.npu_used -= size

        elif device == Device.CPU:
            if self.prefix_storage == Device.CPU and self.enable_prefix_sharing:
                self.second_tier_prefix_cache.free(size)
            else:
                if self.cpu_used - size < 0:
                    raise RuntimeError(
                        f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] CPU: tried to free {size / MB_TO_BYTE:.2f}MB "
                        f"but only {self.cpu_used / MB_TO_BYTE:.2f}MB is used."
                    )
                self.logger.info(
                    "CPU: used: %.2fMB remove: %.2fMB after: %.2fMB",
                    self.cpu_used / MB_TO_BYTE,
                    size / MB_TO_BYTE,
                    (self.cpu_used - size) / MB_TO_BYTE,
                )
                self.cpu_used -= size
        elif device == Device.CXL:
            self.second_tier_prefix_cache.free(size)
        elif device == Device.HBF:
            if not self.tiering_enabled:
                raise RuntimeError(
                    "普通 GPU 未配置 HBF，不能释放 HBF 容量"
                )
            for rank, used in enumerate(self._hbf_used_by_rank):
                if used - size < self._hbf_weight_by_rank[rank]:
                    raise RuntimeError(
                        f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] "
                        "HBF: tried to free more KV than the rank owns."
                    )
            for rank in range(self.num_npus):
                self._hbf_used_by_rank[rank] -= size
            self._sync_accelerator_usage()
            self._observe_tiering_usage()
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to free KV cache in unsupported device {device}")
    
    def is_avail(self, size, device):
        if device == Device.NPU:
            if self.tiering_enabled:
                return all(
                    self.npu_mem - used >= size
                    for used in self._hbm_used_by_rank
                )
            if self.npu_mem - self.npu_used >= size:
                return True
            else:
                return False 
        elif device == Device.CPU:
            if self.enable_prefix_sharing:
                return self.second_tier_prefix_cache.is_avail(size)
            else:
                if self.cpu_mem - self.cpu_used >= size:
                    return True
                else:
                    return False 
        elif device == Device.CXL:
            return self.second_tier_prefix_cache.is_avail(size)
        elif device == Device.HBF:
            if not self.tiering_enabled:
                return False
            return all(
                self.hbf_mem - used >= size
                for used in self._hbf_used_by_rank
            )
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to check available size of unsupported device {device}")
    
    def need_size(self, size, device):
        if device == Device.NPU:
            if self.tiering_enabled:
                return max(0, max(
                    (
                        size - (self.npu_mem - used)
                        for used in self._hbm_used_by_rank
                    ),
                    default=0,
                ))
            needed = (size - (self.npu_mem - self.npu_used))
            if needed > 0:
                return needed
            else:
                return 0
        elif device == Device.CPU:
            if self.enable_prefix_sharing:
                return self.second_tier_prefix_cache.need_size(size)
            else:
                needed = (size - (self.cpu_mem - self.cpu_used))
                if needed > 0:
                    return needed
                else:
                    return 0
        elif device == Device.CXL:
            return self.second_tier_prefix_cache.need_size(size)
        elif device == Device.HBF:
            if not self.tiering_enabled:
                return size
            return max(0, max(
                (
                    size - (self.hbf_mem - used)
                    for used in self._hbf_used_by_rank
                ),
                default=0,
            ))
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to check available size of unsupported device {device}")

    def avail_size(self, device):
        if not self.enable_prefix_caching:
            return 0
        
        if device == Device.NPU:
            return self.npu_prefix_cache.avail_size()
        elif device == Device.CPU or device == Device.CXL:
            return self.second_tier_prefix_cache.avail_size()
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to get available size of prefix cache in unsupported device {device}")
    
    # -------------------- Prefix Cache Management --------------------

    def storage_cache_evicted_req(self, req):
        if self.enable_prefix_caching:
            new_last_node = self.second_tier_prefix_cache.cache_unfinished_req(req, update=False) # do not update hit counts
            # should lock evicted kv cache in cpu
            self.second_tier_prefix_cache.inc_lock_ref(new_last_node)
            req.cpu_last_node = new_last_node
            self.apply_kv_cache_events()

    def evictable_size(self, device):
        if not self.enable_prefix_caching:
            return 0
        
        if device == Device.NPU:
            return self.npu_prefix_cache.evictable_size() * self._bytes_per_token
        elif device == Device.CPU or device == Device.CXL:
            return self.second_tier_prefix_cache.evictable_size() * self._bytes_per_token
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to get evictable size of prefix cache in unsupported device {device}")


    def lock_prefix(self, req, device): 
        # Increment lock ref count on req.npu_last_node (set by prefix_match)
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU and req.npu_last_node is not None:
            node = req.npu_last_node
            # print(f"[LOCK] req={req.id} lock_prefix node_id={node.id} lock_ref_BEFORE={node.lock_ref}")
            self.npu_prefix_cache.inc_lock_ref(req.npu_last_node)
            # print(f"[LOCK] req={req.id} lock_prefix node_id={node.id} lock_ref_AFTER={node.lock_ref}")
        elif (device == Device.CPU or device == Device.CXL) and req.cpu_last_node is not None:
            self.second_tier_prefix_cache.inc_lock_ref(req.cpu_last_node)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to lock prefix cache in unsupported device {device}")
    
    def unlock_prefix(self, req, device):
        # Decrement lock ref count on req.npu_last_node (set by prefix_match)
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU and req.npu_last_node is not None:
            node = req.npu_last_node
            # print(f"[UNLOCK] req={req.id} unlock_prefix node_id={node.id} lock_ref_BEFORE={node.lock_ref}")
            self.npu_prefix_cache.dec_lock_ref(req.npu_last_node)
            # print(f"[UNLOCK] req={req.id} unlock_prefix node_id={node.id} lock_ref_AFTER={node.lock_ref}")
            req.npu_last_node = None
            req._prefix_locked = False
        elif device == Device.CPU and req.cpu_last_node is not None:
            self.second_tier_prefix_cache.dec_lock_ref(req.cpu_last_node)
            req.cpu_last_node = None
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to unlock prefix cache in unsupported device {device}")
    
    def cache_unfinished_req(self, req, device):
        # Get new_last_node via cache_unfinished_req (replaces last node)
        # Decrement old node's lock ref count, increment new node's lock ref count
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU:
            new_last_node = self.npu_prefix_cache.cache_unfinished_req(req)
            
            old_node = req.npu_last_node
            # print(f"[CACHE_UNFINISHED] req={req.id} old_node_id={old_node.id if old_node else None}(lock_ref={old_node.lock_ref if old_node else 'N/A'}) -> new_node_id={new_last_node.id}(lock_ref={new_last_node.lock_ref})")
            if old_node is not None and req._prefix_locked:
                self.npu_prefix_cache.dec_lock_ref(old_node)
            self.npu_prefix_cache.inc_lock_ref(new_last_node)
            # print(f"[CACHE_UNFINISHED] req={req.id} AFTER: old_node_id={old_node.id}(lock_ref={old_node.lock_ref}) new_node_id={new_last_node.id}(lock_ref={new_last_node.lock_ref})")
            req.npu_last_node = new_last_node
            req._prefix_locked = True
            if self.logger.isEnabledFor(logging.DEBUG):
                # print(f"cache_unfinished_req of req {req.id}")
                # print(f"===============NPU PREFIX CAHCE of Instance[{self.instance_id}]=================")
                self.npu_prefix_cache.pretty_print()
        elif device == Device.CPU or device == Device.CXL:
            self.second_tier_prefix_cache.cache_unfinished_req(req)
            if self.logger.isEnabledFor(logging.DEBUG):
                # print(f"cache_unfinished_req of req {req.id}")
                # print(f"===============AFTER INSERT: {self.second_tier_prefix_cache.device} PREFIX CAHCE at pid={os.getpid()} tid={threading.get_ident()} pool_id={id(self.second_tier_prefix_cache)}, size={self.second_tier_prefix_cache.total_size()}=================")
                self.second_tier_prefix_cache.pretty_print()
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to cache prefix cache of unfinished request to unsupported device {device}")
        
        self.apply_kv_cache_events()

    def cache_finished_req(self, req, device):
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU:
            self.npu_prefix_cache.cache_finished_req(req)
            # Only dec_lock_ref if the request was locked
            node = req.npu_last_node
            if not req._prefix_locked:
                # Never locked → skip dec
                pass
                # print(f"[CACHE_FINISHED] req={req.id} node_id={node.id if node else None} lock_ref={node.lock_ref if node else 'N/A'} (SKIPPED dec - not locked)")
            else:
                # print(f"[CACHE_FINISHED] req={req.id} node_id={node.id if node else None} lock_ref_BEFORE={node.lock_ref if node else 'N/A'}")
                if node is not None:
                    self.npu_prefix_cache.dec_lock_ref(node)
                    req.npu_last_node = None
                req._prefix_locked = False
            # node = req.npu_last_node
            # print(f"[CACHE_FINISHED] req={req.id} node_id={node.id if node else None} lock_ref_BEFORE={node.lock_ref if node else 'N/A'}")
            # self.npu_prefix_cache.dec_lock_ref(req.npu_last_node)
                # print(f"[CACHE_FINISHED] req={req.id} node_id={node.id if node else None} lock_ref_AFTER={node.lock_ref if node else 'N/A'}")
            # print(f"[CACHE_FINISHED] req={req.id} evictable_size={self.npu_prefix_cache.evictable_size()} protected_size={self.npu_prefix_cache.protected_size()} total_size={self.npu_prefix_cache.total_size()}")
            if self.logger.isEnabledFor(logging.DEBUG):
                print(f"cache_finished_req of req {req.id}")
                print(f"===============NPU PREFIX CACHE of Instance[{self.instance_id}]=================")
                self.npu_prefix_cache.pretty_print()
        elif device == Device.CPU or device == Device.CXL:
            self.second_tier_prefix_cache.cache_finished_req(req)
            if self.logger.isEnabledFor(logging.DEBUG):
                # print(f"cache_finished_req of req {req.id}")
                # print(f"===============AFTER INSERT: {self.second_tier_prefix_cache.device} PREFIX CAHCE at pid={os.getpid()} tid={threading.get_ident()} pool_id={id(self.second_tier_prefix_cache)}, size={self.second_tier_prefix_cache.total_size()}=================")
                self.second_tier_prefix_cache.pretty_print()
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to cache prefix cache of finished request to unsupported device {device}")
        
        self.apply_kv_cache_events()

    def evict_prefix_cache(self, bytes, device):
        if not self.enable_prefix_caching or bytes <= 0:
            return

        if device == Device.NPU:
            cache = self.npu_prefix_cache
        elif device == Device.CPU:
            cache = self.second_tier_prefix_cache
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to evict prefix cache to unsupported device {device}")

        # Each cache instance carries its own bytes-per-token in kv_size:
        # per-rank for NPU, full-cluster for the second-tier pool.
        space_needed = (bytes + cache.kv_size - 1) // cache.kv_size
        cache.evict(space_needed)

        self.apply_kv_cache_events()

    # -------------------- Prefix Cache Helpers --------------------

    def prefix_match(self, req): # req.prefix_cache_hit initialization 
        if not self.enable_prefix_caching:
            return
        
        tokens = req.input_hash_ids
        if tokens is None:
            return
        old_node = req.npu_last_node
        res = self.npu_prefix_cache.match_prefix(tokens[:req.input])
        req.npu_cache_hit = res.hit_length
        req.npu_last_node = res.last_device_node
        # print(f"[PREFIX_MATCH] req={req.id} old_node_id={old_node.id if old_node else None}(lock_ref={old_node.lock_ref if old_node else 'N/A'}) -> new_node_id={res.last_device_node.id}(lock_ref={res.last_device_node.lock_ref}) hit={res.hit_length} num_computed={req.num_computed_tokens}")

        if self.prefix_storage is not None:
            res_storage = self.second_tier_prefix_cache.match_prefix(tokens[:req.input])
            req.storage_cache_hit = res_storage.hit_length
            req.storage_last_node = res_storage.last_device_node
        else:
            req.storage_cache_hit = 0
            req.storage_last_node = None
        
        req.prefix_cache_hit = max(req.npu_cache_hit, req.storage_cache_hit)
        # if req.num_computed_tokens < req.prefix_cache_hit:
        #     req.num_computed_tokens = req.prefix_cache_hit
        if req.num_computed_tokens == 0:
            req.num_computed_tokens = req.prefix_cache_hit
            # print(f"Request[{req.id}] prefix cache hit: {req.prefix_cache_hit} tokens (NPU: {req.npu_cache_hit}, {self.prefix_storage}: {req.storage_cache_hit})")
        # for debugging
        
        # print(f"===============NPU PREFIX CAHCE of Instance[{self.instance_id}]=================")
        # self.npu_prefix_cache.pretty_print()
        # print("===============CPU PREFIX CAHCE=================")
        # self.second_tier_prefix_cache.pretty_print()
    
    def erase_prefix_info(self, req):
        if not self.enable_prefix_caching:
            return
        
        req.prefix_cache_hit = 0
        req.npu_cache_hit = 0
        req.storage_cache_hit = 0
        req.npu_last_node = None
        req.storage_last_node = None

    def free_prefix_cache(self):
        if not self.enable_prefix_caching:
            return
        # free evictable prefix cache, if evictable_size != total_size there is locked prefix cache
        self.free(self.npu_prefix_cache.evictable_size() * self._bytes_per_token, Device.NPU)
        if not self.enable_prefix_sharing and self.prefix_storage is not None:
            self.free(self.second_tier_prefix_cache.evictable_size() * self._bytes_per_token * self.num_npus, self.prefix_storage)
    
    # Count load/unload events from prefix cache and update memory usage
    def apply_kv_cache_events(self):
        # if not self.enable_prefix_caching:
        #     return
        npu_byte_alloc = 0
        npu_byte_free = 0
        cpu_byte_alloc = 0
        cpu_byte_free = 0
        # self.npu_prefix_cache.take_events() -> [BlockStored, BlockStored, BlockRemoved, ...]
        for ev in self.npu_prefix_cache.take_events():
            # print(f" current event block: {ev}")
            if isinstance(ev, BlockStored):
                tlen = len(ev.token_ids)
                for h in ev.block_hashes:
                    # self._npu_cache_hashtolen[h] = tlen
                    if h in self._npu_cache_hashtolen:
                        self._npu_cache_hashtolen[h][1] += 1
                        # if self._npu_cache_hashtolen[h][1] >= 2:
                        #     print("duplicated hash occurs!! h : {}".format(h))
                    else:
                        self._npu_cache_hashtolen[h] = [tlen, 1]
                npu_byte_alloc += self.get_kv(tlen)
            elif isinstance(ev, BlockRemoved):
                for h in ev.block_hashes:
                    # tlen = self._npu_cache_hashtolen.pop(h, 0)
                    # if tlen == 0:
                    if h in self._npu_cache_hashtolen:
                        tlen = self._npu_cache_hashtolen[h][0]
                        self._npu_cache_hashtolen[h][1] -= 1
                        if self._npu_cache_hashtolen[h][1] <= 0:
                            del self._npu_cache_hashtolen[h]
                        npu_byte_free += self.get_kv(tlen)
                    else:
                        print(f"[HASH_MISS] BlockRemoved hash={h} NOT FOUND in map (map_size={len(self._npu_cache_hashtolen)})")
                        self.logger.warning(f"NPU prefix cache remove unknown block hash {h}")
                    # else:
                    #     print(f"[HASH_HIT] BlockRemoved hash={h} tlen={tlen}")
                    # npu_byte_free += self.get_kv(tlen)
        # free first, then allocate
        if npu_byte_free > 0:
            self.free(npu_byte_free, Device.NPU)
        if npu_byte_alloc > 0:
            self.allocate(npu_byte_alloc, Device.NPU)
        # if npu_byte_free > 0:
        #     self.free(npu_byte_free, Device.NPU)

        # Second-tier (CPU/CXL) prefix cache events.
        if self.prefix_storage is None:
            return

        if self.prefix_storage is Device.CPU and not self.enable_prefix_sharing:
            # Non-shared CPU second_tier: bridge events into the instance's
            # cpu_used counter so allocations are bounded by cpu_mem.
            for ev in self.second_tier_prefix_cache.take_events():
                if isinstance(ev, BlockStored):
                    tlen = len(ev.token_ids)
                    for h in ev.block_hashes:
                        if h in self._cpu_cache_hashtolen:
                            self._cpu_cache_hashtolen[h][1] += 1
                        else:
                            self._cpu_cache_hashtolen[h] = [tlen, 1]
                    cpu_byte_alloc += self.get_kv(tlen) * self.num_npus
                elif isinstance(ev, BlockRemoved):
                    for h in ev.block_hashes:
                        if h in self._cpu_cache_hashtolen:
                            tlen = self._cpu_cache_hashtolen[h][0]
                            self._cpu_cache_hashtolen[h][1] -= 1
                            if self._cpu_cache_hashtolen[h][1] <= 0:
                                del self._cpu_cache_hashtolen[h]
                            cpu_byte_free += self.get_kv(tlen) * self.num_npus
                        else:
                            self.logger.warning(f"CPU prefix cache remove unknown block hash {h}")

            if cpu_byte_free > 0:
                self.free(cpu_byte_free, Device.CPU)
            if cpu_byte_alloc > 0:
                self.allocate(cpu_byte_alloc, Device.CPU)
        else:
            # Shared pool or CXL: the cache itself accounts via
            # total_memory_usage (= kv_stored + total_size * kv_size),
            # so no instance-side counter update is needed. Drain the
            # queue to prevent it from growing unboundedly.
            self.second_tier_prefix_cache.take_events()

    def return_prefix_info(self):
        if not self.enable_prefix_caching:
            return (0, 0, 0, 0)
        if self.prefix_storage is None:
            return (self.npu_prefix_cache.return_prefix_info(), (0, 0))
        return (self.npu_prefix_cache.return_prefix_info(), self.second_tier_prefix_cache.return_prefix_info())

        
def full_cluster_kv_bytes_per_token(model, fp, kv_cache_dtype='auto'):
    """Bytes of KV cache per token aggregated over the full TP cluster.

    Mirrors MemoryModel.get_kv(1) * num_npus but computes directly, avoiding
    the per-rank floor-division roundoff. ``fp`` is the model weight dtype
    in bits (16, 32, ...). ``kv_cache_dtype='fp8'`` forces 1 byte per element
    for the KV cache regardless of weight dtype.
    """
    config = get_config(model)
    n_embd = config['hidden_size']
    n_head = config['num_attention_heads']
    head_dim = config.get('head_dim', n_embd // n_head)
    kv_head = config.get('num_key_value_heads', n_head)
    kv_dim = kv_head * head_dim
    n_layer = config['num_hidden_layers']
    kv_fp = 1 if kv_cache_dtype == 'fp8' else fp // 8
    # 2 (K + V) * kv_dim * n_layer * bytes_per_elem
    return 2 * kv_dim * n_layer * kv_fp


# calculate the per-rank input, weight, output size of each layer
def calculate_sizes(model, layer_name, length, kv_len=None, pim=False, parallel=1, fp=2):
    """Calculate input, weight, and output tensor sizes for a given layer.

    Args:
        parallel: parallelism degree for weight/activation sharding.
            For dense layers this is TP; for MoE experts this is EP.
    """
    config = get_config(model)
    n_embd = config['hidden_size']
    n_head = config['num_attention_heads']
    head_dim = config.get('head_dim', n_embd // n_head)
    vocab_size = config['vocab_size']
    kv_head = config.get("num_key_value_heads", n_head)  # fallback to n_head if not defined
    q_dim = n_head * head_dim       # total Q projection output dim
    kv_dim = kv_head * head_dim     # total KV projection output dim
    ffn_dim = config.get("intermediate_size", config.get("ffn_dim"))  # dense FFN dim
    moe_ffn_dim = config.get("moe_intermediate_size", ffn_dim)  # per-expert FFN dim (may differ from dense)
    # Same both-name fallback as MemoryModel.__init__ — HF / Qwen use
    # ``num_experts`` while Mistral uses ``num_local_experts``.
    num_local_experts = config.get(
        "num_local_experts", config.get("num_experts", 1)
    )

    p = max(int(parallel), 1)

    # NOTE (vLLM-style assumptions):
    # NOTE (vLLM-style assumptions):
    # - Embedding / LM head: vocab-parallel → split vocab_size across ranks.
    # - Q/K/V: ColumnParallelLinear         → split output dim across ranks.
    # - o_proj: RowParallelLinear           → split input dim across ranks.
    # - LayerNorm weights: replicated (NOT sharded).
    # - MoE experts: parallel = EP degree, each rank holds num_local_experts // p experts.

    # ----------------- Embedding & Norms -----------------
    if layer_name == "embedding":
        input_size = length * fp * 2  # token_ids are int32 or int64
        weight_size = (vocab_size // p) * n_embd * fp
        output_size = length * n_embd * fp

    elif layer_name in ["input_layernorm", "post_layernorm", "final_layernorm", "layernorm"]:
        input_size = length * n_embd * fp
        weight_size = 1 * n_embd * fp  # scale only
        output_size = length * n_embd * fp

    elif layer_name == "qk_norm":
        input_size = length * (q_dim + kv_dim) // p * fp
        weight_size = 2 * head_dim * fp
        output_size = length * (q_dim + kv_dim) // p * fp

    # ----------------- RoPE & Attention Core -----------------
    elif layer_name == "rotary_emb":
        input_size = ((n_head // p) + (kv_head // p)) * length * head_dim * fp
        weight_size = 0
        output_size = ((n_head // p) + (kv_head // p)) * length * head_dim * fp

    elif layer_name == "attention":
        if not pim:
            input_size = (
                (n_head // p) * length * head_dim * fp +
                (kv_head // p) * kv_len * head_dim * fp * 2
            )
            weight_size = 0
            output_size = (n_head // p) * length * head_dim * fp
        else:
            input_size = (
                (n_head // p) * 1 * head_dim * fp +
                (kv_head // p) * 1 * head_dim * fp * 2
            )
            weight_size = 0
            output_size = (n_head // p) * 1 * head_dim * fp

    # ----------------- QKV Projection (fused) -----------------
    elif layer_name == "qkv_proj":
        input_size = length * n_embd * fp
        weight_size = n_embd * ((q_dim + 2 * kv_dim) // p) * fp
        output_size = length * ((q_dim + 2 * kv_dim) // p) * fp

    elif layer_name == "o_proj":
        input_size = length * (q_dim // p) * fp
        weight_size = (q_dim // p) * n_embd * fp
        output_size = length * n_embd * fp

    elif layer_name == "gate_up_proj":
        input_size = length * n_embd * fp
        weight_size = n_embd * 2 * (ffn_dim // p) * fp
        output_size = length * 2 * (ffn_dim // p) * fp

    elif layer_name == "act_fn":
        input_size = length * 2 * (ffn_dim // p) * fp
        weight_size = 0
        output_size = length * (ffn_dim // p) * fp

    elif layer_name == "down_proj":
        input_size = length * (ffn_dim // p) * fp
        weight_size = (ffn_dim // p) * n_embd * fp
        output_size = length * n_embd * fp

    elif layer_name == "sampler":
        input_size = length * (vocab_size // p) * fp
        weight_size = 0
        output_size = length * 4  # int32 token IDs

    elif layer_name == "moe":
        experts_per_rank = num_local_experts // p
        input_size = length * n_embd * fp
        weight_size = (n_embd * num_local_experts * fp  # gate (replicated)
                     + experts_per_rank * 3 * n_embd * moe_ffn_dim * fp)  # local experts
        output_size = length * n_embd * fp

    # ----------------- LM Head -----------------
    elif layer_name == "lm_head":
        input_size = length * n_embd * fp
        weight_size = n_embd * (vocab_size // p) * fp
        output_size = length * (vocab_size // p) * fp

    else:
        raise ValueError(f"No matching layer name {layer_name} found for model {model}.")

    return input_size, weight_size, output_size
