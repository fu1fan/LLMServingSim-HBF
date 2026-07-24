import unittest
from unittest.mock import patch

import serving.core.memory_model as memory_model_module
import serving.core.scheduler as scheduler_module
from serving.core.memory_model import Device, MemoryModel
from serving.core.memory_tiering import MemoryTier
from serving.core.memory_tiering_config import parse_instance_memory_tiering
from serving.core.request import Request
from serving.core.scheduler import Scheduler


_MODEL_CONFIG = {
    "hidden_size": 4,
    "num_hidden_layers": 4,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "intermediate_size": 8,
    "vocab_size": 8,
    "max_position_embeddings": 128,
    "model_type": "llama",
}


def _tiering(*, weights=None, kv=None, prefix=None, hbf_gb=1):
    policy = {}
    if weights is not None:
        policy["weights"] = weights
    if kv is not None:
        policy["kv"] = kv
    if prefix is not None:
        policy["prefix"] = prefix
    return parse_instance_memory_tiering(
        {
            "hbf_mem": {"mem_size": hbf_gb},
            "performance_profile": {
                "mode": "memory_scenario_v2",
                "scenario_selection": "residency_derived",
            },
            "memory_tiering": policy,
        },
        _MODEL_CONFIG["num_hidden_layers"],
    )


def _memory(
    config=None,
    *,
    npu_gb=1,
    prefix=False,
    engine=None,
    num_npus=1,
    tp_size=1,
    pp_size=1,
):
    return MemoryModel(
        "test-model",
        0,
        0,
        num_npus,
        tp_size,
        npu_gb,
        1,
        16,
        16,
        prefix,
        False,
        None,
        None,
        pp_size=pp_size,
        memory_tiering=config,
        kv_policy_engine=engine,
    )


def _scheduler(config=None, *, pd_type=None):
    return Scheduler(
        "test-model",
        0,
        0,
        4,
        128,
        1,
        1,
        1,
        1,
        1,
        0,
        pd_type,
        16,
        16,
        4,
        False,
        False,
        False,
        None,
        None,
        memory_tiering=config,
    )


class _StaticWatermarkEngine:
    def __init__(self, tier):
        self.tier = tier
        self.calls = []

    def select_kv_tier(self, **context):
        self.calls.append(context)
        return self.tier


class HbfSchedulerMemoryTest(unittest.TestCase):
    def setUp(self):
        self.patches = (
            patch.object(
                memory_model_module,
                "get_config",
                return_value=_MODEL_CONFIG,
            ),
            patch.object(
                scheduler_module,
                "get_config",
                return_value=_MODEL_CONFIG,
            ),
        )
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_device_extension_preserves_legacy_enum_values(self):
        self.assertEqual(Device.NPU.value, 1)
        self.assertEqual(Device.CPU.value, 2)
        self.assertEqual(Device.CXL.value, 3)
        self.assertEqual(Device.HBF.value, 4)

    def test_ordinary_gpu_keeps_legacy_weight_and_kv_accounting(self):
        memory = _memory()

        # 共享权重 136B，单块 304B，共四块。
        self.assertEqual(memory.weight, 1352)
        self.assertEqual(memory.npu_used, 1352)
        self.assertEqual(memory.hbf_used, 0)
        kv_bytes = memory.get_kv(16)
        self.assertEqual(kv_bytes, 512)

        memory.allocate(kv_bytes, Device.NPU)
        self.assertEqual(memory.npu_used, 1864)
        memory.free(kv_bytes, Device.NPU)
        self.assertEqual(memory.npu_used, memory.weight)

    def test_static_weight_map_accounts_canonical_layers_and_blocks(self):
        config = _tiering(
            weights={
                "policy": "static_map",
                "default_tier": "hbm",
                "layers": {"lm_head": "hbf"},
                "blocks": [{"blocks": "0-1", "tier": "hbf"}],
            }
        )
        memory = _memory(config)
        view = memory.batch_memory_view([])

        self.assertEqual(
            view.weight_tier("qkv_proj", 0),
            MemoryTier.HBF,
        )
        self.assertEqual(
            view.weight_tier("qkv_proj", 3),
            MemoryTier.HBM,
        )
        self.assertEqual(
            view.weight_tier("lm_head", None),
            MemoryTier.HBF,
        )
        self.assertGreater(memory.hbf_weight, 0)
        self.assertLess(memory.hbm_weight, memory.weight)

    def test_hbf_weights_do_not_require_full_model_hbm_capacity(self):
        config = _tiering(weights={"policy": "hbf_only"})

        memory = _memory(config, npu_gb=0.00000001)

        self.assertEqual(memory.npu_used, 0)
        self.assertEqual(memory.hbf_used, memory.weight)
        with self.assertRaisesRegex(RuntimeError, "exceeds total NPU memory"):
            _memory(npu_gb=0.00000001)

    def test_hbf_only_kv_uses_independent_capacity_and_releases_by_request(self):
        config = _tiering(kv={"policy": "hbf_only"})
        memory = _memory(config)
        req = Request(7, "test-model", 16, 32, 0, 0)
        plan = memory.plan_kv_allocation(
            [req],
            scheduled_tokens={7: 16},
        )

        self.assertTrue(memory.is_kv_plan_avail(plan))
        memory.apply_kv_plan(plan)
        self.assertEqual(memory.kv_tier_of(7, 0), MemoryTier.HBF)
        self.assertEqual(memory.npu_used, memory.hbm_weight)
        self.assertGreater(memory.hbf_used, memory.hbf_weight)

        released = memory.release_request_kv(req)
        self.assertEqual(released[MemoryTier.HBM], 0)
        self.assertEqual(released[MemoryTier.HBF], 512)
        self.assertEqual(memory.hbf_used, memory.hbf_weight)

    def test_length_threshold_moves_each_request_layer_as_one_unit(self):
        config = _tiering(
            kv={
                "policy": "length_threshold",
                "threshold_tokens": 32,
            }
        )
        memory = _memory(config)
        req = Request(9, "test-model", 16, 64, 0, 0)
        first = memory.plan_kv_allocation(
            [req],
            scheduled_tokens={9: 16},
        )
        memory.apply_kv_plan(first)
        req.num_computed_tokens = 16

        crossing = memory.plan_kv_allocation(
            [req],
            scheduled_tokens={9: 16},
        )
        memory.apply_kv_plan(crossing)
        events = memory.take_kv_transfer_events()

        self.assertEqual(len(events), _MODEL_CONFIG["num_hidden_layers"])
        self.assertEqual(
            {event.layer_index for event in events},
            set(range(_MODEL_CONFIG["num_hidden_layers"])),
        )
        self.assertTrue(
            all(
                event.source is MemoryTier.HBM
                and event.target is MemoryTier.HBF
                for event in events
            )
        )
        self.assertTrue(
            all(
                memory.kv_tier_of(req.id, layer) is MemoryTier.HBF
                for layer in range(_MODEL_CONFIG["num_hidden_layers"])
            )
        )
        self.assertGreater(memory.npu_used, memory.hbm_weight)
        memory.complete_kv_transfer_events(events)
        self.assertEqual(memory.npu_used, memory.hbm_weight)
        stats = memory.tiering_stats_snapshot()
        self.assertEqual(
            stats.transfer_directions[
                (MemoryTier.HBM, MemoryTier.HBF)
            ].operations,
            _MODEL_CONFIG["num_hidden_layers"],
        )
        self.assertGreater(
            stats.resident_high_water_bytes[MemoryTier.HBF][0],
            memory.hbf_weight,
        )

    def test_pp_kv_is_charged_only_to_the_owning_tp_ranks(self):
        model_config = {
            **_MODEL_CONFIG,
            "num_hidden_layers": 5,
            "num_key_value_heads": 2,
        }
        config = parse_instance_memory_tiering(
            {
                "hbf_mem": {"mem_size": 1},
                "performance_profile": {
                    "mode": "memory_scenario_v2",
                    "scenario_selection": "residency_derived",
                },
                "memory_tiering": {
                    "kv": {
                        "policy": "length_threshold",
                        "threshold_tokens": 32,
                    }
                },
            },
            model_config["num_hidden_layers"],
        )
        with patch.object(
            memory_model_module,
            "get_config",
            return_value=model_config,
        ):
            memory = _memory(
                config,
                num_npus=4,
                tp_size=2,
                pp_size=2,
            )

        req = Request(12, "test-model", 16, 64, 0, 0)
        first = memory.plan_kv_allocation(
            [req],
            scheduled_tokens={12: 16},
        )

        # 五层按 3+2 分到两个 PP stage；每层只在所属 stage 的两个 TP rank 上。
        self.assertEqual(
            first.used_delta[Device.NPU],
            (384, 384, 256, 256),
        )
        memory.apply_kv_plan(first)
        req.num_computed_tokens = 16

        crossing = memory.plan_kv_allocation(
            [req],
            scheduled_tokens={12: 16},
        )
        self.assertEqual(
            crossing.used_delta[Device.NPU],
            (0, 0, 0, 0),
        )
        self.assertEqual(
            crossing.used_delta[Device.HBF],
            (768, 768, 512, 512),
        )
        self.assertEqual(
            [event.pp_stage for event in crossing.transfers],
            [0, 0, 0, 1, 1],
        )

        memory.apply_kv_plan(crossing)
        memory.complete_kv_transfer_events(crossing.transfers)
        memory.release_request_kv(req)
        self.assertEqual(
            memory._hbm_used_by_rank,
            list(memory._hbm_weight_by_rank),
        )
        self.assertEqual(
            memory._hbf_used_by_rank,
            list(memory._hbf_weight_by_rank),
        )

    def test_hbf_capacity_check_is_independent_from_hbm(self):
        config = _tiering(
            kv={"policy": "hbf_only"},
            hbf_gb=0.0000001,
        )
        memory = _memory(config)
        req = Request(3, "test-model", 16, 32, 0, 0)

        plan = memory.plan_kv_allocation(
            [req],
            scheduled_tokens={3: 16},
        )

        self.assertFalse(memory.is_kv_plan_avail(plan))
        self.assertTrue(memory.is_avail(plan.growth_bytes_per_rank, Device.NPU))

    def test_watermark_policy_fails_closed_without_runtime_engine(self):
        config = _tiering(kv={"policy": "watermark_lru"})
        req = Request(4, "test-model", 16, 32, 0, 0)

        with self.assertRaisesRegex(RuntimeError, "select_kv_tier"):
            _memory(config).plan_kv_allocation(
                [req],
                scheduled_tokens={4: 16},
            )

        engine = _StaticWatermarkEngine(MemoryTier.HBF)
        memory = _memory(config, engine=engine)
        plan = memory.plan_kv_allocation(
            [req],
            scheduled_tokens={4: 16},
        )
        memory.apply_kv_plan(plan)
        self.assertEqual(memory.kv_tier_of(4, 0), MemoryTier.HBF)
        self.assertEqual(len(engine.calls), 1)

    def test_prefix_caching_keeps_legacy_hbm_path_and_rejects_hbf_prefix(self):
        weights_hbf = _tiering(weights={"policy": "hbf_only"})
        memory = _memory(weights_hbf, prefix=True)

        self.assertEqual(memory.mem_for_kv, memory.npu_mem)
        self.assertEqual(memory.npu_prefix_cache.capacity, memory.npu_mem)

        prefix_hbf = _tiering(prefix={"policy": "hbf_only"})
        with self.assertRaisesRegex(RuntimeError, "RadixCache"):
            _memory(prefix_hbf, prefix=True)

    def test_tiered_generic_allocation_updates_residency_high_water(self):
        memory = _memory(_tiering())
        before = tuple(memory._hbm_used_by_rank)

        memory.allocate(128, Device.NPU)

        snapshot = memory.tiering_stats_snapshot()
        self.assertEqual(
            snapshot.resident_high_water_bytes[MemoryTier.HBM],
            tuple(value + 128 for value in before),
        )
        memory.free(128, Device.NPU)

    def test_scheduler_attaches_real_residency_and_releases_finished_kv(self):
        config = _tiering(
            weights={"policy": "hbf_only"},
            kv={"policy": "hbm_only"},
        )
        scheduler = _scheduler(config)
        scheduler.add_request([1, "test-model", 16, 17, 0, 0])

        batch = scheduler.schedule(0, 0)

        self.assertIsNotNone(batch)
        self.assertEqual(
            batch.memory_view.weight_tier("qkv_proj", 0),
            MemoryTier.HBF,
        )
        self.assertEqual(
            batch.memory_view.kv_tier(1, 0),
            MemoryTier.HBM,
        )
        self.assertEqual(batch.memory_transfers, ())
        self.assertGreater(scheduler.memory.npu_used, 0)

        _, _, finished = scheduler.add_done(
            batch.batch_id + 1,
            0,
            100,
        )
        self.assertEqual([req.id for req in finished], [1])
        self.assertEqual(
            scheduler.memory.npu_used,
            scheduler.memory.hbm_weight,
        )

    def test_pd_decode_admission_uses_target_kv_tier(self):
        config = _tiering(kv={"policy": "hbf_only"})
        scheduler = _scheduler(config, pd_type="decode")
        req = Request(5, "test-model", 16, 32, 0, 0, is_init=False)
        req.num_computed_tokens = 16

        scheduler.add_decode(req)

        self.assertEqual(
            scheduler.memory.kv_tier_of(req.id, 0),
            MemoryTier.HBF,
        )
        self.assertEqual(scheduler.request[0], req)

    def test_failed_pd_decode_admission_does_not_take_request_ownership(self):
        config = _tiering(
            kv={"policy": "hbf_only"},
            hbf_gb=0.00000001,
        )
        scheduler = _scheduler(config, pd_type="decode")
        req = Request(6, "test-model", 16, 32, 0, 0, is_init=False)
        req.num_computed_tokens = 16
        req.instance_id = 99

        with self.assertRaisesRegex(RuntimeError, "无法容纳"):
            scheduler.add_decode(req)

        self.assertEqual(scheduler.request, [])
        self.assertEqual(req.instance_id, 99)
        self.assertEqual(scheduler.memory._kv_records, {})


if __name__ == "__main__":
    unittest.main()
