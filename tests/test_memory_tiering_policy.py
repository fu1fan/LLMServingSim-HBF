import unittest

from serving.core.memory_tiering import (
    MemoryObjectKey,
    MemoryObjectKind,
    MemoryTier,
    ResidencyState,
    TieredResidencyManager,
)
from serving.core.memory_tiering_config import parse_instance_memory_tiering
from serving.core.memory_tiering_policy import (
    CachePlacementRequest,
    PlacementMode,
    PolicyAction,
    TieringPolicyEngine,
    TieringPolicyError,
    WeightPlacementRequest,
)


def _config(tiering=None):
    instance = {
        "hbf_mem": {"mem_size": 1},
        "performance_profile": {
            "mode": "memory_scenario_v2",
            "memory_profile_id": "test-profile",
            "scenario_selection": "residency_derived",
        },
    }
    if tiering is not None:
        instance["memory_tiering"] = tiering
    return parse_instance_memory_tiering(instance, 8)


def _manager(hbm=(100,), hbf=(500,)):
    return TieredResidencyManager(
        {
            MemoryTier.HBM: hbm,
            MemoryTier.HBF: hbf,
        },
        num_ranks=len(hbm),
    )


def _weight(name, size, layer, block=None):
    return WeightPlacementRequest(
        MemoryObjectKey(MemoryObjectKind.WEIGHT, name, block),
        (size,),
        layer,
        block,
    )


def _cache(kind, name, size, tokens, *, layer=None, hits=0):
    return CachePlacementRequest(
        MemoryObjectKey(kind, name, layer),
        (size,),
        tokens,
        hit_count=hits,
    )


class WeightTieringPolicyTest(unittest.TestCase):
    def test_static_map_is_stable_and_honours_layer_then_block(self):
        config = _config(
            {
                "weights": {
                    "policy": "static_map",
                    "default_tier": "hbm",
                    "layers": {"lm_head": "hbf"},
                    "blocks": [{"blocks": "0-1", "tier": "hbf"}],
                }
            }
        )
        manager = _manager()
        engine = TieringPolicyEngine(config, manager)

        plan = engine.place_weights(
            (
                _weight("z", 20, "qkv_proj", 3),
                _weight("a", 30, "lm_head"),
                _weight("m", 40, "qkv_proj", 1),
            )
        )

        self.assertEqual(
            [item.key.object_id for item in plan.decisions],
            ["a", "m", "z"],
        )
        self.assertEqual(manager.usage(MemoryTier.HBM), (20,))
        self.assertEqual(manager.usage(MemoryTier.HBF), (70,))
        self.assertEqual(
            [item.mode for item in plan.decisions],
            [
                PlacementMode.STATIC_HBF,
                PlacementMode.STATIC_HBF,
                PlacementMode.STATIC_HBM,
            ],
        )

    def test_weight_group_capacity_failure_leaves_no_partial_state(self):
        engine = TieringPolicyEngine(
            _config({"weights": {"policy": "hbm_only"}}),
            manager := _manager(hbm=(50,)),
        )

        with self.assertRaisesRegex(TieringPolicyError, "容量不足"):
            engine.place_weights(
                (
                    _weight("first", 30, "embedding"),
                    _weight("second", 30, "lm_head"),
                )
            )

        self.assertEqual(manager.usage(MemoryTier.HBM), (0,))
        self.assertEqual(manager.snapshot().records, {})

    def test_hbf_backed_weight_prefetch_only_plans_explicit_transfer(self):
        engine = TieringPolicyEngine(
            _config(
                {
                    "weights": {
                        "policy": "hbf_backed_hbm_cache",
                        "hbm_high_watermark": 0.9,
                        "hbm_low_watermark": 0.5,
                    }
                }
            ),
            manager := _manager(),
        )
        key = _weight("block0", 40, "qkv_proj", 0)
        engine.place_weights((key,))

        plan = engine.plan_weight_promotions((key.key,))

        self.assertEqual(plan.explicit_transfer_bytes, 40)
        self.assertEqual(plan.decisions[0].action, PolicyAction.MIGRATE)
        self.assertEqual(plan.decisions[0].source, MemoryTier.HBF)
        self.assertEqual(plan.decisions[0].target, MemoryTier.HBM)
        self.assertEqual(manager.usage(MemoryTier.HBF), (40,))
        self.assertEqual(manager.reserved(MemoryTier.HBM), (40,))
        manager.commit_transfer(plan.transfers[0].transfer_id)
        self.assertEqual(manager.usage(MemoryTier.HBM), (40,))

    def test_weight_promotion_rejects_high_watermark_and_rolls_back(self):
        engine = TieringPolicyEngine(
            _config(
                {
                    "weights": {
                        "policy": "hbf_backed_hbm_cache",
                        "hbm_high_watermark": 0.5,
                        "hbm_low_watermark": 0.25,
                    }
                }
            ),
            manager := _manager(),
        )
        request = _weight("large", 60, "qkv_proj", 0)
        engine.place_weights((request,))

        with self.assertRaisesRegex(TieringPolicyError, "high watermark"):
            engine.plan_weight_promotions((request.key,))

        self.assertEqual(manager.reserved(MemoryTier.HBM), (0,))
        self.assertEqual(
            manager.record(request.key).state,
            ResidencyState.RESIDENT,
        )
        self.assertEqual(manager.record(request.key).tier, MemoryTier.HBF)

    def test_weight_rebalance_uses_stable_lru_order(self):
        engine = TieringPolicyEngine(
            _config(
                {
                    "weights": {
                        "policy": "hbf_backed_hbm_cache",
                        "hbm_high_watermark": 0.8,
                        "hbm_low_watermark": 0.4,
                    }
                }
            ),
            manager := _manager(),
        )
        keys = [
            MemoryObjectKey(MemoryObjectKind.WEIGHT, name)
            for name in ("c", "a", "b")
        ]
        for key in keys:
            manager.register(key, (30,), MemoryTier.HBM)

        plan = engine.rebalance_weights()

        self.assertEqual(
            [operation.object_key.object_id for operation in plan.transfers],
            ["a", "b"],
        )
        self.assertEqual(plan.explicit_transfer_bytes, 60)


class KVTieringPolicyTest(unittest.TestCase):
    def test_static_kv_policies_never_silently_change_tier(self):
        cases = (
            ("hbm_only", MemoryTier.HBM),
            ("hbf_only", MemoryTier.HBF),
        )
        for policy, expected in cases:
            with self.subTest(policy=policy):
                manager = _manager()
                engine = TieringPolicyEngine(
                    _config({"kv": {"policy": policy}}),
                    manager,
                )
                request = _cache(
                    MemoryObjectKind.KV,
                    policy,
                    30,
                    16,
                    layer=0,
                )

                plan = engine.admit_kv(request)

                self.assertEqual(plan.decisions[0].target, expected)
                self.assertEqual(manager.record(request.key).tier, expected)

    def test_length_threshold_uses_hbf_for_long_context(self):
        engine = TieringPolicyEngine(
            _config(
                {
                    "kv": {
                        "policy": "length_threshold",
                        "threshold_tokens": 128,
                    }
                }
            ),
            manager := _manager(),
        )
        short = _cache(MemoryObjectKind.KV, "short", 20, 64, layer=0)
        long = _cache(MemoryObjectKind.KV, "long", 30, 128, layer=0)

        engine.admit_kv(short)
        engine.admit_kv(long)

        self.assertEqual(manager.record(short.key).tier, MemoryTier.HBM)
        self.assertEqual(manager.record(long.key).tier, MemoryTier.HBF)

    def test_watermark_lru_returns_deferred_registration(self):
        engine = TieringPolicyEngine(
            _config(
                {
                    "kv": {
                        "policy": "watermark_lru",
                        "hbm_high_watermark": 0.8,
                        "hbm_low_watermark": 0.5,
                    }
                }
            ),
            manager := _manager(),
        )
        old = _cache(MemoryObjectKind.KV, "old", 60, 64, layer=0)
        new = _cache(MemoryObjectKind.KV, "new", 30, 64, layer=0)
        engine.admit_kv(old)

        plan = engine.admit_kv(new)

        self.assertEqual(
            [item.action for item in plan.decisions],
            [PolicyAction.MIGRATE, PolicyAction.DEFERRED_REGISTER],
        )
        self.assertEqual(plan.transfers[0].object_key, old.key)
        self.assertNotIn(new.key, manager.snapshot().records)
        with self.assertRaisesRegex(TieringPolicyError, "先提交"):
            engine.complete_deferred(plan)

        manager.commit_transfer(plan.transfers[0].transfer_id)
        completed = engine.complete_deferred(plan)
        self.assertEqual(completed.decisions[0].action, PolicyAction.REGISTER)
        self.assertEqual(manager.record(old.key).tier, MemoryTier.HBF)
        self.assertEqual(manager.record(new.key).tier, MemoryTier.HBM)

    def test_watermark_without_evictable_candidate_spills_new_kv_to_hbf(self):
        engine = TieringPolicyEngine(
            _config(
                {
                    "kv": {
                        "policy": "watermark_lru",
                        "hbm_high_watermark": 0.5,
                        "hbm_low_watermark": 0.25,
                    }
                }
            ),
            manager := _manager(),
        )
        request = _cache(MemoryObjectKind.KV, "new", 60, 64, layer=0)

        plan = engine.admit_kv(request)

        self.assertEqual(plan.decisions[0].mode, PlacementMode.HBM_FIRST)
        self.assertEqual(plan.decisions[0].target, MemoryTier.HBF)
        self.assertEqual(manager.record(request.key).tier, MemoryTier.HBF)

    def test_insufficient_static_capacity_is_strictly_rejected(self):
        engine = TieringPolicyEngine(
            _config({"kv": {"policy": "hbm_only"}}),
            manager := _manager(hbm=(20,)),
        )
        request = _cache(MemoryObjectKind.KV, "too-large", 30, 16, layer=0)

        with self.assertRaisesRegex(TieringPolicyError, "容量不足"):
            engine.admit_kv(request)

        self.assertNotIn(request.key, manager.snapshot().records)

    def test_existing_kv_must_match_static_policy_and_be_resident(self):
        engine = TieringPolicyEngine(
            _config({"kv": {"policy": "hbm_only"}}),
            manager := _manager(),
        )
        request = _cache(
            MemoryObjectKind.KV,
            "mismatch",
            20,
            16,
            layer=0,
        )
        manager.register(request.key, request.bytes_per_rank, MemoryTier.HBF)

        with self.assertRaisesRegex(TieringPolicyError, "实际不在 HBM"):
            engine.admit_kv(request)

        manager = _manager()
        engine = TieringPolicyEngine(
            _config({"kv": {"policy": "hbm_only"}}),
            manager,
        )
        manager.register(request.key, request.bytes_per_rank, MemoryTier.HBM)
        manager.plan_transfers(
            [
                (
                    request.key,
                    MemoryTier.HBF,
                    "test",
                    "batch_start",
                    "batch_end",
                )
            ]
        )
        with self.assertRaisesRegex(TieringPolicyError, "迁移中"):
            engine.admit_kv(request)

    def test_cpu_fallback_is_rejected_without_explicit_staging(self):
        engine = TieringPolicyEngine(
            _config(
                {
                    "kv": {
                        "policy": "length_threshold",
                        "threshold_tokens": 16,
                    },
                    "transfer": {"capacity_fallback": "cpu"},
                }
            ),
            _manager(hbf=(20,)),
        )
        request = _cache(
            MemoryObjectKind.KV,
            "offload",
            30,
            16,
            layer=0,
        )

        with self.assertRaisesRegex(TieringPolicyError, "offload/staging"):
            engine.admit_kv(request)


class PrefixTieringPolicyTest(unittest.TestCase):
    def test_static_prefix_policies_use_only_the_configured_tier(self):
        cases = (
            ("hbm_only", MemoryTier.HBM),
            ("hbf_only", MemoryTier.HBF),
        )
        for policy, expected in cases:
            with self.subTest(policy=policy):
                manager = _manager()
                engine = TieringPolicyEngine(
                    _config({"prefix": {"policy": policy}}),
                    manager,
                )
                request = _cache(
                    MemoryObjectKind.PREFIX,
                    policy,
                    20,
                    32,
                )

                plan = engine.admit_prefix(request)

                self.assertEqual(plan.decisions[0].target, expected)
                self.assertEqual(manager.record(request.key).tier, expected)

    def test_hbf_backed_prefix_promotes_only_after_hit_threshold(self):
        engine = TieringPolicyEngine(
            _config(
                {
                    "prefix": {
                        "policy": "hbf_backed_hbm_hot",
                        "promotion_hits": 2,
                    }
                }
            ),
            manager := _manager(),
        )
        request = _cache(
            MemoryObjectKind.PREFIX,
            "shared",
            25,
            64,
        )
        engine.admit_prefix(request)

        cold = engine.access_prefix(request.key, hit_count=1)
        hot = engine.access_prefix(request.key, hit_count=2)

        self.assertEqual(cold.decisions[0].action, PolicyAction.KEEP)
        self.assertEqual(hot.decisions[0].action, PolicyAction.MIGRATE)
        self.assertEqual(hot.explicit_transfer_bytes, 25)
        manager.commit_transfer(hot.transfers[0].transfer_id)
        self.assertEqual(manager.record(request.key).tier, MemoryTier.HBM)

    def test_prefix_promotion_stays_in_hbf_under_hbm_pressure(self):
        engine = TieringPolicyEngine(
            _config(
                {
                    "prefix": {
                        "policy": "hbf_backed_hbm_hot",
                        "promotion_hits": 1,
                        "hbm_high_watermark": 0.5,
                        "hbm_low_watermark": 0.25,
                    }
                }
            ),
            manager := _manager(),
        )
        manager.register(
            MemoryObjectKey(MemoryObjectKind.WEIGHT, "resident"),
            (50,),
            MemoryTier.HBM,
        )
        request = _cache(
            MemoryObjectKind.PREFIX,
            "hot",
            20,
            64,
        )
        engine.admit_prefix(request)

        plan = engine.access_prefix(request.key, hit_count=1)

        self.assertEqual(plan.decisions[0].action, PolicyAction.KEEP)
        self.assertIn("deferred_capacity", plan.decisions[0].reason)
        self.assertEqual(manager.reserved(MemoryTier.HBM), (0,))

    def test_instance_affinity_uses_hbm_first_then_hbf(self):
        engine = TieringPolicyEngine(
            _config(
                {
                    "prefix": {
                        "policy": "instance_affinity",
                        "hbm_high_watermark": 0.8,
                        "hbm_low_watermark": 0.5,
                    }
                }
            ),
            manager := _manager(),
        )
        first = _cache(MemoryObjectKind.PREFIX, "first", 70, 64)
        second = _cache(MemoryObjectKind.PREFIX, "second", 20, 64)

        engine.admit_prefix(first)
        engine.admit_prefix(second)

        self.assertEqual(manager.record(first.key).tier, MemoryTier.HBM)
        self.assertEqual(manager.record(second.key).tier, MemoryTier.HBF)
        kept = engine.access_prefix(second.key, hit_count=10)
        self.assertEqual(kept.decisions[0].action, PolicyAction.KEEP)
        self.assertEqual(kept.decisions[0].mode, PlacementMode.HBM_FIRST)

    def test_prefix_rebalance_uses_stable_lru_order(self):
        engine = TieringPolicyEngine(
            _config(
                {
                    "prefix": {
                        "policy": "hbf_backed_hbm_hot",
                        "promotion_hits": 1,
                        "hbm_high_watermark": 0.8,
                        "hbm_low_watermark": 0.4,
                    }
                }
            ),
            manager := _manager(),
        )
        keys = [
            MemoryObjectKey(MemoryObjectKind.PREFIX, name)
            for name in ("c", "a", "b")
        ]
        for key in keys:
            manager.register(key, (30,), MemoryTier.HBM)

        plan = engine.rebalance_prefix()

        self.assertEqual(
            [operation.object_key.object_id for operation in plan.transfers],
            ["a", "b"],
        )

    def test_wrong_object_kind_is_rejected(self):
        engine = TieringPolicyEngine(_config(), _manager())
        kv = _cache(MemoryObjectKind.KV, "kv", 10, 16, layer=0)
        prefix = _cache(MemoryObjectKind.PREFIX, "prefix", 10, 16)

        with self.assertRaisesRegex(TieringPolicyError, "KV"):
            engine.admit_kv(prefix)
        with self.assertRaisesRegex(TieringPolicyError, "PREFIX"):
            engine.admit_prefix(kv)

    def test_prefix_demand_rejects_pending_migration(self):
        engine = TieringPolicyEngine(
            _config({"prefix": {"policy": "hbm_only"}}),
            manager := _manager(),
        )
        request = _cache(
            MemoryObjectKind.PREFIX,
            "pending",
            10,
            16,
        )
        manager.register(request.key, request.bytes_per_rank, MemoryTier.HBM)
        manager.plan_transfers(
            [
                (
                    request.key,
                    MemoryTier.HBF,
                    "test",
                    "batch_start",
                    "batch_end",
                )
            ]
        )

        with self.assertRaisesRegex(TieringPolicyError, "迁移中"):
            engine.access_prefix(request.key, hit_count=1)


if __name__ == "__main__":
    unittest.main()
