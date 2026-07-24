import json
import unittest

from serving.core.memory_tiering import (
    MemoryObjectKey,
    MemoryObjectKind,
    MemoryTier,
    TieredResidencyManager,
    TransferOperation,
)
from serving.core.memory_tiering_stats import MemoryTieringStats


class MemoryTieringStatsCapacityTest(unittest.TestCase):
    def test_runtime_usage_without_residency_manager_is_supported(self):
        stats = MemoryTieringStats(num_ranks=2)
        stats.observe_usage(
            {
                MemoryTier.HBM: (30, 40),
                MemoryTier.HBF: (50, 60),
            }
        )
        stats.record_explicit_transfer(
            source=MemoryTier.HBF,
            target=MemoryTier.HBM,
            bytes_per_rank=(5, 8),
            reason="kv_threshold",
            object_kind=MemoryObjectKind.KV,
            layer_index=2,
        )

        snapshot = stats.snapshot()
        self.assertEqual(
            snapshot.resident_high_water_bytes[MemoryTier.HBF],
            (50, 60),
        )
        self.assertEqual(
            snapshot.transfers_by_layer[2].bytes_per_rank,
            (5, 8),
        )

    def test_tracks_resident_and_reserved_high_water_per_rank(self):
        manager = TieredResidencyManager(
            {
                MemoryTier.HBM: (200, 180),
                MemoryTier.HBF: (400, 400),
            },
            num_ranks=2,
        )
        key = MemoryObjectKey(MemoryObjectKind.KV, "request-1", layer_index=0)
        manager.register(key, (40, 30), MemoryTier.HBM)
        stats = MemoryTieringStats(num_ranks=2)
        stats.observe_residency(manager.snapshot())
        operation = manager.plan_transfers(
            [(key, MemoryTier.HBF, "kv_demote", "batch:0", "batch:2")]
        )[0]
        stats.observe_residency(manager.snapshot())

        snapshot = stats.snapshot()
        self.assertEqual(
            snapshot.resident_high_water_bytes[MemoryTier.HBM],
            (40, 30),
        )
        self.assertEqual(
            snapshot.capacity_high_water_bytes[MemoryTier.HBF],
            (40, 30),
        )
        manager.commit_transfer(operation.transfer_id)
        stats.observe_residency(manager.snapshot())
        self.assertEqual(
            stats.snapshot().resident_high_water_bytes[MemoryTier.HBF],
            (40, 30),
        )

    def test_completed_explicit_transfers_are_counted_by_direction(self):
        stats = MemoryTieringStats(num_ranks=2)
        manager = TieredResidencyManager(
            {
                MemoryTier.HBM: (300, 300),
                MemoryTier.HBF: (300, 300),
                MemoryTier.CPU: (300, 300),
            },
            num_ranks=2,
        )
        hbm_key = MemoryObjectKey(MemoryObjectKind.WEIGHT, "layer-0")
        hbf_key = MemoryObjectKey(MemoryObjectKind.KV, "req", layer_index=0)
        manager.register(hbm_key, (20, 30), MemoryTier.HBM)
        manager.register(hbf_key, (10, 15), MemoryTier.HBF)
        hbm_to_hbf = manager.plan_transfers(
            [(hbm_key, MemoryTier.HBF, "demote", "a", "b")]
        )[0]
        manager.commit_transfer(hbm_to_hbf.transfer_id)
        stats.record_completed_transfer(hbm_to_hbf)
        hbf_to_cpu = manager.plan_transfers(
            [(hbf_key, MemoryTier.CPU, "evict", "a", "b")]
        )[0]
        manager.commit_transfer(hbf_to_cpu.transfer_id)
        stats.record_completed_transfer(hbf_to_cpu)

        directions = stats.snapshot().transfer_directions
        self.assertEqual(
            directions[(MemoryTier.HBM, MemoryTier.HBF)].bytes_per_rank,
            (20, 30),
        )
        self.assertEqual(
            directions[(MemoryTier.HBM, MemoryTier.HBF)].total_bytes,
            50,
        )
        self.assertEqual(
            directions[(MemoryTier.HBF, MemoryTier.CPU)].operations,
            1,
        )

    def test_explicit_transfers_are_broken_down_by_reason_kind_and_layer(self):
        stats = MemoryTieringStats(num_ranks=2)
        first = TransferOperation(
            transfer_id=1,
            object_key=MemoryObjectKey(
                MemoryObjectKind.KV,
                "request-1",
                layer_index=3,
            ),
            source=MemoryTier.HBF,
            target=MemoryTier.HBM,
            bytes_per_rank=(10, 20),
            reason="kv_promote",
        )
        second = TransferOperation(
            transfer_id=2,
            object_key=MemoryObjectKey(
                MemoryObjectKind.KV,
                "request-2",
                layer_index=3,
            ),
            source=MemoryTier.HBF,
            target=MemoryTier.HBM,
            bytes_per_rank=(5, 8),
            reason="kv_promote",
        )
        unscoped = TransferOperation(
            transfer_id=3,
            object_key=MemoryObjectKey(MemoryObjectKind.PREFIX, "prefix-1"),
            source=MemoryTier.CXL,
            target=MemoryTier.HBF,
            bytes_per_rank=(4, 4),
            reason="prefix_restore",
        )
        for operation in (first, second, unscoped):
            stats.record_completed_transfer(operation)

        snapshot = stats.snapshot()
        promoted = snapshot.transfers_by_reason["kv_promote"]
        self.assertEqual(promoted.operations, 2)
        self.assertEqual(promoted.bytes_per_rank, (15, 28))
        self.assertEqual(
            snapshot.transfers_by_object_kind[MemoryObjectKind.KV].total_bytes,
            43,
        )
        self.assertEqual(snapshot.transfers_by_layer[3].operations, 2)
        self.assertEqual(snapshot.transfers_by_layer[None].operations, 1)

    def test_records_policy_batch_hits_and_attention_residency_groups(self):
        stats = MemoryTieringStats(num_ranks=1)
        stats.record_policy_action("keep", count=2)
        stats.record_policy_action("migrate")
        stats.record_residency_batch(hit=True)
        stats.record_residency_batch(hit=False)
        stats.record_attention_groups(hbm_groups=1, hbf_groups=1)
        stats.record_attention_groups(hbm_groups=0, hbf_groups=1)

        snapshot = stats.snapshot()
        self.assertEqual(snapshot.policy_actions, {"keep": 2, "migrate": 1})
        self.assertEqual(snapshot.residency_batches, 2)
        self.assertEqual(snapshot.residency_hit_batches, 1)
        self.assertEqual(snapshot.attention_group_observations, 2)
        self.assertEqual(snapshot.attention_hbm_groups, 1)
        self.assertEqual(snapshot.attention_hbf_groups, 2)
        payload = snapshot.to_dict()
        self.assertEqual(payload["residency_batches"]["misses"], 1)
        self.assertEqual(payload["attention_groups"]["hbf"], 2)

    def test_non_transfer_signals_do_not_create_profile_demand_migrations(self):
        stats = MemoryTieringStats(num_ranks=1)
        stats.record_policy_action("keep")
        stats.record_residency_batch(hit=True)
        stats.record_attention_groups(hbm_groups=1, hbf_groups=1)

        self.assertEqual(stats.snapshot().transfer_directions, {})

    def test_snapshot_is_immutable_and_json_friendly(self):
        stats = MemoryTieringStats(num_ranks=1)
        snapshot = stats.snapshot()

        with self.assertRaises(TypeError):
            snapshot.resident_high_water_bytes[MemoryTier.HBM] = (1,)
        with self.assertRaises(TypeError):
            snapshot.policy_actions["migrate"] = 1
        payload = snapshot.to_dict()
        self.assertEqual(payload["schema"], "llmservingsim_memory_tiering_stats_v1")
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
