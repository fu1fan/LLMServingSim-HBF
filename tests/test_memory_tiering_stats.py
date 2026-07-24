import json
import unittest

from serving.core.memory_tiering import (
    MemoryObjectKey,
    MemoryObjectKind,
    MemoryTier,
    TieredResidencyManager,
)
from serving.core.memory_tiering_stats import MemoryTieringStats


class MemoryTieringStatsCapacityTest(unittest.TestCase):
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

    def test_snapshot_is_immutable_and_json_friendly(self):
        stats = MemoryTieringStats(num_ranks=1)
        snapshot = stats.snapshot()

        with self.assertRaises(TypeError):
            snapshot.resident_high_water_bytes[MemoryTier.HBM] = (1,)
        payload = snapshot.to_dict()
        self.assertEqual(payload["schema"], "llmservingsim_memory_tiering_stats_v1")
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
