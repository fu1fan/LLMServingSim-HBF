import unittest

from serving.core.memory_tiering import (
    MemoryObjectKey,
    MemoryObjectKind,
    MemoryTier,
    ResidencyState,
    TieredResidencyManager,
    TieringError,
)


def _manager():
    return TieredResidencyManager(
        {
            MemoryTier.HBM: (100, 80),
            MemoryTier.HBF: (300, 300),
        },
        num_ranks=2,
    )


class TieredResidencyManagerTest(unittest.TestCase):
    def test_register_tracks_capacity_per_rank(self):
        manager = _manager()
        key = MemoryObjectKey(MemoryObjectKind.WEIGHT, "layer0")

        manager.register(key, (40, 30), MemoryTier.HBM)

        self.assertEqual(manager.usage(MemoryTier.HBM), (40, 30))
        self.assertEqual(manager.available(MemoryTier.HBM), (60, 50))
        self.assertEqual(manager.snapshot().tier_of(key), MemoryTier.HBM)

    def test_transfer_reserves_target_until_commit(self):
        manager = _manager()
        key = MemoryObjectKey(MemoryObjectKind.KV, "request7", layer_index=3)
        manager.register(key, (40, 30), MemoryTier.HBM)

        operation = manager.plan_transfers(
            [
                (
                    key,
                    MemoryTier.HBF,
                    "kv_watermark",
                    "block:2/attention",
                    "block:3/attention",
                )
            ]
        )[0]

        self.assertEqual(manager.record(key).state, ResidencyState.TRANSFERRING)
        self.assertEqual(manager.usage(MemoryTier.HBM), (40, 30))
        self.assertEqual(manager.reserved(MemoryTier.HBF), (40, 30))
        manager.commit_transfer(operation.transfer_id)
        self.assertEqual(manager.usage(MemoryTier.HBM), (0, 0))
        self.assertEqual(manager.usage(MemoryTier.HBF), (40, 30))
        self.assertEqual(manager.reserved(MemoryTier.HBF), (0, 0))
        self.assertEqual(manager.record(key).tier, MemoryTier.HBF)

    def test_failed_group_reservation_is_atomic(self):
        manager = TieredResidencyManager(
            {
                MemoryTier.HBM: (100, 80),
                MemoryTier.HBF: (80, 70),
            },
            num_ranks=2,
        )
        first = MemoryObjectKey(MemoryObjectKind.KV, "first", layer_index=0)
        second = MemoryObjectKey(MemoryObjectKind.KV, "second", layer_index=0)
        manager.register(first, (60, 60), MemoryTier.HBM)
        manager.register(second, (40, 20), MemoryTier.HBM)

        with self.assertRaisesRegex(TieringError, "容量不足"):
            manager.plan_transfers(
                [
                    (first, MemoryTier.HBF, "test", "batch_start", "batch_end"),
                    (second, MemoryTier.HBF, "test", "batch_start", "batch_end"),
                ]
            )

        self.assertEqual(manager.reserved(MemoryTier.HBF), (0, 0))
        self.assertEqual(manager.record(first).state, ResidencyState.RESIDENT)
        self.assertEqual(manager.record(second).state, ResidencyState.RESIDENT)

    def test_abort_releases_reservation_without_moving_object(self):
        manager = _manager()
        key = MemoryObjectKey(MemoryObjectKind.PREFIX, "prefix-a")
        manager.register(key, (25, 25), MemoryTier.HBF)
        operation = manager.plan_transfers(
            [(key, MemoryTier.HBM, "promotion", "batch_start", "embedding")]
        )[0]

        manager.abort_transfer(operation.transfer_id)

        self.assertEqual(manager.record(key).tier, MemoryTier.HBF)
        self.assertEqual(manager.record(key).state, ResidencyState.RESIDENT)
        self.assertEqual(manager.reserved(MemoryTier.HBM), (0, 0))

    def test_locked_object_cannot_move_or_be_removed(self):
        manager = _manager()
        key = MemoryObjectKey(MemoryObjectKind.WEIGHT, "lm_head")
        manager.register(key, (20, 20), MemoryTier.HBF)
        manager.lock(key)

        with self.assertRaisesRegex(TieringError, "锁定"):
            manager.plan_transfers(
                [(key, MemoryTier.HBM, "prefetch", "batch_start", "lm_head")]
            )
        with self.assertRaisesRegex(TieringError, "锁定"):
            manager.remove(key)

        manager.unlock(key)
        manager.remove(key)
        self.assertNotIn(key, manager.snapshot().records)

    def test_snapshot_is_stable_after_future_changes(self):
        manager = _manager()
        key = MemoryObjectKey(MemoryObjectKind.KV, "request2", layer_index=1)
        manager.register(key, (10, 10), MemoryTier.HBM)
        snapshot = manager.snapshot()
        manager.touch(key)

        self.assertEqual(snapshot.records[key].last_access, 0)
        self.assertEqual(manager.record(key).last_access, 1)

    def test_rejects_per_rank_capacity_overflow(self):
        manager = _manager()
        key = MemoryObjectKey(MemoryObjectKind.WEIGHT, "oversized")

        with self.assertRaisesRegex(TieringError, "rank 1"):
            manager.register(key, (70, 90), MemoryTier.HBM)


if __name__ == "__main__":
    unittest.main()
