import unittest
from types import SimpleNamespace

from serving.core.memory_tiering import MemoryTier
from serving.core.residency_scenario import BatchMemoryView
from serving.core.trace_generator import (
    BatchCtx,
    _attention_residency_groups,
    _tier_transfer_rows,
)


class _Request:
    def __init__(self, request_id, prefill):
        self.id = request_id
        self._prefill = prefill

    def is_prefill(self):
        return self._prefill


class ResidencyTraceTest(unittest.TestCase):
    def test_explicit_transfer_encodes_source_read_and_target_write(self):
        batch = SimpleNamespace(
            memory_transfers=(
                SimpleNamespace(
                    transfer_id=9,
                    source=MemoryTier.HBF,
                    target=MemoryTier.HBM,
                    bytes_per_rank=4096,
                    pp_stage=1,
                ),
            )
        )

        rows = _tier_transfer_rows(batch)

        self.assertEqual(rows[0][0], "tier_transfer_9")
        self.assertEqual(rows[0][2:4], ["HBF", "4096"])
        self.assertEqual(rows[0][6:8], ["LOCAL", "4096"])
        self.assertEqual(rows[0][10], "PP_STAGE:1")

    def test_uneven_rank_transfer_is_rejected(self):
        batch = SimpleNamespace(
            memory_transfers=(
                SimpleNamespace(
                    transfer_id=1,
                    source=MemoryTier.HBM,
                    target=MemoryTier.HBF,
                    bytes_per_rank=(1024, 2048),
                ),
            )
        )

        with self.assertRaisesRegex(RuntimeError, "每个 rank"):
            _tier_transfer_rows(batch)

    def test_attention_is_grouped_by_request_layer_residency(self):
        batch = SimpleNamespace(
            requests=[_Request(10, True), _Request(11, False)],
            prefill_q_list=[4],
            prefill_k_list=[8],
            decode_k_list=[32],
        )
        bctx = BatchCtx(
            batch=batch,
            total_len=5,
            prefill_chunk=4,
            kv_prefill=8,
            n_decode=1,
            kv_decode_mean=32,
            kv_decode_max=32,
            kv_decode_min=32,
            lm_head_len=2,
            decode_lens=None,
            channel_split=0,
        )
        ctx = SimpleNamespace(
            memory_scenario_policy=SimpleNamespace(
                is_residency_derived=True
            ),
            memory_view=BatchMemoryView(
                snapshot_version=1,
                weight_tiers={("attention", 3): MemoryTier.HBM},
                kv_tiers={
                    ("10", 3): MemoryTier.HBF,
                    ("11", 3): MemoryTier.HBM,
                },
            ),
        )

        groups = _attention_residency_groups(ctx, bctx, 3)

        self.assertEqual(
            [tier for _, tier in groups],
            [MemoryTier.HBF, MemoryTier.HBM],
        )
        hbf, hbm = (item for item, _ in groups)
        self.assertEqual(
            (hbf.prefill_chunk, hbf.kv_prefill, hbf.n_decode),
            (4, 8, 0),
        )
        self.assertEqual(
            (hbm.prefill_chunk, hbm.n_decode, hbm.kv_decode_mean),
            (0, 1, 32),
        )

    def test_attention_requires_block_identity(self):
        batch = SimpleNamespace(
            requests=[_Request(1, False)],
            prefill_q_list=[],
            prefill_k_list=[],
            decode_k_list=[16],
        )
        bctx = BatchCtx(
            batch=batch,
            total_len=1,
            prefill_chunk=0,
            kv_prefill=0,
            n_decode=1,
            kv_decode_mean=16,
            kv_decode_max=16,
            kv_decode_min=16,
            lm_head_len=1,
            decode_lens=None,
            channel_split=0,
        )
        ctx = SimpleNamespace(
            memory_scenario_policy=SimpleNamespace(
                is_residency_derived=True
            ),
            memory_view=None,
        )

        with self.assertRaisesRegex(RuntimeError, "block"):
            _attention_residency_groups(ctx, bctx, None)


if __name__ == "__main__":
    unittest.main()
