import unittest

from serving.core.memory_model import (
    MemoryModel,
    moe_experts_per_rank,
    moe_gate_weight_bytes,
    moe_per_expert_weight_bytes,
    split_moe_residency_bytes,
)


QWEN_MOE = "Qwen/Qwen3-235B-A22B"


def _placement(weight_location="HBF"):
    return {
        "default": {
            "weights": weight_location,
            "kv_loc": "LOCAL",
            "kv_evict_loc": "REMOTE:0",
        },
        "block": [],
        "layer": {},
    }


def _hbf_config():
    return {
        "schema_version": 1,
        "num_stacks": 8,
        "stack_capacity_gb": 512,
        "mem_size": 4096,
        "performance": {"source": "scale", "latency_scale": 1.0},
    }


def _memory(moe_hot_expert_frac=1.0, placement=None):
    return MemoryModel(
        model=QWEN_MOE,
        instance_id=0,
        node_id=0,
        num_npus=4,
        tp_size=1,
        npu_mem=180,
        cpu_mem=512,
        block_size=16,
        fp=16,
        enable_prefix_caching=True,
        enable_prefix_sharing=False,
        prefix_pool=None,
        prefix_storage=None,
        ep_size=4,
        placement=placement or _placement(),
        hbf_mem=_hbf_config(),
        moe_hot_expert_frac=moe_hot_expert_frac,
    )


class MoEWeightSplitTest(unittest.TestCase):
    def test_per_expert_and_gate_weights_are_positive(self):
        self.assertGreater(moe_gate_weight_bytes(QWEN_MOE, fp=2), 0)
        self.assertGreater(moe_per_expert_weight_bytes(QWEN_MOE, fp=2), 0)
        self.assertGreaterEqual(moe_experts_per_rank(QWEN_MOE, 4), 1)

    def test_split_consistency(self):
        gate = moe_gate_weight_bytes(QWEN_MOE, fp=2)
        per_expert = moe_per_expert_weight_bytes(QWEN_MOE, fp=2)
        experts_per_rank = moe_experts_per_rank(QWEN_MOE, 4)

        hbm, hbf = split_moe_residency_bytes(QWEN_MOE, 4, 2, 1.0)
        self.assertEqual(hbm, gate + experts_per_rank * per_expert)
        self.assertEqual(hbf, 0)

        hbm, hbf = split_moe_residency_bytes(QWEN_MOE, 4, 2, 0.0)
        self.assertEqual(hbm, gate)
        self.assertEqual(hbf, experts_per_rank * per_expert)

        # Partial split is a monotonic function of the hot fraction.
        full_hbm, _ = split_moe_residency_bytes(QWEN_MOE, 4, 2, 1.0)
        half_hbm, half_hbf = split_moe_residency_bytes(QWEN_MOE, 4, 2, 0.5)
        self.assertLess(half_hbm, full_hbm)
        self.assertGreater(half_hbf, 0)


class MoEHotColdResidencyTest(unittest.TestCase):
    def test_default_full_offload_matches_legacy_hbf_weight(self):
        # Default frac=0.0 offloads all experts (gate stays in HBM), matching
        # the legacy "whole MoE layer in HBF" intent minus the tiny gate.
        memory = _memory(moe_hot_expert_frac=0.0)
        self.assertGreater(memory.hbf_weight, 0)

    def test_partial_hot_reduces_hbf_weight(self):
        all_cold = _memory(moe_hot_expert_frac=0.0)
        half_hot = _memory(moe_hot_expert_frac=0.5)

        # Keeping more experts hot (in HBM) shrinks the HBF residency.
        self.assertLess(half_hot.hbf_weight, all_cold.hbf_weight)
        self.assertGreater(half_hot.hbm_weight, all_cold.hbm_weight)
        # Gate + hot experts stay in HBM; only cold experts in HBF.
        self.assertGreater(half_hot.hbm_weight, 0)

    def test_all_hot_keeps_moe_in_hbm(self):
        all_hot = _memory(moe_hot_expert_frac=1.0)
        all_cold = _memory(moe_hot_expert_frac=0.0)
        # Non-MoE layers still offload, but MoE experts no longer do.
        self.assertLess(all_hot.hbf_weight, all_cold.hbf_weight)

    def test_invalid_hot_fraction_is_rejected(self):
        for frac in (-0.1, 1.1, float("nan")):
            with self.subTest(frac=frac):
                with self.assertRaises(ValueError):
                    _memory(moe_hot_expert_frac=frac)


if __name__ == "__main__":
    unittest.main()
