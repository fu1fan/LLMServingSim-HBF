import unittest

from serving.core.kv_policy import (
    recompute_cost_ns,
    should_recompute,
    spill_cost_ns,
)


class KVPolicyCostModelTest(unittest.TestCase):
    def test_spill_cost_is_round_trip(self):
        # 1000 bytes at 1000 GB/s == 1 ns transfer, + 100 ns latency, × 2.
        self.assertAlmostEqual(
            spill_cost_ns(1000, 1000.0, 100.0),
            2 * (1 + 100),
        )
        # Zero bytes is free (no latency charged).
        self.assertEqual(spill_cost_ns(0, 1000.0, 100.0), 0.0)

    def test_spill_cost_validates_inputs(self):
        with self.assertRaises(ValueError):
            spill_cost_ns(10, 0, 0)
        with self.assertRaises(ValueError):
            spill_cost_ns(10, 1000.0, -1)

    def test_recompute_cost_is_linear(self):
        self.assertEqual(recompute_cost_ns(10, 5.0), 50.0)
        self.assertEqual(recompute_cost_ns(0, 5.0), 0.0)
        with self.assertRaises(ValueError):
            recompute_cost_ns(-1, 5.0)

    def test_should_recompute_zero_tokens_always_recomputes(self):
        # Nothing to store → spilling would be pure waste.
        self.assertTrue(
            should_recompute(
                spill_bytes=1000,
                mem_bw_gb_s=100.0,
                mem_latency_ns=10,
                recompute_tokens=0,
                cost_per_token_ns=1000.0,
            )
        )

    def test_should_recompute_cheap_prefix(self):
        # Short hot prefix recomputes cheaper than a large spill round-trip.
        self.assertTrue(
            should_recompute(
                spill_bytes=1_000_000,
                mem_bw_gb_s=50.0,
                mem_latency_ns=1000,
                recompute_tokens=10,
                cost_per_token_ns=10.0,
            )
        )

    def test_should_recompute_expensive_prefix_spills(self):
        # A long prefix is cheaper to spill than to recompute.
        self.assertFalse(
            should_recompute(
                spill_bytes=1000,
                mem_bw_gb_s=1000.0,
                mem_latency_ns=10,
                recompute_tokens=100_000,
                cost_per_token_ns=10.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
