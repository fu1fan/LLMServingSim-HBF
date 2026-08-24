import unittest
from types import SimpleNamespace

from serving.core.hbf_performance import ScaleHBFPerformanceSource
from serving.core.trace_generator import (
    _apply_hbf_performance,
    _counts_as_dram_weight_traffic,
)


class CapturingSource:
    def __init__(self):
        self.query = None

    def latency_ns(self, query):
        self.query = query
        return query.baseline_latency_ns


def _context(source):
    return SimpleNamespace(
        model="facebook/opt-125m",
        perf_db={"variant": "bf16"},
        tp_size=1,
        ep_total=1,
        hbf_performance_source=source,
    )


def _batch():
    return SimpleNamespace(
        total_len=128,
        lm_head_len=4,
        prefill_chunk=128,
        kv_prefill=0,
        n_decode=0,
        kv_decode_mean=0,
        kv_decode_max=0,
        kv_decode_min=0,
    )


class HBFTraceIntegrationTest(unittest.TestCase):
    def test_only_nonempty_hbf_weight_operators_are_scaled(self):
        ctx = _context(ScaleHBFPerformanceSource(2.0))
        bctx = _batch()

        self.assertEqual(
            _apply_hbf_performance(
                ctx, bctx, "qkv_proj", "dense", 100, 4096, "HBF"
            ),
            200,
        )
        self.assertEqual(
            _apply_hbf_performance(
                ctx, bctx, "qkv_proj", "dense", 100, 4096, "LOCAL"
            ),
            100,
        )
        self.assertEqual(
            _apply_hbf_performance(
                ctx, bctx, "attention", "attention", 100, 0, "HBF"
            ),
            100,
        )

    def test_hbf_weights_require_an_explicit_source(self):
        with self.assertRaisesRegex(RuntimeError, "no performance source"):
            _apply_hbf_performance(
                _context(None),
                _batch(),
                "qkv_proj",
                "dense",
                100,
                4096,
                "HBF",
            )

    def test_hbf_bytes_are_not_counted_as_dram_energy(self):
        self.assertFalse(_counts_as_dram_weight_traffic("LOCAL"))
        self.assertFalse(_counts_as_dram_weight_traffic("HBF"))
        self.assertTrue(_counts_as_dram_weight_traffic("REMOTE:0"))
        self.assertTrue(_counts_as_dram_weight_traffic("CXL:0"))

    def test_moe_query_uses_post_routing_local_tokens(self):
        source = CapturingSource()
        _apply_hbf_performance(
            _context(source),
            _batch(),
            "moe",
            "moe",
            100,
            4096,
            "HBF",
            activated_experts=3,
            tokens=7,
        )
        self.assertEqual(source.query.shape_key["tokens"], 7)
        self.assertEqual(
            source.query.shape_key["activated_experts"], 3
        )


if __name__ == "__main__":
    unittest.main()
