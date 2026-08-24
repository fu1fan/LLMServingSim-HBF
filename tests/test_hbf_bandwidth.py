import unittest

from serving.core.hbf_model import (
    GB_TO_BYTE,
    HBFConfig,
    lower_hbf_trace_location,
    parse_hbf_config,
)
from serving.core.hbf_performance import (
    BandwidthHBFPerformanceSource,
    HBFOperatorQuery,
    build_hbf_performance_source,
)


def _query(baseline_ns=1000, weight_bytes=1 << 20):
    return HBFOperatorQuery(
        model="m",
        variant="v",
        tp_size=1,
        ep_size=1,
        layer_name="gate_up_proj",
        category="dense",
        shape={"tokens": 2048},
        baseline_latency_ns=baseline_ns,
        weight_bytes=weight_bytes,
        weight_location="HBF",
    )


class HBFBandwidthConfigTest(unittest.TestCase):
    def test_bandwidth_source_requires_positive_mem_bw(self):
        value = {
            "schema_version": 1,
            "num_stacks": 8,
            "stack_capacity_gb": 512,
            "performance": {"source": "bandwidth"},
        }
        with self.assertRaisesRegex(ValueError, "mem_bw"):
            parse_hbf_config(value)

        value["mem_bw"] = 0
        with self.assertRaisesRegex(ValueError, "mem_bw"):
            parse_hbf_config(value)

        value["mem_bw"] = 64
        config = parse_hbf_config(value)
        self.assertIsInstance(config, HBFConfig)
        self.assertEqual(config.mem_bw, 64.0)
        self.assertEqual(config.mem_latency, 0.0)

    def test_mem_latency_must_be_non_negative(self):
        value = {
            "schema_version": 1,
            "num_stacks": 8,
            "stack_capacity_gb": 512,
            "performance": {"source": "bandwidth"},
            "mem_bw": 64,
            "mem_latency": -1,
        }
        with self.assertRaisesRegex(ValueError, "mem_latency"):
            parse_hbf_config(value)

    def test_scale_source_does_not_require_mem_bw(self):
        config = parse_hbf_config({
            "schema_version": 1,
            "num_stacks": 8,
            "stack_capacity_gb": 512,
            "performance": {"source": "scale", "latency_scale": 1.5},
        })
        self.assertEqual(config.mem_bw, 0.0)


class HBFBandwidthPerformanceTest(unittest.TestCase):
    def test_source_selection(self):
        source = build_hbf_performance_source(
            {
                "performance": {"source": "bandwidth"},
                "mem_bw": 64,
                "mem_latency": 500,
            },
            model="m",
            variant="v",
            tp_needed=[1],
            model_type="llama",
        )
        self.assertIsInstance(source, BandwidthHBFPerformanceSource)
        self.assertTrue(source.handles_weight_traffic)

    def test_latency_is_unchanged_and_weight_traffic_is_external(self):
        source = BandwidthHBFPerformanceSource(mem_bw=64, mem_latency=500)
        query = _query(baseline_ns=1234)
        # Compute-only: the weight-load penalty is modeled by ASTRA, not here.
        self.assertEqual(source.latency_ns(query), 1234)
        self.assertTrue(source.handles_weight_traffic)

    def test_invalid_bandwidth_is_rejected(self):
        with self.assertRaises(ValueError):
            BandwidthHBFPerformanceSource(mem_bw=0, mem_latency=0)
        with self.assertRaises(ValueError):
            BandwidthHBFPerformanceSource(mem_bw=64, mem_latency=-1)


class HBFTraceLocationTest(unittest.TestCase):
    def test_hbf_is_lowered_to_local_unless_handling_traffic(self):
        self.assertEqual(lower_hbf_trace_location("HBF"), "LOCAL")
        self.assertEqual(
            lower_hbf_trace_location("HBF", handles_weight_traffic=False),
            "LOCAL",
        )
        self.assertEqual(
            lower_hbf_trace_location("HBF", handles_weight_traffic=True),
            "HBF",
        )
        self.assertEqual(lower_hbf_trace_location("CXL"), "CXL")


if __name__ == "__main__":
    unittest.main()
