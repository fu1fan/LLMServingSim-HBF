import unittest

from serving.core.hbf_performance import (
    HBFOperatorQuery,
    IdentityHBFPerformanceSource,
)


def _query(**overrides):
    values = {
        "model": "meta-llama/Llama-3.1-8B",
        "variant": "bf16",
        "tp_size": 1,
        "ep_size": 1,
        "layer_name": "qkv_proj",
        "category": "dense",
        "shape": {"tokens": 16},
        "baseline_latency_ns": 1000,
        "weight_bytes": 4096,
        "weight_location": "HBF",
    }
    values.update(overrides)
    return HBFOperatorQuery(**values)


class HBFPerformanceInterfaceTest(unittest.TestCase):
    def test_query_exposes_stable_read_only_shape_context(self):
        shape = {"tokens": 16}
        query = _query(shape=shape)
        shape["tokens"] = 32

        self.assertEqual(query.shape_key["tokens"], 16)
        with self.assertRaises(TypeError):
            query.shape_key["tokens"] = 64

    def test_query_identifies_only_nonempty_hbf_weights(self):
        self.assertTrue(_query().uses_hbf_weights)
        self.assertFalse(_query(weight_bytes=0).uses_hbf_weights)
        self.assertFalse(_query(weight_location="LOCAL").uses_hbf_weights)

    def test_identity_source_preserves_baseline_latency(self):
        source = IdentityHBFPerformanceSource()
        query = _query(baseline_latency_ns=1234)

        self.assertEqual(source.latency_ns(query), 1234)
        self.assertEqual(source.describe()["evidence_level"], "identity")

    def test_negative_latency_and_weight_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "baseline_latency"):
            _query(baseline_latency_ns=-1)
        with self.assertRaisesRegex(ValueError, "weight_bytes"):
            _query(weight_bytes=-1)


if __name__ == "__main__":
    unittest.main()
