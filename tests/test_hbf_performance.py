import unittest
import csv
import tempfile
from pathlib import Path

import yaml

from serving.core.hbf_performance import (
    HBFOperatorQuery,
    IdentityHBFPerformanceSource,
    ProfileBundleHBFPerformanceSource,
    ScaleHBFPerformanceSource,
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


class ScaleHBFPerformanceTest(unittest.TestCase):
    def test_scale_multiplies_complete_operator_latency(self):
        source = ScaleHBFPerformanceSource(1.25)

        self.assertEqual(
            source.latency_ns(_query(baseline_latency_ns=1000)),
            1250,
        )
        self.assertEqual(source.describe(), {
            "source": "scale",
            "evidence_level": "sensitivity-analysis",
            "latency_scale": 1.25,
        })

    def test_identity_scale_is_numerically_exact(self):
        source = ScaleHBFPerformanceSource(1.0)

        self.assertEqual(
            source.latency_ns(_query(baseline_latency_ns=1234567)),
            1234567,
        )

    def test_invalid_scale_is_rejected(self):
        for value in (0, -1, True, "2"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ScaleHBFPerformanceSource(value)


class ProfileBundleHBFPerformanceTest(unittest.TestCase):
    def _bundle(self, root, *, producer="MacSim", include_layer=True):
        variant_root = (
            Path(root)
            / "B200_HBF"
            / "meta-llama"
            / "Llama-3.1-8B"
            / "bf16"
        )
        tp_dir = variant_root / "tp1"
        tp_dir.mkdir(parents=True)
        meta = {
            "hardware": "B200_HBF",
            "model": "meta-llama/Llama-3.1-8B",
            "variant": "bf16",
            "hbf_profile": {
                "schema_version": 1,
                "producer": producer,
                "source": "cycle-level-simulation",
            },
        }
        (variant_root / "meta.yaml").write_text(
            yaml.safe_dump(meta),
            encoding="utf-8",
        )
        with (tp_dir / "dense.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["layer", "tokens", "time_us"],
            )
            writer.writeheader()
            if include_layer:
                writer.writerow({
                    "layer": "qkv_proj",
                    "tokens": 16,
                    "time_us": 2.5,
                })
        return variant_root

    def _source(self, root):
        return ProfileBundleHBFPerformanceSource(
            profile_root=root,
            profile_hardware="B200_HBF",
            model="meta-llama/Llama-3.1-8B",
            variant="bf16",
            tp_needed={1},
            model_type="llama",
        )

    def test_external_bundle_returns_direct_operator_latency(self):
        with tempfile.TemporaryDirectory() as td:
            self._bundle(td)
            source = self._source(td)

            latency = source.latency_ns(_query())

            self.assertEqual(latency, 2500)
            self.assertEqual(
                source.describe()["evidence_level"],
                "external-simulator-backed",
            )

    def test_missing_layer_never_falls_back_to_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            self._bundle(td, include_layer=False)
            source = self._source(td)

            with self.assertRaisesRegex(KeyError, "Missing dense profile"):
                source.latency_ns(_query())

    def test_provenance_and_tp_coverage_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            self._bundle(td, producer="")
            with self.assertRaisesRegex(ValueError, "producer"):
                self._source(td)

        with tempfile.TemporaryDirectory() as td:
            self._bundle(td)
            with self.assertRaisesRegex(FileNotFoundError, "tp=\\[2\\]"):
                ProfileBundleHBFPerformanceSource(
                    profile_root=td,
                    profile_hardware="B200_HBF",
                    model="meta-llama/Llama-3.1-8B",
                    variant="bf16",
                    tp_needed={2},
                    model_type="llama",
                )


if __name__ == "__main__":
    unittest.main()
