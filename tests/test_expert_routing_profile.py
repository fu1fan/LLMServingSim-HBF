import json
import tempfile
import unittest
from pathlib import Path

from serving.core.expert_routing_profile import (
    load_expert_routing_profile,
    validate_expert_routing_options,
)
from serving.core.gate_function import GateRouter


def _profile():
    return {
        "schema_version": 1,
        "profile_id": "fixture-v1",
        "target_model": "fixture/model",
        "num_experts": 4,
        "top_k": 2,
        "distribution": {
            "kind": "marginal_histogram",
            "source_counts": [8, 6, 4, 2],
            "selection_weights": [0.4, 0.3, 0.2, 0.1],
            "reference_tokens": 10,
            "layer_mapping": "seeded_permutation",
            "sampler": "plackett_luce_gumbel_topk",
        },
        "calibration": {
            "target_cv": 0.45,
            "target_gini": 0.25,
            "target_top5_share": 1.0,
            "target_top10_share": 1.0,
            "source_url": "https://example.test/routing",
            "source_scope": "unit-test fixture",
            "evidence_level": "synthetic-fixture",
        },
    }


class ExpertRoutingProfileTest(unittest.TestCase):
    def _write(self, root, data=None):
        path = Path(root) / "profile.json"
        path.write_text(
            json.dumps(data or _profile(), sort_keys=True),
            encoding="utf-8",
        )
        return path

    def test_loads_valid_profile_and_records_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = load_expert_routing_profile(
                self._write(tmp),
                target_model="fixture/model",
                num_experts=4,
                top_k=2,
            )

        self.assertEqual(profile.profile_id, "fixture-v1")
        self.assertEqual(profile.source_counts, (8.0, 6.0, 4.0, 2.0))
        self.assertEqual(len(profile.sha256), 64)

    def test_rejects_model_expert_and_top_k_mismatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp)
            cases = [
                ("other/model", 4, 2, "model mismatch"),
                ("fixture/model", 8, 2, "expert-count mismatch"),
                ("fixture/model", 4, 1, "top-k mismatch"),
            ]
            for model, experts, top_k, message in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        load_expert_routing_profile(
                            path, model, experts, top_k
                        )

    def test_rejects_invalid_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = _profile()
            data["distribution"]["selection_weights"][1] = 0
            with self.assertRaisesRegex(ValueError, "must be positive"):
                load_expert_routing_profile(
                    self._write(tmp, data),
                    "fixture/model",
                    4,
                    2,
                )

    def test_custom_cli_contract(self):
        with self.assertRaisesRegex(ValueError, "is required"):
            validate_expert_routing_options("CUSTOM", None, False)
        with self.assertRaisesRegex(ValueError, "no-enable-block-copy"):
            validate_expert_routing_options("CUSTOM", "profile.json", True)
        with self.assertRaisesRegex(ValueError, "may only be used"):
            validate_expert_routing_options(
                "BALANCED", "profile.json", True
            )
        validate_expert_routing_options(
            "CUSTOM", "profile.json", False
        )

    def test_gate_requires_profile_and_no_block_copy(self):
        with self.assertRaisesRegex(ValueError, "requires a calibrated"):
            GateRouter(0, 0, 4, 2, "CUSTOM", block_copy=False)
        with tempfile.TemporaryDirectory() as tmp:
            profile = load_expert_routing_profile(
                self._write(tmp), "fixture/model", 4, 2
            )
            with self.assertRaisesRegex(ValueError, "block_copy=False"):
                GateRouter(
                    0,
                    0,
                    4,
                    2,
                    "CUSTOM",
                    block_copy=True,
                    custom_profile=profile,
                )


if __name__ == "__main__":
    unittest.main()
