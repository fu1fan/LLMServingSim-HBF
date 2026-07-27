import unittest
from pathlib import Path

import numpy as np

from serving.core.expert_routing_profile import load_expert_routing_profile
from serving.core.gate_function import GateRouter


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = (
    ROOT / "configs" / "routing" / "qwen3_public_calibrated_v1.json"
)


def _metrics(counts):
    values = np.asarray(counts, dtype=np.float64)
    descending = np.sort(values)[::-1]
    ascending = descending[::-1]
    n = len(values)
    total = float(values.sum())
    gini = (
        2.0 * np.dot(np.arange(1, n + 1), ascending) / (n * total)
        - (n + 1) / n
    )
    return {
        "cv": float(values.std() / values.mean()),
        "gini": float(gini),
        "top5": float(descending[:5].sum() / total),
        "top10": float(descending[:10].sum() / total),
    }


class Qwen3RoutingProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_expert_routing_profile(
            PROFILE_PATH,
            "Qwen/Qwen3-235B-A22B",
            128,
            8,
        )

    def test_public_counts_match_published_summary(self):
        self.assertEqual(sum(self.profile.source_counts), 49920)
        metrics = _metrics(self.profile.source_counts)
        self.assertAlmostEqual(metrics["cv"], 0.55, delta=0.01)
        self.assertAlmostEqual(metrics["gini"], 0.30, delta=0.01)
        self.assertAlmostEqual(metrics["top5"], 0.099, delta=0.002)
        self.assertAlmostEqual(metrics["top10"], 0.177, delta=0.002)

    def test_runtime_sampler_matches_calibration_targets(self):
        observed = []
        for seed in range(8):
            gate = GateRouter(
                0,
                0,
                128,
                8,
                "CUSTOM",
                seed=seed,
                block_copy=False,
                custom_profile=self.profile,
            )
            observed.append(
                _metrics(gate.route(0, f"calibration-{seed}", 6240))
            )

        means = {
            key: float(np.mean([row[key] for row in observed]))
            for key in observed[0]
        }
        self.assertAlmostEqual(means["cv"], 0.55, delta=0.03)
        self.assertAlmostEqual(means["gini"], 0.30, delta=0.02)
        self.assertAlmostEqual(means["top5"], 0.099, delta=0.01)
        self.assertAlmostEqual(means["top10"], 0.177, delta=0.015)


if __name__ == "__main__":
    unittest.main()
