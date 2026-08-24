import unittest
from types import SimpleNamespace

from serving.core.hbf_summary import build_hbf_runtime_summary


class HBFRuntimeSummaryTest(unittest.TestCase):
    def test_runtime_summary_exports_capacity_and_evidence(self):
        hbf_memory = SimpleNamespace(capacity_bytes=4096)
        memory = SimpleNamespace(
            npu_mem=1024,
            hbm_weight=128,
            hbf_weight=512,
            hbf_memory=hbf_memory,
            weight_residency_by_pp_rank=[
                {
                    "pp_rank": 0,
                    "hbm_weight_used_bytes": 128,
                    "hbf_weight_used_bytes": 512,
                }
            ],
        )
        instance = {
            "model_name": "model",
            "hardware": "gpu",
            "num_npus": 2,
            "tp_size": 2,
            "pp_size": 1,
            "ep_total": 1,
            "hbf_mem": {
                "performance": {
                    "source": "scale",
                    "latency_scale": 1.5,
                }
            },
        }

        result = build_hbf_runtime_summary(
            [instance],
            [SimpleNamespace(memory=memory)],
            num_devices=2,
            run_id="test",
        )

        self.assertTrue(result["hbf_energy_unmodeled"])
        self.assertEqual(result["num_devices"], 2)
        row = result["instances"][0]
        self.assertEqual(row["hbm_kv_capacity_bytes"], 896)
        self.assertEqual(row["hbf_weight_used_bytes"], 512)
        self.assertEqual(row["latency_scale"], 1.5)
        self.assertEqual(row["evidence_level"], "sensitivity-analysis")
        self.assertEqual(
            row["weight_residency_by_pp_rank"][0]["pp_rank"], 0
        )


if __name__ == "__main__":
    unittest.main()
