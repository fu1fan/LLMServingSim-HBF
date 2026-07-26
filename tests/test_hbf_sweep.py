import json
import tempfile
import unittest
from pathlib import Path

from scripts.hbf_sweep import (
    PAPER_CORE_SCALES,
    build_run,
    parse_scales,
    validate_scale_cluster,
)


class HBFSweepTest(unittest.TestCase):
    def test_default_and_custom_scales(self):
        self.assertEqual(parse_scales(None), PAPER_CORE_SCALES)
        self.assertEqual(
            parse_scales(["1,1.25", "13.3"]),
            (1.0, 1.25, 13.3),
        )
        for invalid in (["0"], ["-1"], ["nan"], []):
            with self.assertRaises(ValueError):
                parse_scales(invalid)

    def test_each_scale_builds_an_independent_serving_process(self):
        run = build_run(
            "python-test",
            "configs/cluster/hbf.json",
            "workloads/hbf.jsonl",
            "/tmp/results",
            "paper-core",
            6.5,
            num_reqs=300,
            serving_args=("--log-level", "INFO"),
        )
        self.assertEqual(run.run_id, "paper-core-k6p5")
        self.assertEqual(run.output_dir, Path("/tmp/results/k6p5"))
        self.assertEqual(run.command[:3], ("python-test", "-m", "serving"))
        self.assertIn("--hbf-latency-scale", run.command)
        self.assertIn("6.5", run.command)
        self.assertIn("300", run.command)

    def test_cluster_requires_scale_based_hbf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cluster.json"
            cluster = {
                "nodes": [
                    {
                        "instances": [
                            {
                                "hbf_mem": {
                                    "performance": {"source": "scale"}
                                }
                            }
                        ]
                    }
                ]
            }
            path.write_text(json.dumps(cluster), encoding="utf-8")
            validate_scale_cluster(path)

            cluster["nodes"][0]["instances"][0]["hbf_mem"][
                "performance"
            ]["source"] = "profile"
            path.write_text(json.dumps(cluster), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source='scale'"):
                validate_scale_cluster(path)


if __name__ == "__main__":
    unittest.main()
