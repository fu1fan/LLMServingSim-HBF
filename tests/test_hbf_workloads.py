import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class HBFSaturatedWorkloadTest(unittest.TestCase):
    def test_every_scenario_has_300_simultaneous_requests(self):
        scenarios = {
            "hbf-saturated-1024-128-300.jsonl": (1024, 128),
            "hbf-saturated-2048-512-300.jsonl": (2048, 512),
            "hbf-saturated-2048-2048-300.jsonl": (2048, 2048),
        }
        for filename, expected_shape in scenarios.items():
            path = REPO_ROOT / "workloads" / filename
            with open(path, encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual(len(rows), 300)
            self.assertEqual(
                {
                    (row["input_toks"], row["output_toks"])
                    for row in rows
                },
                {expected_shape},
            )
            self.assertEqual(
                {row["arrival_time_ns"] for row in rows},
                {0},
            )


if __name__ == "__main__":
    unittest.main()
