import csv
import json
import math
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
B200_ROOT = ROOT / "profiler" / "perf" / "B200"


EXPECTED = {
    "meta-llama/Llama-3.1-405B-Instruct": {
        "tp": (1, 2, 4, 8),
        "categories": ("dense.csv", "per_sequence.csv", "attention.csv"),
    },
    "Qwen/Qwen3-235B-A22B": {
        "tp": (1, 2, 4),
        "categories": (
            "dense.csv",
            "per_sequence.csv",
            "attention.csv",
            "moe.csv",
        ),
    },
}


class B200ProfileBundleTest(unittest.TestCase):
    def test_provenance_is_pinned(self):
        provenance = json.loads(
            (B200_ROOT / "PROVENANCE.json").read_text(encoding="utf-8")
        )
        self.assertEqual(provenance["schema_version"], 1)
        self.assertEqual(provenance["hardware"], "B200")
        self.assertEqual(
            len(provenance["source_archive_sha256"]),
            64,
        )
        self.assertEqual(
            provenance["evidence_level"],
            "b200-operator-profile-backed",
        )

    def test_model_tp_and_category_coverage(self):
        for model, contract in EXPECTED.items():
            with self.subTest(model=model):
                variant = B200_ROOT / model / "bf16"
                meta = yaml.safe_load(
                    (variant / "meta.yaml").read_text(encoding="utf-8")
                )
                self.assertEqual(meta["hardware"], "B200")
                self.assertEqual(meta["model"], model)
                self.assertEqual(tuple(meta["tp_degrees"]), contract["tp"])

                for tp in contract["tp"]:
                    tp_root = variant / f"tp{tp}"
                    for category in contract["categories"]:
                        self.assertTrue(
                            (tp_root / category).is_file(),
                            f"Missing {model}/bf16/tp{tp}/{category}",
                        )

    def test_required_latency_columns_are_finite_and_positive(self):
        for model, contract in EXPECTED.items():
            variant = B200_ROOT / model / "bf16"
            for tp in contract["tp"]:
                for category in contract["categories"]:
                    path = variant / f"tp{tp}" / category
                    with path.open(
                        "r", encoding="utf-8", newline=""
                    ) as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertTrue(rows, f"Empty profile table: {path}")
                    for row in rows:
                        latency = float(row["time_us"])
                        self.assertTrue(math.isfinite(latency))
                        self.assertGreater(latency, 0)


if __name__ == "__main__":
    unittest.main()
