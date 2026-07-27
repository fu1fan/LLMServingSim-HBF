import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "validate_hbf_parallel_results.py"
SPEC = importlib.util.spec_from_file_location(
    "validate_hbf_parallel_results", MODULE_PATH
)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def row(topology, per_device, total, ttft, tpot):
    return {
        "status": "completed",
        "model_id": "qwen235b",
        "topology_id": topology,
        "workload_id": "p2048-g512",
        "routing_policy": "CUSTOM",
        "routing_seed": 42,
        "memory_tier": "hbf",
        "hbf_scale": 1.0,
        "network_scenario": "central",
        "num_devices": 8,
        "generation_throughput_tok_s_per_device": per_device,
        "generation_throughput_tok_s": per_device * 8,
        "total_throughput_tok_s": total,
        "ttft_ms_p99": ttft,
        "tpot_ms_p99": tpot,
    }


class ValidateHbfParallelResultsTests(unittest.TestCase):
    def test_plan_coverage_requires_all_1176_run_ids(self):
        run_ids = [
            f"run-{index}"
            for index in range(validator.EXPECTED_STATIC_RUN_COUNT)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plan.json").write_text(
                json.dumps(
                    {
                        "phase": "all",
                        "runs": [
                            {"run_id": run_id} for run_id in run_ids
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = validator.audit_plan_coverage(
                root,
                [{"run_id": run_id} for run_id in run_ids],
            )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["manifest_run_count"], 1176)

    def test_plan_coverage_reports_missing_run_id(self):
        run_ids = [
            f"run-{index}"
            for index in range(validator.EXPECTED_STATIC_RUN_COUNT)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plan.json").write_text(
                json.dumps(
                    {
                        "phase": "all",
                        "runs": [
                            {"run_id": run_id} for run_id in run_ids
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = validator.audit_plan_coverage(
                root,
                [{"run_id": run_id} for run_id in run_ids[:-1]],
            )
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["missing_run_ids"], [run_ids[-1]])

    def test_routing_log_statistics_are_recomputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "simulator.log"
            path.write_text(
                "route local=[10, 4, 2, 0] activated=[4, 3, 2, 0]\n"
                "route local=[8, 8, 4, 4] activated=[4, 4, 3, 2]\n",
                encoding="utf-8",
            )
            metrics = validator.parse_routing_log(path)
        self.assertEqual(metrics["routing_layer_samples"], 2)
        self.assertEqual(metrics["ep_rank_tokens_max"], 10)
        self.assertAlmostEqual(metrics["ep_rank_tokens_mean"], 5.0)
        self.assertGreater(metrics["ep_rank_load_cv_mean"], 0)
        self.assertGreater(metrics["ep_rank_load_gini_mean"], 0)
        self.assertAlmostEqual(
            metrics["activated_experts_per_rank_mean"], 2.75
        )

    def test_runtime_capacity_uses_instance_device_multiplicity(self):
        runtime = {
            "num_devices": 6,
            "hbf_energy_unmodeled": True,
            "instances": [
                {
                    "num_devices": 2,
                    "hbm_capacity_bytes": 100,
                    "hbm_weight_used_bytes": 10,
                    "hbm_kv_capacity_bytes": 90,
                    "hbf_capacity_bytes": 1000,
                    "hbf_weight_used_bytes": 80,
                },
                {
                    "num_devices": 4,
                    "hbm_capacity_bytes": 100,
                    "hbm_weight_used_bytes": 20,
                    "hbm_kv_capacity_bytes": 80,
                    "hbf_capacity_bytes": 1000,
                    "hbf_weight_used_bytes": 70,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime.json"
            path.write_text(json.dumps(runtime), encoding="utf-8")
            aggregate = validator.aggregate_runtime(path)
        self.assertEqual(aggregate["hbm_capacity_bytes"], 600)
        self.assertEqual(aggregate["hbm_weight_used_bytes"], 100)
        self.assertEqual(aggregate["hbf_weight_used_bytes"], 440)

    def test_hbm_hbf_identity_detects_changed_metric(self):
        base = {
            "status": "completed",
            "model_id": "qwen235b",
            "topology_id": "tp4-ep4",
            "workload_id": "p2048-g512",
            "routing_seed": 42,
            "network_scenario": "central",
            "routing_policy": "CUSTOM",
            "hbf_scale": 1.0,
        }
        hbm = {**base, "memory_tier": "hbm"}
        hbf = {**base, "memory_tier": "hbf"}
        for metric in validator.PAIR_METRICS:
            hbm[metric] = 1.0
            hbf[metric] = 1.0
        self.assertEqual(
            validator.compare_hbm_hbf_identity([hbm, hbf])[0]["status"],
            "pass",
        )
        hbf["ttft_ms_p99"] = 1.1
        comparison = validator.compare_hbm_hbf_identity([hbm, hbf])[0]
        self.assertEqual(comparison["status"], "fail")
        self.assertEqual(comparison["mismatched_metrics"], "ttft_ms_p99")

    def test_winner_rule_selects_throughput_and_low_latency_modes(self):
        rows = [
            row("fast-per-device", 12, 90, 20, 4),
            row("peak-total", 11, 100, 30, 3),
            row("low-latency-near-peak", 10, 96, 10, 5),
            row("too-slow", 8, 94, 1, 1),
        ]
        result = validator.select_winners(
            rows,
            {
                "device_budgets": [8],
                "throughput_floor_ratio": 0.95,
            },
        )
        self.assertEqual(len(result["selections"]), 1)
        selected = result["selections"][0]
        self.assertEqual(
            selected["throughput_topology_id"], "fast-per-device"
        )
        self.assertEqual(
            selected["latency_topology_id"],
            "low-latency-near-peak",
        )

    def test_capacity_enabled_hbf_result_is_labeled(self):
        common = {
            "model_id": "llama405b",
            "topology_id": "tp1",
            "workload_id": "p2048-g512",
            "routing_seed": 42,
            "network_scenario": "central",
            "routing_policy": "BALANCED",
            "hbf_scale": 1.0,
        }
        hbm = {
            **common,
            "status": "failed",
            "memory_tier": "hbm",
            "failure_class": "hbm-capacity-precheck-failed",
            "evidence_labels": "balanced-ideal-routing",
        }
        hbf = {
            **common,
            "status": "completed",
            "memory_tier": "hbf",
            "failure_class": "",
            "evidence_labels": "hbf-scale-sensitivity",
        }
        validator.annotate_capacity_enabled([hbm, hbf])
        self.assertIn(
            "capacity-enabled-no-hbm-pair",
            hbf["evidence_labels"].split("+"),
        )


if __name__ == "__main__":
    unittest.main()
