import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "hbf_parallel_matrix.py"
SPEC = importlib.util.spec_from_file_location(
    "hbf_parallel_matrix", MODULE_PATH
)
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


class HbfParallelMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = matrix.load_manifest(
            REPO_ROOT / matrix.DEFAULT_MANIFEST
        )

    def test_declares_locked_topology_counts(self):
        models = {
            model["id"]: model for model in self.manifest["models"]
        }
        self.assertEqual(len(models["llama405b"]["topologies"]), 21)
        self.assertEqual(len(models["qwen235b"]["topologies"]), 26)

    def test_stage1_contains_219_hbf_k1_and_matching_hbm_runs(self):
        specs = matrix.expand_run_specs(self.manifest, "stage1")
        self.assertEqual(len(specs), 438)
        hbf = [spec for spec in specs if spec.memory_tier == "hbf"]
        hbm = [spec for spec in specs if spec.memory_tier == "hbm"]
        self.assertEqual(len(hbf), 219)
        self.assertEqual(len(hbm), 219)
        self.assertTrue(all(spec.hbf_scale == 1.0 for spec in specs))

    def test_anchor_phase_adds_exactly_588_non_identity_runs(self):
        specs = matrix.expand_run_specs(self.manifest, "anchors")
        self.assertEqual(len(specs), 588)
        self.assertTrue(all(spec.memory_tier == "hbf" for spec in specs))
        self.assertTrue(all(spec.hbf_scale != 1.0 for spec in specs))

    def test_qwen_stage1_keeps_balanced_and_custom_as_primary(self):
        specs = matrix.expand_run_specs(self.manifest, "stage1")
        qwen = [spec for spec in specs if spec.model_id == "qwen235b"]
        self.assertEqual(
            {spec.routing_policy for spec in qwen},
            {"BALANCED", "CUSTOM"},
        )
        counts = {
            policy: sum(
                spec.routing_policy == policy for spec in qwen
            )
            for policy in ("BALANCED", "CUSTOM")
        }
        self.assertEqual(counts, {"BALANCED": 156, "CUSTOM": 156})

    def test_hbf_changes_weight_placement_only(self):
        spec = next(
            spec
            for spec in matrix.expand_run_specs(
                self.manifest, "stage1"
            )
            if spec.memory_tier == "hbf"
        )
        cluster = matrix.build_cluster_config(self.manifest, spec)
        instance = cluster["nodes"][0]["instances"][0]
        placement = instance["placement"]["default"]
        self.assertEqual(placement["weights"], "hbf")
        self.assertEqual(placement["kv_loc"], "npu")
        self.assertEqual(placement["kv_evict_loc"], "cpu")
        self.assertEqual(instance["hbf_mem"]["num_stacks"], 8)
        self.assertEqual(instance["hbf_mem"]["stack_capacity_gb"], 512)

    def test_cross_node_dp_group_uses_global_ep_and_two_network_dims(self):
        spec = next(
            spec
            for spec in matrix.expand_run_specs(
                self.manifest, "stage1"
            )
            if spec.model_id == "qwen235b"
            and spec.topology_id == "dpg4-tp4-ep16"
            and spec.routing_policy == "CUSTOM"
            and spec.memory_tier == "hbf"
        )
        cluster = matrix.build_cluster_config(self.manifest, spec)
        instances = [
            instance
            for node in cluster["nodes"]
            for instance in node["instances"]
        ]
        self.assertEqual(spec.num_devices, 16)
        self.assertEqual(cluster["num_nodes"], 2)
        self.assertEqual(len(instances), 4)
        self.assertTrue(all(instance["dp_group"] == "experts"
                            for instance in instances))
        self.assertTrue(all(instance["ep_size"] == 16
                            for instance in instances))
        self.assertEqual(cluster["link_bw"], [1800, 50])
        self.assertEqual(cluster["link_latency"], [500, 10000])

    def test_custom_command_is_explicit_and_disables_block_copy(self):
        spec = next(
            spec
            for spec in matrix.expand_run_specs(
                self.manifest, "stage1"
            )
            if spec.routing_policy == "CUSTOM"
        )
        with tempfile.TemporaryDirectory() as tmp:
            command = matrix.build_command(
                REPO_ROOT,
                self.manifest,
                spec,
                Path(tmp),
                "python",
            )
        self.assertIn("--expert-routing-profile", command)
        self.assertIn("--no-enable-block-copy", command)
        self.assertIn("--log-level", command)
        self.assertEqual(
            command[command.index("--expert-routing-policy") + 1],
            "CUSTOM",
        )

    def test_routing_sensitivity_contract(self):
        specs = matrix.expand_run_specs(self.manifest, "routing")
        custom = [spec for spec in specs
                  if spec.routing_policy == "CUSTOM"]
        rand = [spec for spec in specs if spec.routing_policy == "RAND"]
        self.assertEqual(len(custom), 45)
        self.assertEqual(len(rand), 6)
        self.assertEqual(
            {spec.routing_seed for spec in custom},
            {17, 42, 101, 2027, 4096},
        )
        self.assertEqual(
            {spec.hbf_scale for spec in specs}, {1.0, 4.0, 10.0}
        )

    def test_winner_selection_expands_only_remaining_scales(self):
        selection = {
            "schema_version": 1,
            "selections": [
                {
                    "model_id": "qwen235b",
                    "workload_id": "p2048-g512",
                    "routing_policy": "CUSTOM",
                    "device_budget": 8,
                    "throughput_topology_id": "tp4-ep4-pp2",
                    "latency_topology_id": "dpg2-tp4-ep8",
                }
            ],
        }
        specs = matrix.expand_run_specs(
            self.manifest, "winners", selection
        )
        self.assertEqual(len(specs), 14)
        self.assertEqual(
            {spec.topology_id for spec in specs},
            {"tp4-ep4-pp2", "dpg2-tp4-ep8"},
        )
        self.assertTrue(all(spec.hbf_scale != 1.0 for spec in specs))


if __name__ == "__main__":
    unittest.main()
