import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch


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

    def test_pure_pipeline_parallelism_uses_one_network_dimension(self):
        spec = next(
            spec
            for spec in matrix.expand_run_specs(
                self.manifest, "stage1"
            )
            if spec.model_id == "llama405b"
            and spec.topology_id == "pp8"
            and spec.memory_tier == "hbf"
        )
        cluster = matrix.build_cluster_config(self.manifest, spec)
        self.assertEqual(cluster["link_bw"], 1800)
        self.assertEqual(cluster["link_latency"], 500)

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
        self.assertIn("--cleanup-inputs", command)
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

    def test_parallel_runner_executes_each_spec_once(self):
        base = matrix.expand_run_specs(self.manifest, "stage1")[0]
        specs = [
            replace(base, workload_id=f"workload-{index}")
            for index in range(4)
        ]
        calls = []

        def fake_run_one(_repo_root, _manifest, spec, *_args):
            calls.append(spec.run_id)
            return spec.run_id, "completed", 0

        run_args = [
            (spec, Path("/tmp") / spec.run_id, ["python"], "sha",
             "profile-sha", False)
            for spec in specs
        ]
        with patch.object(matrix, "_run_one", side_effect=fake_run_one):
            failures = matrix._run_specs(
                REPO_ROOT,
                self.manifest,
                jobs=3,
                keep_going=True,
                run_args=run_args,
            )
        self.assertEqual(failures, [])
        self.assertCountEqual(calls, [spec.run_id for spec in specs])

    def test_parallel_runner_stops_submitting_after_failure(self):
        base = matrix.expand_run_specs(self.manifest, "stage1")[0]
        specs = [
            replace(base, workload_id=f"workload-{index}")
            for index in range(5)
        ]
        calls = []

        def fake_run_one(_repo_root, _manifest, spec, *_args):
            calls.append(spec.run_id)
            status = "failed" if spec == specs[0] else "completed"
            return spec.run_id, status, int(status == "failed")

        run_args = [
            (spec, Path("/tmp") / spec.run_id, ["python"], "sha",
             "profile-sha", False)
            for spec in specs
        ]
        with patch.object(matrix, "_run_one", side_effect=fake_run_one):
            failures = matrix._run_specs(
                REPO_ROOT,
                self.manifest,
                jobs=2,
                keep_going=False,
                run_args=run_args,
            )
        self.assertEqual(failures, [specs[0].run_id])
        self.assertLessEqual(len(calls), 2)

    def test_recorded_failure_is_skipped_unless_rerun_requested(self):
        spec = matrix.expand_run_specs(self.manifest, "stage1")[0]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "manifest.json").write_text(
                json.dumps({"status": "failed"}),
                encoding="utf-8",
            )
            with patch.object(matrix.subprocess, "run") as run:
                result = matrix._run_one(
                    REPO_ROOT,
                    self.manifest,
                    spec,
                    run_dir,
                    ["python"],
                    "sha",
                    "profile-sha",
                    rerun_failed=False,
                )
        self.assertEqual(result, (spec.run_id, "skipped", 0))
        run.assert_not_called()

    def test_completed_run_is_skipped_when_rerunning_failures(self):
        spec = matrix.expand_run_specs(self.manifest, "stage1")[0]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "manifest.json").write_text(
                json.dumps({"status": "completed"}),
                encoding="utf-8",
            )
            with patch.object(matrix.subprocess, "run") as run:
                result = matrix._run_one(
                    REPO_ROOT,
                    self.manifest,
                    spec,
                    run_dir,
                    ["python"],
                    "sha",
                    "profile-sha",
                    rerun_failed=True,
                )
        self.assertEqual(result, (spec.run_id, "skipped", 0))
        run.assert_not_called()

    def test_recorded_failure_runs_when_rerun_is_requested(self):
        spec = matrix.expand_run_specs(self.manifest, "stage1")[0]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "manifest.json").write_text(
                json.dumps({"status": "failed"}),
                encoding="utf-8",
            )
            with patch.object(
                matrix,
                "_run_monitored_command",
                return_value=(1, None, None),
            ) as run:
                result = matrix._run_one(
                    REPO_ROOT,
                    self.manifest,
                    spec,
                    run_dir,
                    ["python"],
                    "sha",
                    "profile-sha",
                    rerun_failed=True,
                )
        self.assertEqual(result, (spec.run_id, "failed", 1))
        run.assert_called_once()

    def test_unreadable_manifest_is_rerun(self):
        spec = matrix.expand_run_specs(self.manifest, "stage1")[0]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "manifest.json").write_text(
                "", encoding="utf-8"
            )
            with patch.object(
                matrix,
                "_run_monitored_command",
                return_value=(1, None, None),
            ) as run:
                result = matrix._run_one(
                    REPO_ROOT,
                    self.manifest,
                    spec,
                    run_dir,
                    ["python"],
                    "sha",
                    "profile-sha",
                    rerun_failed=False,
                )
        self.assertEqual(result, (spec.run_id, "failed", 1))
        run.assert_called_once()

    def test_recorded_timeout_is_not_retried_by_default(self):
        spec = matrix.expand_run_specs(self.manifest, "stage1")[0]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "manifest.json").write_text(
                json.dumps({
                    "status": "failed",
                    "failure_kind": "stall_timeout",
                }),
                encoding="utf-8",
            )
            with patch.object(
                matrix, "_run_monitored_command"
            ) as run:
                result = matrix._run_one(
                    REPO_ROOT,
                    self.manifest,
                    spec,
                    run_dir,
                    ["python"],
                    "sha",
                    "profile-sha",
                    rerun_failed=True,
                )
        self.assertEqual(result, (spec.run_id, "skipped", 0))
        run.assert_not_called()

    def test_legacy_capacity_failure_is_classified_and_not_retried(self):
        spec = matrix.expand_run_specs(self.manifest, "stage1")[0]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest_path = run_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps({"status": "failed"}),
                encoding="utf-8",
            )
            (run_dir / "simulator.log").write_text(
                "RuntimeError: HBM weight capacity exceeded: "
                "required=811706777600 available=193273528320\n",
                encoding="utf-8",
            )
            with patch.object(
                matrix, "_run_monitored_command"
            ) as run:
                result = matrix._run_one(
                    REPO_ROOT,
                    self.manifest,
                    spec,
                    run_dir,
                    ["python"],
                    "sha",
                    "profile-sha",
                    rerun_failed=True,
                )
            recorded = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        self.assertEqual(result, (spec.run_id, "skipped", 0))
        self.assertEqual(
            recorded["failure_kind"], "capacity_precheck"
        )
        run.assert_not_called()

    def test_new_capacity_failure_records_non_retryable_kind(self):
        spec = matrix.expand_run_specs(self.manifest, "stage1")[0]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)

            def fail_with_capacity(
                _command,
                _repo_root,
                log,
                _log_path,
                *_args,
            ):
                log.write(
                    "RuntimeError: HBF weight capacity exceeded: "
                    "required=20 available=10\n"
                )
                log.flush()
                return 1, None, None

            with patch.object(
                matrix,
                "_run_monitored_command",
                side_effect=fail_with_capacity,
            ):
                result = matrix._run_one(
                    REPO_ROOT,
                    self.manifest,
                    spec,
                    run_dir,
                    ["python"],
                    "sha",
                    "profile-sha",
                    rerun_failed=False,
                )
            recorded = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
        self.assertEqual(result, (spec.run_id, "failed", 1))
        self.assertEqual(
            recorded["failure_kind"], "capacity_precheck"
        )

    def test_failed_run_cleans_generated_simulator_inputs(self):
        spec = matrix.expand_run_specs(self.manifest, "stage1")[0]
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            inputs_root = (
                repo_root / "astra-sim" / "inputs" / "runs"
                / spec.run_id
            )
            inputs_root.mkdir(parents=True)
            (inputs_root / "trace.et").write_text(
                "generated", encoding="utf-8"
            )
            workload_path = repo_root / spec.workload_path
            workload_path.parent.mkdir(parents=True)
            workload_path.write_text("{}\n", encoding="utf-8")
            run_dir = Path(tmp) / "results" / spec.run_id
            with patch.object(
                matrix,
                "_run_monitored_command",
                return_value=(1, None, None),
            ):
                result = matrix._run_one(
                    repo_root,
                    self.manifest,
                    spec,
                    run_dir,
                    ["python"],
                    "sha",
                    "profile-sha",
                    rerun_failed=False,
                )
            self.assertFalse(inputs_root.exists())
        self.assertEqual(result, (spec.run_id, "failed", 1))

    def test_timeout_failure_is_recorded_with_watchdog_provenance(self):
        spec = matrix.expand_run_specs(self.manifest, "stage1")[0]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with patch.object(
                matrix,
                "_run_monitored_command",
                return_value=(
                    124,
                    "stall_timeout",
                    "zero throughput with unchanged request state",
                ),
            ):
                result = matrix._run_one(
                    REPO_ROOT,
                    self.manifest,
                    spec,
                    run_dir,
                    ["python"],
                    "sha",
                    "profile-sha",
                    rerun_failed=False,
                    run_timeout_seconds=7200,
                    stall_timeout_seconds=600,
                    stall_sim_seconds=3600,
                )
            recorded = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
        self.assertEqual(result, (spec.run_id, "failed", 124))
        self.assertEqual(recorded["failure_kind"], "stall_timeout")
        self.assertEqual(
            recorded["watchdog"]["stall_sim_seconds"], 3600
        )

    def test_progress_snapshot_uses_latest_status_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "simulator.log"
            log_path.write_text(
                "[10.0s] Avg prompt throughput: 1.0 tokens/s, "
                "Avg generation throughput: 0.0 tokens/s\n"
                "Running Instance[0]: 2 reqs, Waiting: 4 reqs\n"
                "[20.0s] Avg prompt throughput: 0.0 tokens/s, "
                "Avg generation throughput: 0.0 tokens/s\n"
                "Running Instance[0]: 2 reqs, Waiting: 4 reqs\n"
                "Running Instance[1]: 1 reqs, Waiting: 5 reqs\n",
                encoding="utf-8",
            )
            snapshot = matrix._read_progress_snapshot(log_path)
        self.assertEqual(snapshot["sim_seconds"], 20.0)
        self.assertEqual(snapshot["prompt_throughput"], 0.0)
        self.assertEqual(
            snapshot["instances"],
            ((0, 2, 4), (1, 1, 5)),
        )

    def test_watchdog_cli_defaults_are_bounded(self):
        args = matrix.build_parser().parse_args([
            "--output-dir", "/tmp/results",
        ])
        self.assertEqual(
            args.run_timeout_seconds,
            matrix.DEFAULT_RUN_TIMEOUT_SECONDS,
        )
        self.assertEqual(
            args.stall_timeout_seconds,
            matrix.DEFAULT_STALL_TIMEOUT_SECONDS,
        )
        self.assertEqual(
            args.stall_sim_seconds,
            matrix.DEFAULT_STALL_SIM_SECONDS,
        )

    def test_monitored_command_stops_zero_throughput_stall(self):
        process = MagicMock()
        process.pid = 123
        process.poll.side_effect = [None, None]
        snapshots = [
            {
                "sim_seconds": 10.0,
                "prompt_throughput": 0.0,
                "generation_throughput": 0.0,
                "instances": ((0, 2, 4),),
            },
            {
                "sim_seconds": 4010.0,
                "prompt_throughput": 0.0,
                "generation_throughput": 0.0,
                "instances": ((0, 2, 4),),
            },
        ]
        with (
            patch.object(matrix.subprocess, "Popen", return_value=process),
            patch.object(
                matrix.time,
                "monotonic",
                side_effect=[0.0, 0.0, 601.0],
            ),
            patch.object(matrix.time, "sleep"),
            patch.object(
                matrix,
                "_read_progress_snapshot",
                side_effect=snapshots,
            ),
            patch.object(matrix, "_terminate_process_group") as terminate,
        ):
            result = matrix._run_monitored_command(
                ["python"],
                REPO_ROOT,
                MagicMock(),
                Path("/tmp/simulator.log"),
                run_timeout_seconds=7200,
                stall_timeout_seconds=600,
                stall_sim_seconds=3600,
            )
        self.assertEqual(result[0:2], (124, "stall_timeout"))
        terminate.assert_called_once_with(process)

    def test_collect_process_tree_is_deepest_first(self):
        with tempfile.TemporaryDirectory() as directory:
            proc_root = Path(directory)
            for pid, start_time, children in (
                (100, 1000, "101 102"),
                (101, 1001, "103"),
                (102, 1002, ""),
                (103, 1003, ""),
            ):
                process_dir = proc_root / str(pid)
                task_dir = process_dir / "task" / str(pid)
                task_dir.mkdir(parents=True)
                fields = ["S"] + ["0"] * 18 + [str(start_time)]
                (process_dir / "stat").write_text(
                    f"{pid} (worker {pid}) {' '.join(fields)}\n",
                    encoding="utf-8",
                )
                (task_dir / "children").write_text(
                    children,
                    encoding="utf-8",
                )

            processes = matrix._collect_process_tree(100, proc_root)

        self.assertEqual(
            processes,
            [(103, 1003), (101, 1001), (102, 1002), (100, 1000)],
        )

    def test_signal_processes_skips_reused_pid(self):
        processes = [(101, 1001), (102, 1002)]
        with (
            patch.object(
                matrix,
                "_same_process",
                side_effect=[False, True],
            ),
            patch.object(matrix.os, "kill") as kill,
        ):
            matrix._signal_processes(processes, matrix.signal.SIGTERM)

        kill.assert_called_once_with(102, matrix.signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
