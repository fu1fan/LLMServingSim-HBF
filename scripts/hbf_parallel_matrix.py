#!/usr/bin/env python3
"""Generate and run the native LLMServingSim HBF parallel-mode matrix."""

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_MANIFEST = "configs/experiments/hbf_parallel_modes_v1.json"
STRESS_SCALES = (1.0, 4.0, 10.0)
DEFAULT_RUN_TIMEOUT_SECONDS = 7200
DEFAULT_STALL_TIMEOUT_SECONDS = 600
DEFAULT_STALL_SIM_SECONDS = 3600
TIMEOUT_FAILURE_KINDS = frozenset({"stall_timeout", "wall_timeout"})
NON_RETRYABLE_FAILURE_KINDS = frozenset({
    "capacity_precheck",
    *TIMEOUT_FAILURE_KINDS,
})
STATUS_RE = re.compile(
    r"^\[(?P<sim_seconds>[0-9.]+)s\] Avg prompt throughput: "
    r"(?P<prompt>[0-9.]+) tokens/s, Avg generation throughput: "
    r"(?P<generation>[0-9.]+) tokens/s",
    re.MULTILINE,
)
INSTANCE_RE = re.compile(
    r"Running Instance\[(?P<instance>[0-9]+)\]: "
    r"(?P<running>[0-9]+) reqs, Waiting: (?P<waiting>[0-9]+) reqs"
)


@dataclass(frozen=True)
class RunSpec:
    phase: str
    model_id: str
    model_name: str
    model_kind: str
    topology_id: str
    topology: dict
    workload_id: str
    workload_path: str
    routing_policy: str
    routing_seed: int
    memory_tier: str
    hbf_scale: float
    network_scenario: str
    evidence_level: str

    @property
    def num_devices(self):
        return topology_num_devices(self.topology)

    @property
    def run_id(self):
        scale = format(self.hbf_scale, ".12g").replace(".", "p")
        return "__".join(
            (
                self.phase,
                self.model_id,
                self.topology_id,
                self.workload_id,
                self.routing_policy.lower(),
                f"s{self.routing_seed}",
                self.memory_tier,
                f"k{scale}",
                self.network_scenario,
            )
        )


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1:
        raise ValueError("Experiment manifest schema_version must be 1")
    models = {model["id"]: model for model in manifest.get("models", [])}
    if set(models) != {"llama405b", "qwen235b"}:
        raise ValueError("Manifest must define llama405b and qwen235b")
    if len(models["llama405b"]["topologies"]) != 21:
        raise ValueError("Llama matrix must contain exactly 21 topologies")
    if len(models["qwen235b"]["topologies"]) != 26:
        raise ValueError("Qwen matrix must contain exactly 26 topologies")
    return manifest


def topology_num_devices(topology):
    replicas = topology.get(
        "dp_group_size",
        topology.get("replicas", 1),
    )
    return int(replicas) * int(topology["tp"]) * int(topology["pp"])


def _model_map(manifest):
    return {model["id"]: model for model in manifest["models"]}


def _workload_map(manifest):
    return {workload["id"]: workload for workload in manifest["workloads"]}


def _topology_map(model):
    return {topology["id"]: topology for topology in model["topologies"]}


def _routing_evidence(manifest, policy):
    return manifest["routing"]["evidence"][policy]


def _make_spec(
    manifest,
    phase,
    model,
    topology,
    workload,
    routing_policy,
    memory_tier,
    hbf_scale,
    network_scenario="central",
    routing_seed=None,
):
    if routing_seed is None:
        routing_seed = manifest["routing"]["main_seed"]
    evidence = [_routing_evidence(manifest, routing_policy)]
    if memory_tier == "hbf":
        evidence.append("hbf-scale-sensitivity")
    evidence.append("network-estimated")
    return RunSpec(
        phase=phase,
        model_id=model["id"],
        model_name=model["model_name"],
        model_kind=model["kind"],
        topology_id=topology["id"],
        topology=copy.deepcopy(topology),
        workload_id=workload["id"],
        workload_path=workload["path"],
        routing_policy=routing_policy,
        routing_seed=int(routing_seed),
        memory_tier=memory_tier,
        hbf_scale=float(hbf_scale),
        network_scenario=network_scenario,
        evidence_level="+".join(evidence),
    )


def _expand_stage1(manifest):
    specs = []
    for model in manifest["models"]:
        for topology in model["topologies"]:
            for workload in manifest["workloads"]:
                for policy in model["routing_policies"]:
                    for tier in ("hbm", "hbf"):
                        specs.append(
                            _make_spec(
                                manifest,
                                "stage1",
                                model,
                                topology,
                                workload,
                                policy,
                                tier,
                                1.0,
                            )
                        )
    return specs


def _expand_anchors(manifest):
    specs = []
    for model in manifest["models"]:
        topologies = _topology_map(model)
        for topology_id in model["anchors"]:
            topology = topologies[topology_id]
            for workload in manifest["workloads"]:
                for policy in model["routing_policies"]:
                    for scale in manifest["hbf_scales"]:
                        if float(scale) == 1.0:
                            continue
                        specs.append(
                            _make_spec(
                                manifest,
                                "anchors",
                                model,
                                topology,
                                workload,
                                policy,
                                "hbf",
                                scale,
                            )
                        )
    return specs


def _expand_routing(manifest):
    model = _model_map(manifest)["qwen235b"]
    topologies = _topology_map(model)
    workload = _workload_map(manifest)["p2048-g512"]
    specs = []
    for topology_id in model["routing_seed_representatives"]:
        for seed in manifest["routing"]["mapping_seeds"]:
            for scale in STRESS_SCALES:
                specs.append(
                    _make_spec(
                        manifest,
                        "routing",
                        model,
                        topologies[topology_id],
                        workload,
                        "CUSTOM",
                        "hbf",
                        scale,
                        routing_seed=seed,
                    )
                )
    for topology_id in model["rand_representatives"]:
        for scale in STRESS_SCALES:
            specs.append(
                _make_spec(
                    manifest,
                    "routing",
                    model,
                    topologies[topology_id],
                    workload,
                    "RAND",
                    "hbf",
                    scale,
                )
            )
    return specs


def _expand_network(manifest):
    workload = _workload_map(manifest)["p2048-g512"]
    specs = []
    for model in manifest["models"]:
        topologies = _topology_map(model)
        for topology_id in model["network_representatives"]:
            for policy in model["routing_policies"]:
                for scenario in manifest["network_scenarios"]:
                    for scale in STRESS_SCALES:
                        specs.append(
                            _make_spec(
                                manifest,
                                "network",
                                model,
                                topologies[topology_id],
                                workload,
                                policy,
                                "hbf",
                                scale,
                                network_scenario=scenario,
                            )
                        )
    return specs


def _expand_winners(manifest, selection):
    if not selection or selection.get("schema_version") != 1:
        raise ValueError("The winners phase requires a schema v1 selection")
    models = _model_map(manifest)
    workloads = _workload_map(manifest)
    specs = []
    for row in selection.get("selections", []):
        model = models[row["model_id"]]
        topology_map = _topology_map(model)
        topology_ids = {
            row["throughput_topology_id"],
            row["latency_topology_id"],
        }
        for topology_id in topology_ids:
            for scale in manifest["hbf_scales"]:
                if float(scale) == 1.0:
                    continue
                specs.append(
                    _make_spec(
                        manifest,
                        "winners",
                        model,
                        topology_map[topology_id],
                        workloads[row["workload_id"]],
                        row["routing_policy"],
                        "hbf",
                        scale,
                    )
                )
    return specs


def expand_run_specs(manifest, phase, selection=None):
    builders = {
        "stage1": _expand_stage1,
        "anchors": _expand_anchors,
        "routing": _expand_routing,
        "network": _expand_network,
    }
    if phase == "winners":
        specs = _expand_winners(manifest, selection)
    elif phase == "all":
        specs = []
        for name in builders:
            specs.extend(builders[name](manifest))
    else:
        specs = builders[phase](manifest)
    unique = {}
    for spec in specs:
        unique.setdefault(spec.run_id, spec)
    return list(unique.values())


def _instance_config(manifest, spec):
    topology = spec.topology
    instance = {
        "model_name": spec.model_name,
        "hardware": manifest["hardware"],
        "npu_mem": {
            "mem_size": manifest["memory"]["hbm_gib_per_gpu"],
            "mem_bw": manifest["memory"]["hbm_bandwidth_gb_s"],
            "mem_latency": 0,
        },
        "placement": {
            "default": {
                "weights": "hbf" if spec.memory_tier == "hbf" else "npu",
                "kv_loc": "npu",
                "kv_evict_loc": "cpu",
            }
        },
        "num_npus": int(topology["tp"]) * int(topology["pp"]),
        "tp_size": int(topology["tp"]),
        "pp_size": int(topology["pp"]),
        "pd_type": None,
    }
    if spec.model_kind == "moe":
        instance["ep_size"] = int(topology["ep"])
    if topology.get("dp_group_size") is not None:
        instance["dp_group"] = "experts"
    if spec.memory_tier == "hbf":
        instance["hbf_mem"] = {
            "schema_version": 1,
            "num_stacks": manifest["memory"]["hbf_stacks_per_gpu"],
            "stack_capacity_gb": manifest["memory"][
                "hbf_stack_capacity_gib"
            ],
            "performance": {
                "source": "scale",
                "latency_scale": spec.hbf_scale,
            },
        }
    return instance


def _pack_nodes(instances):
    nodes = []
    current = []
    used = 0
    for instance in instances:
        devices = int(instance["num_npus"])
        if devices > 8:
            raise ValueError(
                "A single model-parallel group may not exceed one 8-GPU node"
            )
        if current and used + devices > 8:
            nodes.append(current)
            current = []
            used = 0
        current.append(instance)
        used += devices
    if current:
        nodes.append(current)
    return [
        {
            "num_instances": len(node_instances),
            "cpu_mem": {
                "mem_size": 4096,
                "mem_bw": 512,
                "mem_latency": 0,
            },
            "instances": node_instances,
        }
        for node_instances in nodes
    ]


def _network_values(manifest, spec, num_nodes):
    scenario = manifest["network_scenarios"][spec.network_scenario]
    topology = spec.topology
    dp_group = topology.get("dp_group_size")
    replicas = int(topology.get("replicas", 1))
    tp = int(topology["tp"])
    pp = int(topology["pp"])
    if dp_group is not None:
        num_dims = 2
    else:
        second_dim = replicas if tp == 1 else replicas * pp
        num_dims = 2 if second_dim > 1 else 1
    if num_dims == 1:
        return (
            scenario["intra_bw_gb_s"],
            scenario["intra_latency_ns"],
        )

    second_is_cross_node_ep = dp_group is not None and num_nodes > 1
    second_bw = (
        scenario["inter_bw_gb_s"]
        if second_is_cross_node_ep
        else scenario["intra_bw_gb_s"]
    )
    second_latency = (
        scenario["inter_latency_ns"]
        if second_is_cross_node_ep
        else scenario["intra_latency_ns"]
    )
    return (
        [scenario["intra_bw_gb_s"], second_bw],
        [scenario["intra_latency_ns"], second_latency],
    )


def build_cluster_config(manifest, spec):
    topology = spec.topology
    instance_count = int(
        topology.get("dp_group_size", topology.get("replicas", 1))
    )
    instances = [
        _instance_config(manifest, spec)
        for _ in range(instance_count)
    ]
    nodes = _pack_nodes(instances)
    link_bw, link_latency = _network_values(
        manifest, spec, len(nodes)
    )
    return {
        "num_nodes": len(nodes),
        "link_bw": link_bw,
        "link_latency": link_latency,
        "nodes": nodes,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path):
    path = Path(path)
    digest = hashlib.sha256()
    for file_path in sorted(
        candidate for candidate in path.rglob("*")
        if candidate.is_file()
    ):
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(file_path)))
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def build_command(repo_root, manifest, spec, run_dir, python):
    cluster_path = (run_dir / "cluster.json").resolve()
    workload = (repo_root / spec.workload_path).resolve()
    command = [
        python,
        "-m",
        "serving",
        "--cluster-config",
        os.fspath(cluster_path),
        "--dataset",
        os.fspath(workload),
        "--output",
        os.fspath((run_dir / "requests.csv").resolve()),
        "--hbf-summary-output",
        os.fspath((run_dir / "runtime.json").resolve()),
        "--run-id",
        spec.run_id,
        "--dtype",
        manifest["dtype"],
        "--max-num-batched-tokens",
        "2048",
        "--max-num-seqs",
        "256",
        "--request-routing-policy",
        "LOAD",
        "--expert-routing-policy",
        spec.routing_policy,
        "--expert-routing-seed",
        str(spec.routing_seed),
        "--log-level",
        "INFO",
        "--no-enable-prefix-caching",
    ]
    if spec.memory_tier == "hbf":
        command.extend(
            ["--hbf-latency-scale", format(spec.hbf_scale, ".12g")]
        )
    if spec.routing_policy == "CUSTOM":
        command.extend(
            [
                "--expert-routing-profile",
                os.fspath(
                    (repo_root / manifest["routing"]["custom_profile"]).resolve()
                ),
                "--no-enable-block-copy",
            ]
        )
    elif spec.routing_policy == "RAND":
        command.append("--no-enable-block-copy")
    return command


def _git_sha(repo_root):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def _read_tail(path, max_bytes=262144):
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - max_bytes))
            return stream.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _read_progress_snapshot(log_path):
    text = _read_tail(log_path)
    matches = list(STATUS_RE.finditer(text))
    if not matches:
        return None
    status = matches[-1]
    instances = tuple(
        (
            int(match.group("instance")),
            int(match.group("running")),
            int(match.group("waiting")),
        )
        for match in INSTANCE_RE.finditer(text[status.end():])
    )
    return {
        "sim_seconds": float(status.group("sim_seconds")),
        "prompt_throughput": float(status.group("prompt")),
        "generation_throughput": float(status.group("generation")),
        "instances": instances,
    }


def _infer_failure(log_path):
    text = _read_tail(log_path)
    for line in reversed(text.splitlines()):
        if (
            "capacity exceeded:" in line
            and "required=" in line
            and "available=" in line
        ):
            return "capacity_precheck", line.strip()
    return None, None


def _read_process_identity(pid, proc_root=Path("/proc")):
    try:
        stat = (proc_root / str(pid) / "stat").read_text(
            encoding="utf-8"
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    closing_paren = stat.rfind(")")
    if closing_paren < 0:
        return None
    fields = stat[closing_paren + 2:].split()
    if len(fields) <= 19:
        return None
    return int(fields[19])


def _read_process_children(pid, proc_root=Path("/proc")):
    try:
        text = (
            proc_root / str(pid) / "task" / str(pid) / "children"
        ).read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ()
    return tuple(int(child) for child in text.split())


def _collect_process_tree(root_pid, proc_root=Path("/proc")):
    """Snapshot descendants deepest-first, retaining PID identities."""
    seen = set()
    processes = []

    def visit(pid):
        if pid in seen:
            return
        seen.add(pid)
        identity = _read_process_identity(pid, proc_root)
        if identity is None:
            return
        for child in _read_process_children(pid, proc_root):
            visit(child)
        processes.append((pid, identity))

    visit(root_pid)
    return processes


def _same_process(pid, identity):
    return _read_process_identity(pid) == identity


def _signal_processes(processes, sig):
    for pid, identity in processes:
        if not _same_process(pid, identity):
            continue
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _terminate_process_group(process):
    processes = _collect_process_tree(process.pid)
    _signal_processes(processes, signal.SIGTERM)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if (
            process.poll() is not None
            and not any(_same_process(*item) for item in processes)
        ):
            process.wait()
            return
        time.sleep(0.1)

    _signal_processes(processes, signal.SIGKILL)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _run_monitored_command(command, repo_root, log, log_path,
                           run_timeout_seconds, stall_timeout_seconds,
                           stall_sim_seconds, poll_seconds=5):
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    started = time.monotonic()
    last_progress = started
    baseline_sim_seconds = None
    last_instances = None
    failure_kind = None
    failure_detail = None

    while process.poll() is None:
        now = time.monotonic()
        if now - started >= run_timeout_seconds:
            failure_kind = "wall_timeout"
            failure_detail = (
                f"wall clock exceeded {run_timeout_seconds:g} seconds"
            )
            break

        snapshot = _read_progress_snapshot(log_path)
        if snapshot is not None:
            has_throughput = (
                snapshot["prompt_throughput"] > 0
                or snapshot["generation_throughput"] > 0
            )
            instances = snapshot["instances"]
            state_changed = (
                last_instances is not None and instances != last_instances
            )
            if (
                baseline_sim_seconds is None
                or has_throughput
                or state_changed
            ):
                last_progress = now
                baseline_sim_seconds = snapshot["sim_seconds"]
            last_instances = instances
            simulated_stall = (
                snapshot["sim_seconds"] - baseline_sim_seconds
                if baseline_sim_seconds is not None
                else 0
            )
            if (
                not has_throughput
                and now - last_progress >= stall_timeout_seconds
                and simulated_stall >= stall_sim_seconds
            ):
                failure_kind = "stall_timeout"
                failure_detail = (
                    "zero throughput with unchanged request state for "
                    f"{now - last_progress:.1f} wall seconds while simulated "
                    f"time advanced {simulated_stall:.1f} seconds"
                )
                break
        time.sleep(poll_seconds)

    if failure_kind is not None:
        _terminate_process_group(process)
        return 124, failure_kind, failure_detail
    return process.returncode, None, None


def _cleanup_simulator_inputs(repo_root, run_id):
    """Remove one matrix run's generated ASTRA-Sim inputs."""
    runs_root = (
        Path(repo_root) / "astra-sim" / "inputs" / "runs"
    ).resolve()
    inputs_root = (runs_root / run_id).resolve()
    if inputs_root.parent != runs_root:
        raise RuntimeError(
            f"Refusing to remove inputs outside {runs_root}: "
            f"{inputs_root}"
        )
    shutil.rmtree(inputs_root, ignore_errors=True)


def _run_one(repo_root, manifest, spec, run_dir, command,
             code_sha, profile_bundle_sha, rerun_failed,
             run_timeout_seconds=DEFAULT_RUN_TIMEOUT_SECONDS,
             stall_timeout_seconds=DEFAULT_STALL_TIMEOUT_SECONDS,
             stall_sim_seconds=DEFAULT_STALL_SIM_SECONDS,
    retry_timeouts=False):
    previous_manifest = run_dir / "manifest.json"
    if previous_manifest.is_file():
        try:
            previous = json.loads(
                previous_manifest.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            previous = None
        if previous is None:
            print(
                f"[rerun] {spec.run_id}: unreadable manifest",
                flush=True,
            )
        else:
            previous_status = previous.get("status")
            if previous_status == "completed":
                return spec.run_id, "skipped", 0
            if previous_status == "failed":
                failure_kind = previous.get("failure_kind")
                if failure_kind is None:
                    failure_kind, failure_detail = _infer_failure(
                        run_dir / "simulator.log"
                    )
                    if failure_kind is not None:
                        previous["failure_kind"] = failure_kind
                        previous["failure_detail"] = failure_detail
                        write_json(previous_manifest, previous)
                if (
                    failure_kind in NON_RETRYABLE_FAILURE_KINDS
                    and (
                        failure_kind not in TIMEOUT_FAILURE_KINDS
                        or not retry_timeouts
                    )
                ):
                    return spec.run_id, "skipped", 0
                if not rerun_failed:
                    return spec.run_id, "skipped", 0

    run_dir.mkdir(parents=True, exist_ok=True)
    cluster_path = run_dir / "cluster.json"
    write_json(cluster_path, build_cluster_config(manifest, spec))
    log_path = run_dir / "simulator.log"
    print(f"[run] {spec.run_id}", flush=True)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            exit_code, failure_kind, failure_detail = (
                _run_monitored_command(
                    command,
                    repo_root,
                    log,
                    log_path,
                    run_timeout_seconds,
                    stall_timeout_seconds,
                    stall_sim_seconds,
                )
            )
    finally:
        _cleanup_simulator_inputs(repo_root, spec.run_id)
    if exit_code != 0 and failure_kind is None:
        failure_kind, failure_detail = _infer_failure(log_path)
    status = "completed" if exit_code == 0 else "failed"
    run_manifest = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "run_id": spec.run_id,
        "status": status,
        "exit_code": exit_code,
        "git_sha": code_sha,
        "profile_bundle_sha256": profile_bundle_sha,
        "spec": asdict(spec),
        "num_devices": spec.num_devices,
        "cluster_config_sha256": sha256_file(cluster_path),
        "workload_sha256": sha256_file(
            repo_root / spec.workload_path
        ),
        "routing_profile_sha256": (
            sha256_file(
                repo_root / manifest["routing"]["custom_profile"]
            )
            if spec.routing_policy == "CUSTOM"
            else None
        ),
        "command": command,
        "watchdog": {
            "run_timeout_seconds": run_timeout_seconds,
            "stall_timeout_seconds": stall_timeout_seconds,
            "stall_sim_seconds": stall_sim_seconds,
        },
        "outputs": {
            "requests": os.fspath(run_dir / "requests.csv"),
            "runtime": os.fspath(run_dir / "runtime.json"),
            "log": os.fspath(log_path),
        },
    }
    if failure_kind is not None:
        run_manifest["failure_kind"] = failure_kind
        run_manifest["failure_detail"] = failure_detail
    write_json(previous_manifest, run_manifest)
    return spec.run_id, status, exit_code


def _run_specs(repo_root, manifest, jobs, keep_going, run_args):
    failures = []
    if jobs == 1:
        for args in run_args:
            run_id, status, _ = _run_one(
                repo_root, manifest, *args
            )
            print(f"[{status}] {run_id}", flush=True)
            if status == "failed":
                failures.append(run_id)
                if not keep_going:
                    break
        return failures

    pending_args = iter(run_args)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        in_flight = {}

        def submit_next():
            try:
                args = next(pending_args)
            except StopIteration:
                return False
            future = executor.submit(
                _run_one, repo_root, manifest, *args
            )
            in_flight[future] = args[0].run_id
            return True

        for _ in range(jobs):
            if not submit_next():
                break

        stop_submitting = False
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                in_flight.pop(future)
                run_id, status, _ = future.result()
                print(f"[{status}] {run_id}", flush=True)
                if status == "failed":
                    failures.append(run_id)
                    if not keep_going:
                        stop_submitting = True
            if not stop_submitting:
                for _ in done:
                    submit_next()
    return failures


def run_matrix(args):
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    manifest = load_manifest(manifest_path)
    selection = None
    if args.phase == "winners":
        if not args.selection:
            raise ValueError("--phase winners requires --selection")
        with open(args.selection, "r", encoding="utf-8") as stream:
            selection = json.load(stream)
    specs = expand_run_specs(manifest, args.phase, selection)
    if args.limit is not None:
        specs = specs[: args.limit]

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    code_sha = _git_sha(repo_root)
    profile_bundle_sha = sha256_tree(
        repo_root / manifest["profiles"]["root"]
    )
    plan_rows = []
    run_args = []

    for spec in specs:
        run_dir = output_root / spec.run_id
        command = build_command(
            repo_root, manifest, spec, run_dir, args.python
        )
        plan_rows.append(
            {
                **asdict(spec),
                "run_id": spec.run_id,
                "num_devices": spec.num_devices,
                "run_dir": os.fspath(run_dir),
                "command": command,
            }
        )
        if args.dry_run:
            continue
        run_args.append((
            spec,
            run_dir,
            command,
            code_sha,
            profile_bundle_sha,
            args.rerun_failed,
            args.run_timeout_seconds,
            args.stall_timeout_seconds,
            args.stall_sim_seconds,
            args.retry_timeouts,
        ))

    failures = []
    if not args.dry_run:
        failures = _run_specs(
            repo_root,
            manifest,
            args.jobs,
            args.keep_going,
            run_args,
        )

    write_json(
        output_root / "plan.json",
        {
            "schema_version": 1,
            "experiment_id": manifest["experiment_id"],
            "phase": args.phase,
            "git_sha": code_sha,
            "manifest_sha256": sha256_file(manifest_path),
            "dry_run": args.dry_run,
            "runs": plan_rows,
        },
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} matrix run(s) failed: "
            + ", ".join(failures[:10])
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the native HBF parallel-mode experiment matrix."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--phase", choices=(
        "stage1", "anchors", "routing", "network", "winners", "all"
    ), default="stage1")
    parser.add_argument(
        "--selection",
        help="Selection JSON emitted by validate_hbf_parallel_results.py",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--jobs",
        type=_positive_int,
        default=1,
        help="number of simulator subprocesses to run concurrently",
    )
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument(
        "--retry-timeouts",
        action="store_true",
        help="rerun failures previously classified as watchdog timeouts",
    )
    parser.add_argument(
        "--run-timeout-seconds",
        type=_positive_float,
        default=DEFAULT_RUN_TIMEOUT_SECONDS,
        help="absolute wall-clock limit for one simulator subprocess",
    )
    parser.add_argument(
        "--stall-timeout-seconds",
        type=_positive_float,
        default=DEFAULT_STALL_TIMEOUT_SECONDS,
        help="wall-clock zero-throughput interval before stall detection",
    )
    parser.add_argument(
        "--stall-sim-seconds",
        type=_positive_float,
        default=DEFAULT_STALL_SIM_SECONDS,
        help="minimum simulated-time advance required to classify a stall",
    )
    return parser


def main():
    run_matrix(build_parser().parse_args())


if __name__ == "__main__":
    main()
