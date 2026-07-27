#!/usr/bin/env python3
"""Generate and run the native LLMServingSim HBF parallel-mode matrix."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_MANIFEST = "configs/experiments/hbf_parallel_modes_v1.json"
STRESS_SCALES = (1.0, 4.0, 10.0)


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
    pp = int(topology["pp"])
    num_dims = 2 if dp_group is not None or replicas * pp > 1 else 1
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
    failures = []

    for spec in specs:
        run_dir = output_root / spec.run_id
        cluster = build_cluster_config(manifest, spec)
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

        previous_manifest = run_dir / "manifest.json"
        if previous_manifest.is_file() and not args.rerun_failed:
            previous = json.loads(
                previous_manifest.read_text(encoding="utf-8")
            )
            if previous.get("status") == "completed":
                print(f"[skip] {spec.run_id}", flush=True)
                continue

        run_dir.mkdir(parents=True, exist_ok=True)
        cluster_path = run_dir / "cluster.json"
        write_json(cluster_path, cluster)
        log_path = run_dir / "simulator.log"
        print(f"[run] {spec.run_id}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        status = "completed" if completed.returncode == 0 else "failed"
        run_manifest = {
            "schema_version": 1,
            "experiment_id": manifest["experiment_id"],
            "run_id": spec.run_id,
            "status": status,
            "exit_code": completed.returncode,
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
            "outputs": {
                "requests": os.fspath(run_dir / "requests.csv"),
                "runtime": os.fspath(run_dir / "runtime.json"),
                "log": os.fspath(log_path),
            },
        }
        write_json(previous_manifest, run_manifest)
        if completed.returncode != 0:
            failures.append(spec.run_id)
            if not args.keep_going:
                break

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
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    return parser


def main():
    run_matrix(build_parser().parse_args())


if __name__ == "__main__":
    main()
