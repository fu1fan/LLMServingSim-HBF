#!/usr/bin/env python3
"""Independently validate and summarize HBF parallel-matrix outputs."""

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    from scripts.hbf_sweep import summarize_requests
except ModuleNotFoundError:
    from hbf_sweep import summarize_requests


ROUTING_RE = re.compile(
    r"local=(\[[0-9,\s]+\]).*activated=(\[[0-9,\s]+\])"
)
PAIR_METRICS = (
    "ttft_ms_mean",
    "ttft_ms_p50",
    "ttft_ms_p90",
    "ttft_ms_p99",
    "tpot_ms_mean",
    "tpot_ms_p50",
    "tpot_ms_p90",
    "tpot_ms_p99",
    "prompt_throughput_tok_s",
    "generation_throughput_tok_s",
    "total_throughput_tok_s",
)
EXPECTED_STATIC_RUN_COUNT = 1176


def gini(values):
    values = [float(value) for value in values]
    total = sum(values)
    if not values or total == 0:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    weighted = sum(
        (index + 1) * value for index, value in enumerate(ordered)
    )
    return 2 * weighted / (n * total) - (n + 1) / n


def coefficient_of_variation(values):
    values = [float(value) for value in values]
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


def parse_routing_log(path):
    layers = []
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = ROUTING_RE.search(line)
            if not match:
                continue
            local = ast.literal_eval(match.group(1))
            activated = ast.literal_eval(match.group(2))
            layers.append(
                {
                    "local": local,
                    "activated": activated,
                    "max": max(local, default=0),
                    "mean": sum(local) / len(local) if local else 0.0,
                    "cv": coefficient_of_variation(local),
                    "gini": gini(local),
                }
            )
    if not layers:
        return {}
    ratios = [
        layer["max"] / layer["mean"]
        for layer in layers
        if layer["mean"] > 0
    ]
    return {
        "routing_layer_samples": len(layers),
        "ep_rank_tokens_max": max(layer["max"] for layer in layers),
        "ep_rank_tokens_mean": (
            sum(layer["mean"] for layer in layers) / len(layers)
        ),
        "ep_rank_load_cv_mean": (
            sum(layer["cv"] for layer in layers) / len(layers)
        ),
        "ep_rank_load_gini_mean": (
            sum(layer["gini"] for layer in layers) / len(layers)
        ),
        "activated_experts_per_rank_mean": (
            sum(
                sum(layer["activated"]) / len(layer["activated"])
                for layer in layers
                if layer["activated"]
            )
            / len(layers)
        ),
        "slow_rank_amplification_mean": sum(ratios) / len(ratios),
    }


def aggregate_runtime(path):
    runtime = json.loads(path.read_text(encoding="utf-8"))
    result = {
        "runtime_num_devices": runtime["num_devices"],
        "hbf_energy_unmodeled": runtime["hbf_energy_unmodeled"],
    }
    fields = (
        "hbm_capacity_bytes",
        "hbm_weight_used_bytes",
        "hbm_kv_capacity_bytes",
        "hbf_capacity_bytes",
        "hbf_weight_used_bytes",
    )
    for field in fields:
        result[field] = sum(
            int(instance[field]) * int(instance["num_devices"])
            for instance in runtime["instances"]
        )
    return result


def classify_failure(manifest, log_path):
    text = ""
    if log_path.is_file():
        text = log_path.read_text(
            encoding="utf-8", errors="replace"
        ).lower()
    tier = manifest["spec"]["memory_tier"]
    if "required=" in text and "available=" in text:
        return f"{tier}-capacity-precheck-failed"
    return "simulator-failed"


def _evidence_labels(manifest, runtime):
    labels = set(
        value for value in manifest["spec"]["evidence_level"].split("+")
        if value
    )
    labels.add("b200-operator-profile-backed")
    if manifest["spec"]["routing_policy"] == "CUSTOM":
        labels.add("community-routing-statistics")
    if runtime and runtime.get("hbf_energy_unmodeled"):
        labels.add("hbf-energy-unmodeled")
    return "+".join(sorted(labels))


def summarize_run(run_dir):
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = manifest["spec"]
    base = {
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "git_sha": manifest["git_sha"],
        "profile_bundle_sha256": manifest.get(
            "profile_bundle_sha256"
        ),
        "model_id": spec["model_id"],
        "model_name": spec["model_name"],
        "topology_id": spec["topology_id"],
        "workload_id": spec["workload_id"],
        "routing_policy": spec["routing_policy"],
        "routing_seed": spec["routing_seed"],
        "memory_tier": spec["memory_tier"],
        "hbf_scale": spec["hbf_scale"],
        "network_scenario": spec["network_scenario"],
        "num_devices": manifest["num_devices"],
        "routing_profile_sha256": manifest.get(
            "routing_profile_sha256"
        ),
        "workload_sha256": manifest["workload_sha256"],
        "cluster_config_sha256": manifest["cluster_config_sha256"],
    }
    topology = spec["topology"]
    base.update(
        {
            "replicas": topology.get("replicas", 1),
            "tp": topology["tp"],
            "pp": topology["pp"],
            "ep": topology.get("ep", 1),
            "dp_group_size": topology.get("dp_group_size", 1),
        }
    )
    log_path = run_dir / "simulator.log"
    if manifest["status"] != "completed":
        base["failure_class"] = classify_failure(manifest, log_path)
        base["evidence_labels"] = _evidence_labels(manifest, None)
        return base

    requests_path = run_dir / "requests.csv"
    runtime_path = run_dir / "runtime.json"
    metrics = summarize_requests(requests_path, manifest["num_devices"])
    runtime = aggregate_runtime(runtime_path)
    base.update(metrics)
    base.update(runtime)
    base.update(parse_routing_log(log_path))
    base["failure_class"] = ""
    base["evidence_labels"] = _evidence_labels(manifest, runtime)
    return base


def discover_runs(output_root):
    return sorted(
        path.parent for path in output_root.rglob("manifest.json")
    )


def audit_plan_coverage(output_root, rows):
    plan_path = output_root / "plan.json"
    if not plan_path.is_file():
        return {
            "status": "fail",
            "expected_run_count": EXPECTED_STATIC_RUN_COUNT,
            "planned_run_count": 0,
            "manifest_run_count": len(rows),
            "missing_run_ids": [],
            "extra_run_ids": [],
            "duplicate_plan_run_ids": [],
            "duplicate_manifest_run_ids": [],
            "reason": "plan.json is missing",
        }

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    planned = [run["run_id"] for run in plan.get("runs", [])]
    actual = [row["run_id"] for row in rows]
    planned_counts = Counter(planned)
    actual_counts = Counter(actual)
    missing = sorted(set(planned) - set(actual))
    extra = sorted(set(actual) - set(planned))
    duplicate_plan = sorted(
        run_id for run_id, count in planned_counts.items() if count != 1
    )
    duplicate_actual = sorted(
        run_id for run_id, count in actual_counts.items() if count != 1
    )
    valid = (
        plan.get("phase") == "all"
        and len(planned) == EXPECTED_STATIC_RUN_COUNT
        and not missing
        and not extra
        and not duplicate_plan
        and not duplicate_actual
    )
    return {
        "status": "pass" if valid else "fail",
        "expected_run_count": EXPECTED_STATIC_RUN_COUNT,
        "planned_run_count": len(planned),
        "manifest_run_count": len(actual),
        "missing_run_ids": missing,
        "extra_run_ids": extra,
        "duplicate_plan_run_ids": duplicate_plan,
        "duplicate_manifest_run_ids": duplicate_actual,
        "reason": "" if valid else "plan-to-manifest coverage mismatch",
    }


def _pair_key(row, include_routing=True):
    key = [
        row["model_id"],
        row["topology_id"],
        row["workload_id"],
        row["routing_seed"],
        row["network_scenario"],
    ]
    if include_routing:
        key.append(row["routing_policy"])
    return tuple(key)


def compare_hbm_hbf_identity(rows, tolerance=1e-12):
    groups = defaultdict(dict)
    for row in rows:
        if row["status"] != "completed" or float(row["hbf_scale"]) != 1.0:
            continue
        groups[_pair_key(row)][row["memory_tier"]] = row
    comparisons = []
    for key, pair in groups.items():
        if set(pair) != {"hbm", "hbf"}:
            continue
        mismatches = []
        for metric in PAIR_METRICS:
            left = float(pair["hbm"][metric])
            right = float(pair["hbf"][metric])
            if not math.isclose(left, right, rel_tol=tolerance,
                                abs_tol=tolerance):
                mismatches.append(metric)
        comparisons.append(
            {
                "comparison": "hbm-hbf-k1-identity",
                "key": "|".join(map(str, key)),
                "status": "pass" if not mismatches else "fail",
                "mismatched_metrics": ",".join(mismatches),
            }
        )
    return comparisons


def compare_routing(rows):
    groups = defaultdict(dict)
    for row in rows:
        if row["status"] != "completed":
            continue
        groups[
            (
                row["model_id"],
                row["topology_id"],
                row["workload_id"],
                row["memory_tier"],
                row["hbf_scale"],
                row["network_scenario"],
                row["routing_seed"],
            )
        ][row["routing_policy"]] = row
    comparisons = []
    for key, pair in groups.items():
        if not {"BALANCED", "CUSTOM"}.issubset(pair):
            continue
        balanced = pair["BALANCED"]
        custom = pair["CUSTOM"]
        comparisons.append(
            {
                "comparison": "custom-minus-balanced",
                "key": "|".join(map(str, key)),
                "status": "observed",
                "generation_throughput_delta_tok_s": (
                    custom["generation_throughput_tok_s"]
                    - balanced["generation_throughput_tok_s"]
                ),
                "ttft_p99_delta_ms": (
                    custom["ttft_ms_p99"] - balanced["ttft_ms_p99"]
                ),
                "tpot_p99_delta_ms": (
                    custom["tpot_ms_p99"] - balanced["tpot_ms_p99"]
                ),
            }
        )
    return comparisons


def select_winners(rows, selection_contract):
    completed = [
        row for row in rows
        if row["status"] == "completed"
        and row["memory_tier"] == "hbf"
        and float(row["hbf_scale"]) == 1.0
        and row["network_scenario"] == "central"
        and row["routing_policy"] in {"BALANCED", "CUSTOM"}
    ]
    groups = defaultdict(list)
    for row in completed:
        budget = int(row["num_devices"])
        if budget not in selection_contract["device_budgets"]:
            continue
        groups[
            (
                row["model_id"],
                row["workload_id"],
                row["routing_policy"],
                budget,
            )
        ].append(row)

    selections = []
    floor_ratio = float(selection_contract["throughput_floor_ratio"])
    for key, candidates in sorted(groups.items()):
        throughput = sorted(
            candidates,
            key=lambda row: (
                -row["generation_throughput_tok_s_per_device"],
                row["ttft_ms_p99"],
                row["tpot_ms_p99"],
                row["topology_id"],
            ),
        )[0]
        max_total = max(row["total_throughput_tok_s"]
                        for row in candidates)
        near_peak = [
            row for row in candidates
            if row["total_throughput_tok_s"] >= floor_ratio * max_total
        ]
        latency = sorted(
            near_peak,
            key=lambda row: (
                row["ttft_ms_p99"],
                row["tpot_ms_p99"],
                row["topology_id"],
            ),
        )[0]
        selections.append(
            {
                "model_id": key[0],
                "workload_id": key[1],
                "routing_policy": key[2],
                "device_budget": key[3],
                "throughput_topology_id": throughput["topology_id"],
                "latency_topology_id": latency["topology_id"],
            }
        )
    return {"schema_version": 1, "selections": selections}


def annotate_capacity_enabled(rows):
    hbm_capacity_failures = {
        _pair_key(row)
        for row in rows
        if row["status"] != "completed"
        and row.get("memory_tier") == "hbm"
        and row.get("failure_class") == "hbm-capacity-precheck-failed"
    }
    for row in rows:
        if (
            row["status"] == "completed"
            and row["memory_tier"] == "hbf"
            and float(row["hbf_scale"]) == 1.0
            and _pair_key(row) in hbm_capacity_failures
        ):
            labels = set(row["evidence_labels"].split("+"))
            labels.add("capacity-enabled-no-hbm-pair")
            row["evidence_labels"] = "+".join(sorted(labels))


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate(args):
    output_root = Path(args.output_dir).resolve()
    manifest = json.loads(Path(args.experiment_manifest).read_text(
        encoding="utf-8"
    ))
    rows = [summarize_run(path) for path in discover_runs(output_root)]
    annotate_capacity_enabled(rows)
    completed = [row for row in rows if row["status"] == "completed"]
    failures = [row for row in rows if row["status"] != "completed"]
    comparisons = (
        compare_hbm_hbf_identity(rows) + compare_routing(rows)
    )
    winners = select_winners(completed, manifest["selection"])
    coverage = audit_plan_coverage(output_root, rows)

    write_csv(output_root / "summary.csv", completed)
    write_csv(output_root / "failures.csv", failures)
    write_csv(output_root / "comparisons.csv", comparisons)
    (output_root / "winners.json").write_text(
        json.dumps(winners, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    identity_failures = [
        row for row in comparisons
        if row["comparison"] == "hbm-hbf-k1-identity"
        and row["status"] == "fail"
    ]
    failure_classes = Counter(
        row.get("failure_class", "unclassified") for row in failures
    )
    validation = {
        "schema_version": 1,
        "status": (
            "pass"
            if coverage["status"] == "pass" and not identity_failures
            else "fail"
        ),
        "coverage": coverage,
        "completed_run_count": len(completed),
        "failed_run_count": len(failures),
        "failure_classes": dict(sorted(failure_classes.items())),
        "identity_comparison_failure_count": len(identity_failures),
    }
    (output_root / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    errors = []
    if coverage["status"] != "pass":
        errors.append(
            "static matrix coverage mismatch: "
            f"planned={coverage['planned_run_count']} "
            f"manifests={coverage['manifest_run_count']} "
            f"missing={len(coverage['missing_run_ids'])} "
            f"extra={len(coverage['extra_run_ids'])}"
        )
    if identity_failures:
        errors.append(
            f"{len(identity_failures)} HBM/HBF k=1 pairs differ"
        )
    if errors:
        raise RuntimeError("; ".join(errors))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate HBF parallel-matrix output independently."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--experiment-manifest",
        default="configs/experiments/hbf_parallel_modes_v1.json",
    )
    return parser


if __name__ == "__main__":
    validate(build_parser().parse_args())
