#!/usr/bin/env python3
"""Run one fresh LLMServingSim process per HBF latency coefficient."""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PAPER_CORE_SCALES = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.5, 10.0)


def parse_scales(values):
    if values is None:
        return PAPER_CORE_SCALES
    scales = []
    for item in values:
        for token in item.split(","):
            token = token.strip()
            if not token:
                continue
            value = float(token)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Every HBF scale must be a positive number")
            scales.append(value)
    if not scales:
        raise ValueError("At least one HBF scale is required")
    return tuple(scales)


def scale_slug(scale):
    return format(scale, ".12g").replace(".", "p").replace("+", "")


def default_run_prefix():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"hbf-sweep-{stamp}"


def validate_scale_cluster(cluster_config):
    with open(cluster_config, "r", encoding="utf-8") as stream:
        cluster = json.load(stream)
    hbf_instances = []
    for node in cluster.get("nodes", []):
        for instance in node.get("instances", []):
            hbf_mem = instance.get("hbf_mem")
            if hbf_mem is None:
                continue
            hbf_instances.append(instance)
            source = (hbf_mem.get("performance") or {}).get("source")
            if source != "scale":
                raise ValueError(
                    "HBF coefficient sweeps require performance.source='scale' "
                    f"for every HBF instance; got {source!r}"
                )
    if not hbf_instances:
        raise ValueError("HBF coefficient sweep requires an hbf_mem instance")


@dataclass(frozen=True)
class SweepRun:
    scale: float
    run_id: str
    output_dir: Path
    request_csv: Path
    log_file: Path
    runtime_summary: Path
    manifest: Path
    command: tuple


def build_run(
    python,
    cluster_config,
    dataset,
    output_root,
    run_prefix,
    scale,
    num_reqs=0,
    serving_args=(),
):
    run_id = f"{run_prefix}-k{scale_slug(scale)}"
    run_dir = Path(output_root) / f"k{scale_slug(scale)}"
    command = [
        python,
        "-m",
        "serving",
        "--cluster-config",
        os.fspath(Path(cluster_config).resolve()),
        "--dataset",
        os.fspath(Path(dataset).resolve()),
        "--output",
        os.fspath((run_dir / "requests.csv").resolve()),
        "--run-id",
        run_id,
        "--hbf-summary-output",
        os.fspath((run_dir / "runtime.json").resolve()),
        "--hbf-latency-scale",
        format(scale, ".12g"),
    ]
    if num_reqs:
        command.extend(["--num-reqs", str(num_reqs)])
    command.extend(serving_args)
    return SweepRun(
        scale=scale,
        run_id=run_id,
        output_dir=run_dir,
        request_csv=run_dir / "requests.csv",
        log_file=run_dir / "simulator.log",
        runtime_summary=run_dir / "runtime.json",
        manifest=run_dir / "manifest.json",
        command=tuple(command),
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path):
    path = Path(path)
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(file_path)))
    return digest.hexdigest()


def profile_fingerprints(repo_root, cluster_config):
    with open(cluster_config, "r", encoding="utf-8") as stream:
        cluster = json.load(stream)
    profiles = []
    for node in cluster.get("nodes", []):
        for instance in node.get("instances", []):
            model = instance["model_name"]
            baseline = (
                Path(repo_root)
                / "profiler"
                / "perf"
                / instance["hardware"]
                / model
            )
            profiles.append(
                {
                    "kind": "baseline",
                    "path": os.fspath(baseline),
                    "sha256": sha256_tree(baseline),
                }
            )
            hbf_mem = instance.get("hbf_mem") or {}
            performance = hbf_mem.get("performance") or {}
            if performance.get("source") == "profile":
                external = (
                    Path(performance["profile_root"])
                    / performance["profile_hardware"]
                    / model
                )
                profiles.append(
                    {
                        "kind": "hbf",
                        "path": os.fspath(external),
                        "sha256": sha256_tree(external),
                    }
                )
    return profiles


def percentile(values, quantile):
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_requests(request_csv, num_devices):
    with open(request_csv, "r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No completed requests in {request_csv}")

    ttft_ns = [float(row["TTFT"]) for row in rows]
    tpot_ns = [float(row["TPOT"]) for row in rows]
    end_ns = max(float(row["end_time"]) for row in rows)
    if end_ns <= 0:
        raise ValueError("Simulation end time must be positive")
    simulation_seconds = end_ns / 1_000_000_000
    prompt_tokens = sum(int(row["input"]) for row in rows)
    generation_tokens = sum(int(row["output"]) for row in rows)
    total_tokens = prompt_tokens + generation_tokens

    summary = {
        "requests": len(rows),
        "simulation_seconds": simulation_seconds,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "total_tokens": total_tokens,
        "prompt_throughput_tok_s": prompt_tokens / simulation_seconds,
        "generation_throughput_tok_s": (
            generation_tokens / simulation_seconds
        ),
        "total_throughput_tok_s": total_tokens / simulation_seconds,
    }
    for prefix, values in (("ttft", ttft_ns), ("tpot", tpot_ns)):
        summary[f"{prefix}_ms_mean"] = statistics.fmean(values) / 1_000_000
        for label, quantile in (
            ("p50", 0.50),
            ("p90", 0.90),
            ("p99", 0.99),
        ):
            summary[f"{prefix}_ms_{label}"] = (
                percentile(values, quantile) / 1_000_000
            )
    for name in (
        "prompt_throughput_tok_s",
        "generation_throughput_tok_s",
        "total_throughput_tok_s",
    ):
        summary[f"{name}_per_device"] = summary[name] / num_devices
    return summary


def git_sha(repo_root):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_summary_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(args):
    cluster_config = Path(args.cluster_config).resolve()
    dataset = Path(args.dataset).resolve()
    validate_scale_cluster(cluster_config)
    if not dataset.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset}")

    scales = parse_scales(args.scales)
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    prefix = args.run_prefix or default_run_prefix()
    repo_root = Path(__file__).resolve().parents[1]
    code_sha = git_sha(repo_root)
    fingerprints = profile_fingerprints(repo_root, cluster_config)
    config_sha = sha256_file(cluster_config)
    workload_sha = sha256_file(dataset)

    failures = []
    summary_rows = []
    root_runs = []
    for scale in scales:
        run = build_run(
            args.python,
            cluster_config,
            dataset,
            output_root,
            prefix,
            scale,
            args.num_reqs,
            args.serving_args,
        )
        if run.output_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing sweep point: {run.output_dir}"
            )
        run.output_dir.mkdir(parents=True)
        print(
            f"[HBF sweep] scale={scale:g} run_id={run.run_id} "
            f"output={run.output_dir}",
            flush=True,
        )
        with open(run.log_file, "w", encoding="utf-8") as log:
            completed = subprocess.run(
                run.command,
                cwd=repo_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        status = "completed" if completed.returncode == 0 else "failed"
        manifest = {
            "schema_version": 1,
            "run_id": run.run_id,
            "status": status,
            "exit_code": completed.returncode,
            "git_sha": code_sha,
            "cluster_config": {
                "path": os.fspath(cluster_config),
                "sha256": config_sha,
            },
            "workload": {
                "path": os.fspath(dataset),
                "sha256": workload_sha,
            },
            "profiles": fingerprints,
            "latency_scale": scale,
            "performance_source": "scale",
            "evidence_level": "sensitivity-analysis",
            "command": list(run.command),
            "outputs": {
                "requests": os.fspath(run.request_csv),
                "simulator_log": os.fspath(run.log_file),
                "runtime_summary": os.fspath(run.runtime_summary),
            },
        }
        if completed.returncode == 0:
            try:
                with open(
                    run.runtime_summary, "r", encoding="utf-8"
                ) as stream:
                    runtime = json.load(stream)
                metrics = summarize_requests(
                    run.request_csv, runtime["num_devices"]
                )
                memory_rows = runtime["instances"]

                def aggregate(field):
                    return sum(
                        row[field] * row["num_devices"]
                        for row in memory_rows
                    )

                metrics.update(
                    {
                        "run_id": run.run_id,
                        "latency_scale": scale,
                        "performance_source": "scale",
                        "evidence_level": "sensitivity-analysis",
                        "num_devices": runtime["num_devices"],
                        "hbm_weight_used_bytes": aggregate(
                            "hbm_weight_used_bytes"
                        ),
                        "hbf_weight_used_bytes": aggregate(
                            "hbf_weight_used_bytes"
                        ),
                        "hbf_capacity_bytes": aggregate(
                            "hbf_capacity_bytes"
                        ),
                        "hbm_kv_capacity_bytes": aggregate(
                            "hbm_kv_capacity_bytes"
                        ),
                        "hbf_energy_unmodeled": runtime[
                            "hbf_energy_unmodeled"
                        ],
                    }
                )
                summary_rows.append(metrics)
                manifest["metrics"] = metrics
            except Exception as exc:
                status = "failed"
                manifest["status"] = status
                manifest["summary_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                failures.append((run, "summary"))
        write_json(run.manifest, manifest)
        root_runs.append(
            {
                "run_id": run.run_id,
                "latency_scale": scale,
                "status": status,
                "manifest": os.fspath(run.manifest),
            }
        )
        if completed.returncode != 0:
            failures.append((run, completed.returncode))
            if not args.keep_going:
                break
        elif status == "failed" and not args.keep_going:
            break

    write_summary_csv(output_root / "summary.csv", summary_rows)
    write_json(
        output_root / "manifest.json",
        {
            "schema_version": 1,
            "git_sha": code_sha,
            "cluster_config_sha256": config_sha,
            "workload_sha256": workload_sha,
            "profiles": fingerprints,
            "runs": root_runs,
        },
    )

    if failures:
        details = ", ".join(
            f"k={run.scale:g} exit={code}" for run, code in failures
        )
        raise RuntimeError(f"HBF sweep failed: {details}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run each HBF latency coefficient in a separate "
            "`python -m serving` process."
        )
    )
    parser.add_argument("--cluster-config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--scales",
        nargs="+",
        help=(
            "Positive coefficients, separated by spaces or commas. "
            "Defaults to the paper-core set."
        ),
    )
    parser.add_argument("--run-prefix")
    parser.add_argument("--num-reqs", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "serving_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to python -m serving after `--`.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.serving_args[:1] == ["--"]:
        args.serving_args = args.serving_args[1:]
    run_sweep(args)


if __name__ == "__main__":
    main()
