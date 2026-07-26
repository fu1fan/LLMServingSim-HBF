#!/usr/bin/env python3
"""Run one fresh LLMServingSim process per HBF latency coefficient."""

import argparse
import json
import math
import os
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
        command=tuple(command),
    )


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

    failures = []
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
                cwd=Path(__file__).resolve().parents[1],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            failures.append((run, completed.returncode))
            if not args.keep_going:
                break

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
