#!/usr/bin/env python3
"""Validate an LLMServingSim profiler output bundle."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


LATENCY_COLUMNS = ("microseconds", "time_us", "latency_us", "latency")
COMMON_CSVS = ("dense.csv", "per_sequence.csv", "attention.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate LLMServingSim CSVs, meta.yaml, and TP coverage."
    )
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--hardware", default="B200")
    parser.add_argument("--tp", required=True, help="Expected TP list, e.g. 1,2,4,8")
    parser.add_argument("--moe", action="store_true", help="Require moe.csv")
    parser.add_argument(
        "--skew",
        action="store_true",
        help="Require and validate skew.csv plus enabled skew metadata",
    )
    parser.add_argument("--expected-vllm-version", default="0.19.0")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        # macOS ships Ruby with a YAML parser, while the pinned vLLM image
        # already ships PyYAML. The fallback keeps local validation
        # dependency-free without changing the server-side path.
        ruby = (
            "data = YAML.safe_load(File.read(ARGV[0]), "
            "permitted_classes: [], permitted_symbols: [], aliases: true); "
            "puts JSON.generate(data)"
        )
        try:
            result = subprocess.run(
                ["ruby", "-ryaml", "-rjson", "-e", ruby, str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "PyYAML is not installed and the Ruby YAML fallback failed. "
                "Run this checker inside the pinned vLLM profiler container."
            ) from exc
        data = json.loads(result.stdout)
    else:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("meta.yaml must contain a YAML mapping")
    return data


def parse_tp_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or len(values) != len(set(values)) or any(value < 1 for value in values):
        raise ValueError("--tp must be a non-empty, unique list of positive integers")
    return values


def is_nan_token(value: str) -> bool:
    return value.strip().lower() in {"nan", "+nan", "-nan"}


def validate_csv(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing CSV: {path}")
        return
    if path.stat().st_size == 0:
        errors.append(f"empty CSV file: {path}")
        return

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        latency_column = next((name for name in LATENCY_COLUMNS if name in fieldnames), None)
        if latency_column is None:
            errors.append(
                f"{path}: no latency column; expected one of {', '.join(LATENCY_COLUMNS)}"
            )
        rows = list(reader)

    if not rows:
        errors.append(f"{path}: header exists but there are no data rows")
        return

    for row_number, row in enumerate(rows, start=2):
        for column, raw_value in row.items():
            value = "" if raw_value is None else raw_value.strip()
            if is_nan_token(value):
                errors.append(f"{path}:{row_number}: NaN in column {column}")
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number):
                errors.append(
                    f"{path}:{row_number}: non-finite value {value!r} in column {column}"
                )

        if latency_column is None:
            continue
        raw_latency = row.get(latency_column, "")
        try:
            latency = float(raw_latency)
        except (TypeError, ValueError):
            errors.append(
                f"{path}:{row_number}: invalid latency {raw_latency!r} "
                f"in column {latency_column}"
            )
            continue
        if not math.isfinite(latency) or latency <= 0:
            errors.append(
                f"{path}:{row_number}: latency must be finite and positive, got {latency}"
            )


def validate_skew_csv(path: Path, errors: list[str]) -> None:
    """Validate the optional skew profile's three measured latency columns."""
    if not path.is_file():
        errors.append(f"missing CSV: {path}")
        return
    if path.stat().st_size == 0:
        errors.append(f"empty CSV file: {path}")
        return

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        latency_columns = ("t_mean_us", "t_max_us", "t_skew_us")
        for column in latency_columns:
            if column not in fieldnames:
                errors.append(f"{path}: missing skew latency column {column}")
        rows = list(reader)

    if not rows:
        errors.append(f"{path}: header exists but there are no data rows")
        return

    for row_number, row in enumerate(rows, start=2):
        for column, raw_value in row.items():
            value = "" if raw_value is None else raw_value.strip()
            if is_nan_token(value):
                errors.append(f"{path}:{row_number}: NaN in column {column}")
            if not value:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number):
                errors.append(
                    f"{path}:{row_number}: non-finite value {value!r} in column {column}"
                )
        for column in latency_columns:
            raw_latency = row.get(column, "")
            try:
                latency = float(raw_latency)
            except (TypeError, ValueError):
                errors.append(
                    f"{path}:{row_number}: invalid latency {raw_latency!r} "
                    f"in column {column}"
                )
                continue
            if not math.isfinite(latency) or latency <= 0:
                errors.append(
                    f"{path}:{row_number}: {column} must be finite and positive, "
                    f"got {latency}"
                )


def nested(meta: dict[str, Any], *keys: str) -> Any:
    current: Any = meta
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def validate_meta(
    path: Path,
    *,
    model_id: str,
    hardware: str,
    expected_tps: list[int],
    expected_vllm_version: str,
    expect_skew: bool,
    errors: list[str],
) -> None:
    if not path.is_file():
        errors.append(f"missing metadata: {path}")
        return
    if path.stat().st_size == 0:
        errors.append(f"empty metadata: {path}")
        return

    try:
        meta = load_yaml(path)
    except Exception as exc:
        errors.append(f"cannot read {path}: {exc}")
        return

    exact_checks = {
        "model": model_id,
        "hardware": hardware,
        "variant": "bf16",
        "vllm_version": expected_vllm_version,
        "measurement_iterations": 3,
    }
    for key, expected in exact_checks.items():
        actual = meta.get(key)
        if str(actual) != str(expected):
            errors.append(f"{path}: {key}={actual!r}, expected {expected!r}")

    actual_tps = meta.get("tp_degrees")
    try:
        normalized_tps = [int(value) for value in actual_tps]
    except (TypeError, ValueError):
        normalized_tps = []
    if normalized_tps != expected_tps:
        errors.append(
            f"{path}: tp_degrees={normalized_tps!r}, expected {expected_tps!r}"
        )

    gpu_name = str(meta.get("gpu") or "")
    if "B200" not in gpu_name:
        errors.append(f"{path}: gpu={gpu_name!r} does not identify a B200")
    if not meta.get("cuda_version"):
        errors.append(f"{path}: cuda_version is missing")
    if not meta.get("profiler_version"):
        errors.append(f"{path}: profiler_version is missing")
    if not meta.get("profiled_at"):
        errors.append(f"{path}: profiled_at is missing")

    if nested(meta, "attention_grid", "max_kv") != 16384:
        errors.append(
            f"{path}: attention_grid.max_kv="
            f"{nested(meta, 'attention_grid', 'max_kv')!r}, expected 16384"
        )
    expected_skew_enabled = bool(expect_skew)
    if nested(meta, "skew_profile", "enabled") is not expected_skew_enabled:
        errors.append(
            f"{path}: skew_profile.enabled must be {expected_skew_enabled}"
        )
    if nested(meta, "skew_fit", "enabled") is not expected_skew_enabled:
        errors.append(
            f"{path}: skew_fit.enabled must be {expected_skew_enabled}"
        )

    dtype = nested(meta, "engine_effective", "dtype")
    if str(dtype).lower() not in {"bfloat16", "bf16"}:
        errors.append(
            f"{path}: engine_effective.dtype={dtype!r}, expected bfloat16"
        )
    if nested(meta, "engine_effective", "max_num_batched_tokens") not in {2048, "2048"}:
        errors.append(
            f"{path}: engine_effective.max_num_batched_tokens must be 2048"
        )
    if nested(meta, "engine_effective", "max_num_seqs") not in {256, "256"}:
        errors.append(f"{path}: engine_effective.max_num_seqs must be 256")


def main() -> int:
    args = parse_args()
    try:
        expected_tps = parse_tp_list(args.tp)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    profile_dir = args.profile_dir.resolve()
    errors: list[str] = []
    validate_meta(
        profile_dir / "meta.yaml",
        model_id=args.model_id,
        hardware=args.hardware,
        expected_tps=expected_tps,
        expected_vllm_version=args.expected_vllm_version,
        expect_skew=args.skew,
        errors=errors,
    )

    required_csvs = list(COMMON_CSVS)
    if args.moe:
        required_csvs.append("moe.csv")

    expected_dirs = {f"tp{tp}" for tp in expected_tps}
    actual_dirs = {
        path.name
        for path in profile_dir.glob("tp*")
        if path.is_dir() and path.name[2:].isdigit()
    }
    missing_dirs = sorted(expected_dirs - actual_dirs)
    if missing_dirs:
        errors.append(f"missing TP directories: {', '.join(missing_dirs)}")

    for tp in expected_tps:
        tp_dir = profile_dir / f"tp{tp}"
        for filename in required_csvs:
            validate_csv(tp_dir / filename, errors)
        if args.skew:
            validate_skew_csv(tp_dir / "skew.csv", errors)

    if errors:
        print(f"FAILED: {len(errors)} validation error(s)", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    total_csvs = len(expected_tps) * (len(required_csvs) + int(args.skew))
    print(
        f"PASS: {profile_dir} contains {total_csvs} valid CSV files, "
        f"complete TP coverage {expected_tps}, and matching meta.yaml."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
