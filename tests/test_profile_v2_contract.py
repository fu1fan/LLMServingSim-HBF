import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from serving.core.profile_contract import (
    ProfileContractError,
    ProfileV2RuntimeNotReadyError,
    load_profile_contract,
    validate_profile_meta,
)
from serving.core import trace_generator


TABLE_COLUMNS = {
    "dense.csv": ("layer", "tokens"),
    "per_sequence.csv": ("layer", "sequences"),
    "attention.csv": (
        "prefill_chunk",
        "kv_prefill",
        "n_decode",
        "kv_decode",
    ),
    "moe.csv": ("tokens", "activated_experts"),
}

AUDIT_COLUMNS = (
    "hbm_read_bytes",
    "hbm_write_bytes",
    "hbf_read_bytes",
    "hbf_write_bytes",
)


def _v2_meta(profile_id="cli-a"):
    return {
        "profile_schema_version": 2,
        "memory_profile_id": profile_id,
        "scenario_catalog": {
            "all_hbm": {
                "strict": True,
                "accesses": {
                    "op/input": "hbm",
                    "op/output": "hbm",
                },
            },
            "input_hbf": {
                "strict": True,
                "accesses": {
                    "op/input": "hbf",
                    "op/output": "hbm",
                },
            },
        },
        "latency_accounting": {
            "demand_access_included": True,
            "includes": [
                "compute",
                "hbm_demand_access",
                "hbf_demand_access",
            ],
            "excludes": [
                "migration",
                "prefetch",
                "eviction",
                "network_collective",
            ],
        },
    }


def _base_row(filename, scenario="all_hbm"):
    shapes = {
        "dense.csv": {"layer": "qkv_proj", "tokens": 128},
        "per_sequence.csv": {"layer": "lm_head", "sequences": 4},
        "attention.csv": {
            "prefill_chunk": 128,
            "kv_prefill": 0,
            "n_decode": 4,
            "kv_decode": 256,
        },
        "moe.csv": {"tokens": 128, "activated_experts": 2},
    }
    return {
        **shapes[filename],
        "memory_scenario": scenario,
        "time_us": 12.5,
        "hbm_read_bytes": 4096,
        "hbm_write_bytes": 1024,
        "hbf_read_bytes": 0,
        "hbf_write_bytes": 0,
    }


def _write_yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(value, f, sort_keys=False)


def _write_csv(path, filename, rows, *, omit_columns=()):
    columns = [
        *TABLE_COLUMNS[filename],
        "memory_scenario",
        "time_us",
        *AUDIT_COLUMNS,
    ]
    columns = [column for column in columns if column not in omit_columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _make_v2_bundle(
    root,
    *,
    profile_id="cli-a",
    filenames=None,
    rows_by_file=None,
    meta=None,
):
    root = Path(root)
    _write_yaml(root / "meta.yaml", meta or _v2_meta(profile_id))
    filenames = filenames or tuple(TABLE_COLUMNS)
    rows_by_file = rows_by_file or {}
    for filename in filenames:
        rows = rows_by_file.get(filename, [_base_row(filename)])
        _write_csv(root / "tp1" / filename, filename, rows)
    return root


class ProfileV2ContractTest(unittest.TestCase):
    def tearDown(self):
        trace_generator._perf_db_cache.clear()

    def test_missing_schema_version_is_legacy_v1(self):
        contract = validate_profile_meta(
            {"profiler_version": "legacy"},
            source="legacy/meta.yaml",
        )

        self.assertEqual(contract.schema_version, 1)
        self.assertFalse(contract.is_v2)
        self.assertIsNone(contract.memory_profile_id)

    def test_valid_v2_bundle_validates_every_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_v2_bundle(Path(tmp) / "bundle")

            contract = load_profile_contract(
                str(root),
                requested_memory_profile_id="cli-a",
            )

        self.assertTrue(contract.is_v2)
        self.assertEqual(contract.memory_profile_id, "cli-a")
        self.assertEqual(set(contract.scenario_catalog), {"all_hbm", "input_hbf"})

    def test_v2_requires_explicit_matching_identity(self):
        meta = _v2_meta("cli-a")

        with self.assertRaisesRegex(ProfileContractError, "显式选择"):
            validate_profile_meta(meta, source="meta.yaml")
        with self.assertRaisesRegex(ProfileContractError, "不一致"):
            validate_profile_meta(
                meta,
                requested_memory_profile_id="csi-b",
                source="meta.yaml",
            )
        with self.assertRaisesRegex(ProfileContractError, "合法标识"):
            validate_profile_meta(
                meta,
                requested_memory_profile_id="../cli-a",
                source="meta.yaml",
            )

    def test_v2_requires_canonical_scenario_mapping(self):
        cases = {
            "empty catalog": {},
            "missing mapping": {"all_hbm": {"strict": True}},
            "unknown tier": {
                "all_hbm": {"accesses": {"op/input": "cpu"}},
            },
            "ambiguous mapping": {
                "all_hbm": {
                    "accesses": {"op/input": "hbm"},
                    "placements": {"op/input": "hbm"},
                },
            },
        }
        for name, catalog in cases.items():
            with self.subTest(name=name):
                meta = _v2_meta()
                meta["scenario_catalog"] = catalog
                with self.assertRaises(ProfileContractError):
                    validate_profile_meta(
                        meta,
                        requested_memory_profile_id="cli-a",
                        source="meta.yaml",
                    )

    def test_v2_requires_complete_latency_accounting(self):
        cases = {
            "demand not included": {
                "demand_access_included": False,
                "includes": [
                    "compute",
                    "hbm_demand_access",
                    "hbf_demand_access",
                ],
                "excludes": [
                    "migration",
                    "prefetch",
                    "eviction",
                    "network_collective",
                ],
            },
            "missing include": {
                "demand_access_included": True,
                "includes": ["compute", "hbm_demand_access"],
                "excludes": [
                    "migration",
                    "prefetch",
                    "eviction",
                    "network_collective",
                ],
            },
            "missing exclude": {
                "demand_access_included": True,
                "includes": [
                    "compute",
                    "hbm_demand_access",
                    "hbf_demand_access",
                ],
                "excludes": ["migration", "prefetch", "eviction"],
            },
        }
        for name, accounting in cases.items():
            with self.subTest(name=name):
                meta = _v2_meta()
                meta["latency_accounting"] = accounting
                with self.assertRaises(ProfileContractError):
                    validate_profile_meta(
                        meta,
                        requested_memory_profile_id="cli-a",
                        source="meta.yaml",
                    )

    def test_each_existing_category_requires_all_v2_columns(self):
        for filename in TABLE_COLUMNS:
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / "bundle"
                    _write_yaml(root / "meta.yaml", _v2_meta())
                    _write_csv(
                        root / "tp1" / filename,
                        filename,
                        [_base_row(filename)],
                        omit_columns=("hbf_write_bytes",),
                    )
                    with self.assertRaisesRegex(
                        ProfileContractError,
                        "hbf_write_bytes",
                    ):
                        load_profile_contract(
                            str(root),
                            requested_memory_profile_id="cli-a",
                        )

    def test_csv_rejects_unknown_scenario_invalid_values_and_duplicates(self):
        cases = {
            "unknown scenario": [
                {**_base_row("dense.csv"), "memory_scenario": "unknown"},
            ],
            "non-positive time": [
                {**_base_row("dense.csv"), "time_us": 0},
            ],
            "negative audit bytes": [
                {**_base_row("dense.csv"), "hbf_read_bytes": -1},
            ],
            "duplicate shape and scenario": [
                _base_row("dense.csv"),
                _base_row("dense.csv"),
            ],
        }
        for name, rows in cases.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp:
                    root = _make_v2_bundle(
                        Path(tmp) / "bundle",
                        filenames=("dense.csv",),
                        rows_by_file={"dense.csv": rows},
                    )
                    with self.assertRaises(ProfileContractError):
                        load_profile_contract(
                            str(root),
                            requested_memory_profile_id="cli-a",
                        )

    def test_v2_requires_at_least_one_performance_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bundle"
            _write_yaml(root / "meta.yaml", _v2_meta())

            with self.assertRaisesRegex(ProfileContractError, "至少需要一个"):
                load_profile_contract(
                    str(root),
                    requested_memory_profile_id="cli-a",
                )

    def test_profile_root_and_cache_key_keep_v1_compatibility(self):
        with mock.patch.object(
            trace_generator,
            "_PROFILER_ROOT_REL",
            "/profiles",
        ):
            self.assertEqual(
                trace_generator._profile_root("gpu", "org/model", "bf16"),
                "/profiles/perf/gpu/org/model/bf16",
            )
            self.assertEqual(
                trace_generator._profile_root(
                    "gpu",
                    "org/model",
                    "bf16",
                    "cli-a",
                ),
                "/profiles/perf/gpu/org/model/bf16/cli-a",
            )

        self.assertEqual(
            trace_generator._perf_db_cache_key("gpu", "model", "bf16"),
            ("gpu", "model", "bf16"),
        )
        self.assertEqual(
            trace_generator._perf_db_cache_key(
                "gpu",
                "model",
                "bf16",
                "cli-a",
            ),
            ("gpu", "model", "bf16", "cli-a"),
        )

    def test_runtime_load_rejects_valid_v2_until_scenario_lookup_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiler_root = Path(tmp) / "profiler"
            bundle = (
                profiler_root
                / "perf"
                / "gpu"
                / "org"
                / "model"
                / "bf16"
                / "cli-a"
            )
            _make_v2_bundle(bundle, filenames=("dense.csv",))

            with mock.patch.object(
                trace_generator,
                "_PROFILER_ROOT_REL",
                str(profiler_root),
            ):
                with self.assertRaisesRegex(
                    ProfileV2RuntimeNotReadyError,
                    "scenario-aware lookup",
                ):
                    trace_generator._load_perf_db(
                        "gpu",
                        "org/model",
                        "bf16",
                        {1},
                        "llama",
                        memory_profile_id="cli-a",
                    )

        self.assertNotIn(
            ("gpu", "org/model", "bf16", "cli-a"),
            trace_generator._perf_db_cache,
        )

    def test_legacy_v1_runtime_load_and_cache_key_remain_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiler_root = Path(tmp) / "profiler"
            bundle = (
                profiler_root
                / "perf"
                / "gpu"
                / "org"
                / "model"
                / "bf16"
            )
            _write_yaml(bundle / "meta.yaml", {"profiler_version": "legacy"})
            dense_columns = ["layer", "tokens", "time_us"]
            (bundle / "tp1").mkdir(parents=True)
            with (bundle / "tp1" / "dense.csv").open(
                "w",
                encoding="utf-8",
                newline="",
            ) as f:
                writer = csv.DictWriter(f, fieldnames=dense_columns)
                writer.writeheader()
                writer.writerow(
                    {"layer": "qkv_proj", "tokens": 128, "time_us": 12.5}
                )

            with mock.patch.object(
                trace_generator,
                "_PROFILER_ROOT_REL",
                str(profiler_root),
            ):
                perf_db = trace_generator._load_perf_db(
                    "gpu",
                    "org/model",
                    "bf16",
                    {1},
                    "llama",
                )

        self.assertEqual(perf_db["available_tps"], [1])
        self.assertIn(
            ("gpu", "org/model", "bf16"),
            trace_generator._perf_db_cache,
        )
        self.assertEqual(
            perf_db["tables"][1]["dense"]["qkv_proj"]["values"],
            [12500],
        )


if __name__ == "__main__":
    unittest.main()
