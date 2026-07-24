import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from serving.core import trace_generator
from serving.core.profile_contract import ProfileV2RuntimeNotReadyError


TABLE_SHAPES = {
    "dense.csv": {"layer": "qkv_proj", "tokens": 128},
    "per_sequence.csv": {"layer": "lm_head", "sequences": 4},
    "attention.csv": {
        "prefill_chunk": 0,
        "kv_prefill": 0,
        "n_decode": 1,
        "kv_decode": 16,
    },
    "moe.csv": {"tokens": 128, "activated_experts": 2},
}

AUDIT_COLUMNS = (
    "hbm_read_bytes",
    "hbm_write_bytes",
    "hbf_read_bytes",
    "hbf_write_bytes",
)


TOY_ARCHITECTURE = {
    "sequence": {
        "prologue": [],
        "pre_attn": ["qkv_proj", "attention"],
        "post_attn": [],
        "mlp_dense": [],
        "mlp_moe": ["moe"],
        "head": ["lm_head"],
    },
    "catalog": {
        "dense": {"qkv_proj": {}},
        "per_sequence": {"lm_head": {}},
        "attention": {"attention": {}},
        "moe": {"moe": {}},
    },
}


def _meta(profile_id, *, skew_enabled=False, readiness="runtime_ready"):
    meta = {
        "profile_schema_version": 2,
        "memory_profile_id": profile_id,
        "tp_degrees": [1],
        "bundle_readiness": readiness,
        "runtime_compatible": readiness == "runtime_ready",
        "scenario_binding": (
            "producer_verified_v1"
            if readiness == "runtime_ready"
            else "caller_asserted"
        ),
        "scenario_catalog": {
            "all_hbm": {
                "accesses": {
                    "op/input": "hbm",
                    "op/output": "hbm",
                },
            },
            "input_hbf": {
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
    if readiness == "runtime_ready":
        meta["architecture_requirements"] = {
            "model_type": "toy",
            "tp_degrees": [1],
            "scenario_ids": ["all_hbm", "input_hbf"],
            "dense_layers": ["qkv_proj"],
            "per_sequence_layers": ["lm_head"],
            "attention_required": True,
            "moe_required": True,
        }
        calibration_digest = "a" * 64
        memory_integration = {"mode": "cli", "parameters": {}}
        identity = {
            "identity_schema": "llmcompass_profile_v2_performance_identity_v1",
            "hardware": {"hardware_id": "test-gpu", "parameters": {}},
            "memory_integration": memory_integration,
            "traffic_resolver": {
                "model_id": "test-resolver",
                "parameters": {},
            },
            "latency_model": {
                "model_id": "test-latency",
                "parameters": {
                    "calibration_digest": calibration_digest,
                },
            },
            "scenario_catalog": meta["scenario_catalog"],
        }
        digest = hashlib.sha256(
            json.dumps(
                identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        meta.update(
            {
                "calibration": {
                    "schema": "llmcompass_hbm_calibration_v1",
                    "digest": calibration_digest,
                    "acceptance_passed": True,
                },
                "performance_basis": {
                    "hbm": "measured_calibrated",
                    "hbf": "parameterized_projection",
                },
                "memory_integration": memory_integration,
                "engine_effective": {
                    "max_num_batched_tokens": 128,
                    "max_num_seqs": 4,
                    "block_size": 16,
                },
                "attention_grid": {
                    "max_kv": 256,
                    "chunk_factor": 2.0,
                    "kv_factor": 2.0,
                },
                "performance_identity": {
                    "algorithm": "sha256",
                    "digest": digest,
                    "manifest": identity,
                },
            }
        )
    if skew_enabled:
        meta["skew_fit"] = {"enabled": True}
    return meta


def _row(filename, scenario, time_us, **shape_overrides):
    audit = {
        "hbm_read_bytes": shape_overrides.pop("hbm_read_bytes", 4096),
        "hbm_write_bytes": shape_overrides.pop("hbm_write_bytes", 1024),
        "hbf_read_bytes": shape_overrides.pop("hbf_read_bytes", 0),
        "hbf_write_bytes": shape_overrides.pop("hbf_write_bytes", 0),
    }
    return {
        **TABLE_SHAPES[filename],
        **shape_overrides,
        "memory_scenario": scenario,
        "time_us": time_us,
        **audit,
    }


def _write_bundle(
    root,
    profile_id,
    rows_by_file,
    *,
    skew_enabled=False,
    fill_missing=True,
):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "meta.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            _meta(profile_id, skew_enabled=skew_enabled),
            f,
            sort_keys=False,
        )

    tp_dir = root / "tp1"
    tp_dir.mkdir()
    if fill_missing:
        rows_by_file = {
            filename: rows_by_file.get(
                filename,
                [
                    _row(filename, scenario, 10)
                    for scenario in ("all_hbm", "input_hbf")
                ],
            )
            for filename in TABLE_SHAPES
        }
    for filename, rows in rows_by_file.items():
        shape_columns = list(TABLE_SHAPES[filename])
        fieldnames = [
            *shape_columns,
            "memory_scenario",
            "time_us",
            *AUDIT_COLUMNS,
        ]
        with (tp_dir / filename).open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _bundle_path(profiler_root, profile_id):
    return (
        Path(profiler_root)
        / "perf"
        / "gpu"
        / "org"
        / "model"
        / "bf16"
        / profile_id
    )


def _load_v2(profiler_root, profile_id):
    with mock.patch.object(
        trace_generator,
        "_load_architecture",
        return_value=TOY_ARCHITECTURE,
    ):
        return trace_generator._load_perf_db(
            "gpu",
            "org/model",
            "bf16",
            {1},
            "toy",
            memory_profile_id=profile_id,
            model_config={"num_experts": 4},
        )


class ProfileV2LookupTest(unittest.TestCase):
    def tearDown(self):
        trace_generator._perf_db_cache.clear()

    def test_same_shape_scenarios_are_isolated_in_all_four_categories(self):
        rows = {
            filename: [
                _row(filename, "all_hbm", 10),
                _row(filename, "input_hbf", 30),
            ]
            for filename in TABLE_SHAPES
        }
        with tempfile.TemporaryDirectory() as tmp:
            profiler_root = Path(tmp) / "profiler"
            _write_bundle(_bundle_path(profiler_root, "cli-a"), "cli-a", rows)
            with mock.patch.object(
                trace_generator,
                "_PROFILER_ROOT_REL",
                str(profiler_root),
            ):
                perf_db = _load_v2(profiler_root, "cli-a")

        tables = perf_db["tables"][1]
        self.assertEqual(
            set(tables["dense"]["qkv_proj"]),
            {"all_hbm", "input_hbf"},
        )
        self.assertEqual(
            set(tables["per_sequence"]["lm_head"]),
            {"all_hbm", "input_hbf"},
        )
        self.assertEqual(
            set(tables["attention"]),
            {"all_hbm", "input_hbf"},
        )
        self.assertEqual(
            set(tables["moe"]),
            {"all_hbm", "input_hbf"},
        )

        lookups = (
            lambda scenario: trace_generator._lookup_dense(
                perf_db,
                "qkv_proj",
                1,
                128,
                memory_scenario=scenario,
            ),
            lambda scenario: trace_generator._lookup_per_sequence(
                perf_db,
                "lm_head",
                1,
                4,
                memory_scenario=scenario,
            ),
            lambda scenario: trace_generator._lookup_attention(
                perf_db,
                1,
                0,
                0,
                1,
                16,
                memory_scenario=scenario,
            ),
            lambda scenario: trace_generator._lookup_moe(
                perf_db,
                128,
                2,
                memory_scenario=scenario,
            ),
        )
        for lookup in lookups:
            self.assertEqual(lookup("all_hbm"), 10000)
            self.assertEqual(lookup("input_hbf"), 30000)

        for layer in ("qkv_proj", "lm_head", "attention", "moe"):
            self.assertTrue(
                trace_generator._layer_available(
                    perf_db,
                    1,
                    layer,
                    memory_scenario="input_hbf",
                )
            )

    def test_sample_lookups_preserve_and_interpolate_four_way_bytes(self):
        rows = {
            "dense.csv": [
                _row(
                    "dense.csv",
                    scenario,
                    time_us,
                    tokens=tokens,
                    hbm_read_bytes=hbm_read,
                    hbf_write_bytes=hbf_write,
                )
                for scenario in ("all_hbm", "input_hbf")
                for tokens, time_us, hbm_read, hbf_write in (
                    (128, 10, 4096, 1024),
                    (256, 20, 8192, 2048),
                )
            ],
            "per_sequence.csv": [
                _row("per_sequence.csv", scenario, 10)
                for scenario in ("all_hbm", "input_hbf")
            ],
            "attention.csv": [
                _row(
                    "attention.csv",
                    scenario,
                    10,
                    hbm_write_bytes=2048,
                    hbf_read_bytes=8192,
                )
                for scenario in ("all_hbm", "input_hbf")
            ],
            "moe.csv": [
                _row("moe.csv", scenario, 10)
                for scenario in ("all_hbm", "input_hbf")
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            profiler_root = Path(tmp) / "profiler"
            profile_id = "sample-audit"
            _write_bundle(
                _bundle_path(profiler_root, profile_id),
                profile_id,
                rows,
            )
            with mock.patch.object(
                trace_generator,
                "_PROFILER_ROOT_REL",
                str(profiler_root),
            ):
                perf_db = _load_v2(profiler_root, profile_id)

        dense = trace_generator._lookup_dense_sample(
            perf_db,
            "qkv_proj",
            1,
            192,
            memory_scenario="input_hbf",
        )
        self.assertIsInstance(dense, trace_generator.ProfileLatencySample)
        self.assertEqual(dense.latency_ns, 15_000)
        self.assertEqual(dense.hbm_read_bytes, 6144)
        self.assertEqual(dense.hbf_write_bytes, 1536)
        self.assertEqual(
            trace_generator._lookup_dense(
                perf_db,
                "qkv_proj",
                1,
                192,
                memory_scenario="input_hbf",
            ),
            dense.latency_ns,
        )

        attention = trace_generator._lookup_attention_sample(
            perf_db,
            1,
            0,
            0,
            1,
            16,
            memory_scenario="input_hbf",
        )
        self.assertEqual(attention.latency_ns, 10_000)
        self.assertEqual(
            attention.four_way_bytes,
            {
                "hbm_read_bytes": 4096,
                "hbm_write_bytes": 2048,
                "hbf_read_bytes": 8192,
                "hbf_write_bytes": 0,
            },
        )
        per_sequence = trace_generator._lookup_per_sequence_sample(
            perf_db,
            "lm_head",
            1,
            4,
            memory_scenario="input_hbf",
        )
        moe = trace_generator._lookup_moe_sample(
            perf_db,
            128,
            2,
            memory_scenario="input_hbf",
        )
        self.assertEqual(per_sequence.hbm_read_bytes, 4096)
        self.assertEqual(moe.hbm_write_bytes, 1024)

    def test_runtime_engine_and_attention_bounds_are_hard_gates(self):
        rows = {
            filename: [
                _row(filename, scenario, 10)
                for scenario in ("all_hbm", "input_hbf")
            ]
            for filename in TABLE_SHAPES
        }
        with tempfile.TemporaryDirectory() as tmp:
            profiler_root = Path(tmp) / "profiler"
            profile_id = "runtime-bounds"
            _write_bundle(
                _bundle_path(profiler_root, profile_id),
                profile_id,
                rows,
            )
            with (
                mock.patch.object(
                    trace_generator,
                    "_PROFILER_ROOT_REL",
                    str(profiler_root),
                ),
                mock.patch.object(
                    trace_generator,
                    "_load_architecture",
                    return_value=TOY_ARCHITECTURE,
                ),
            ):
                perf_db = trace_generator._load_perf_db(
                    "gpu",
                    "org/model",
                    "bf16",
                    {1},
                    "toy",
                    memory_profile_id=profile_id,
                    model_config={"num_experts": 4},
                    runtime_max_num_batched_tokens=128,
                    runtime_max_num_seqs=4,
                    runtime_block_size=16,
                )
                with self.assertRaisesRegex(
                    ProfileV2RuntimeNotReadyError,
                    "block_size",
                ):
                    trace_generator._load_perf_db(
                        "gpu",
                        "org/model",
                        "bf16",
                        {1},
                        "toy",
                        memory_profile_id=profile_id,
                        model_config={"num_experts": 4},
                        runtime_block_size=32,
                    )
                with self.assertRaisesRegex(
                    ProfileV2RuntimeNotReadyError,
                    "max_num_batched_tokens",
                ):
                    trace_generator._load_perf_db(
                        "gpu",
                        "org/model",
                        "bf16",
                        {1},
                        "toy",
                        memory_profile_id=profile_id,
                        model_config={"num_experts": 4},
                        runtime_max_num_batched_tokens=129,
                    )

        with self.assertRaisesRegex(
            ProfileV2RuntimeNotReadyError,
            "kv_decode",
        ):
            trace_generator._lookup_attention(
                perf_db,
                1,
                0,
                0,
                1,
                257,
                memory_scenario="all_hbm",
            )

    def test_missing_scenario_is_rejected_before_runtime_lookup(self):
        rows = {
            filename: [_row(filename, "all_hbm", 10)]
            for filename in TABLE_SHAPES
        }
        with tempfile.TemporaryDirectory() as tmp:
            profiler_root = Path(tmp) / "profiler"
            _write_bundle(_bundle_path(profiler_root, "cli-a"), "cli-a", rows)
            with mock.patch.object(
                trace_generator,
                "_PROFILER_ROOT_REL",
                str(profiler_root),
            ):
                with self.assertRaisesRegex(
                    ProfileV2RuntimeNotReadyError,
                    "scenario='input_hbf'",
                ):
                    _load_v2(profiler_root, "cli-a")

    def test_unknown_and_implicit_scenarios_are_hard_errors(self):
        rows = {
            filename: [
                _row(filename, "all_hbm", 10),
                _row(filename, "input_hbf", 30),
            ]
            for filename in TABLE_SHAPES
        }
        with tempfile.TemporaryDirectory() as tmp:
            profiler_root = Path(tmp) / "profiler"
            _write_bundle(_bundle_path(profiler_root, "cli-a"), "cli-a", rows)
            with mock.patch.object(
                trace_generator,
                "_PROFILER_ROOT_REL",
                str(profiler_root),
            ):
                perf_db = _load_v2(profiler_root, "cli-a")

        lookups = (
            lambda **kwargs: trace_generator._lookup_dense(
                perf_db, "qkv_proj", 1, 128, **kwargs
            ),
            lambda **kwargs: trace_generator._lookup_per_sequence(
                perf_db, "lm_head", 1, 4, **kwargs
            ),
            lambda **kwargs: trace_generator._lookup_attention(
                perf_db, 1, 0, 0, 1, 16, **kwargs
            ),
            lambda **kwargs: trace_generator._lookup_moe(
                perf_db, 128, 2, **kwargs
            ),
        )
        for lookup in lookups:
            with self.assertRaisesRegex(KeyError, "显式提供"):
                lookup()
            with self.assertRaisesRegex(KeyError, "未知 memory_scenario"):
                lookup(memory_scenario="not_declared")

        self.assertTrue(
            trace_generator._layer_available(
                perf_db, 1, "qkv_proj", memory_scenario="input_hbf",
            )
        )
        with self.assertRaisesRegex(KeyError, "显式提供"):
            trace_generator._layer_available(perf_db, 1, "qkv_proj")

    def test_memory_profile_ids_use_independent_cache_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiler_root = Path(tmp) / "profiler"
            for profile_id, time_us in (("cli-a", 10), ("csi-b", 40)):
                _write_bundle(
                    _bundle_path(profiler_root, profile_id),
                    profile_id,
                    {
                        "dense.csv": [
                            _row("dense.csv", "all_hbm", time_us),
                            _row("dense.csv", "input_hbf", time_us),
                        ],
                    },
                )

            with mock.patch.object(
                trace_generator,
                "_PROFILER_ROOT_REL",
                str(profiler_root),
            ):
                cli_db = _load_v2(profiler_root, "cli-a")
                csi_db = _load_v2(profiler_root, "csi-b")

        self.assertIsNot(cli_db, csi_db)
        self.assertIn(
            ("gpu", "org/model", "bf16", "cli-a"),
            trace_generator._perf_db_cache,
        )
        self.assertIn(
            ("gpu", "org/model", "bf16", "csi-b"),
            trace_generator._perf_db_cache,
        )
        self.assertEqual(
            trace_generator._lookup_dense(
                cli_db,
                "qkv_proj",
                1,
                128,
                memory_scenario="all_hbm",
            ),
            10000,
        )
        self.assertEqual(
            trace_generator._lookup_dense(
                csi_db,
                "qkv_proj",
                1,
                128,
                memory_scenario="all_hbm",
            ),
            40000,
        )
        trace_generator.warn_if_runtime_exceeds_profiled(cli_db, None, None)
        trace_generator.warn_if_runtime_exceeds_profiled(csi_db, None, None)
        self.assertIn(
            ("warned", "gpu", "org/model", "bf16", "cli-a"),
            trace_generator._perf_db_cache,
        )
        self.assertIn(
            ("warned", "gpu", "org/model", "bf16", "csi-b"),
            trace_generator._perf_db_cache,
        )
        with self.assertRaisesRegex(FileNotFoundError, "bf16/cli-a/"):
            trace_generator._check_tp_coverage(
                cli_db,
                {2},
                "gpu",
                "org/model",
                "bf16",
            )
        self.assertTrue(
            trace_generator._layer_available(
                cli_db,
                1,
                "lm_head",
                memory_scenario="all_hbm",
            )
        )

        with (
            mock.patch.object(
                trace_generator,
                "_load_architecture",
                return_value=TOY_ARCHITECTURE,
            ),
        ):
            with self.assertRaisesRegex(FileNotFoundError, "tp=\\[2\\]"):
                trace_generator._load_perf_db(
                    "gpu",
                    "org/model",
                    "bf16",
                    {2},
                    "toy",
                    memory_profile_id="cli-a",
                    model_config={"num_experts": 4},
                )

    def test_builder_uses_contract_canonical_scenario_and_layer_text(self):
        rows = [
            _row("dense.csv", " all_hbm ", 10),
            _row("dense.csv", "input_hbf", 30),
        ]
        rows[0]["layer"] = " qkv_proj "
        with tempfile.TemporaryDirectory() as tmp:
            profiler_root = Path(tmp) / "profiler"
            _write_bundle(
                _bundle_path(profiler_root, "cli-a"),
                "cli-a",
                {"dense.csv": rows},
            )
            with mock.patch.object(
                trace_generator,
                "_PROFILER_ROOT_REL",
                str(profiler_root),
            ):
                perf_db = _load_v2(profiler_root, "cli-a")

        self.assertEqual(
            trace_generator._lookup_dense(
                perf_db,
                "qkv_proj",
                1,
                128,
                memory_scenario="all_hbm",
            ),
            10000,
        )

    def test_disabled_v2_skew_does_not_apply_legacy_fallback(self):
        attention_rows = [
            _row(
                "attention.csv",
                "all_hbm",
                10,
                n_decode=2,
                kv_decode=10,
            ),
            _row(
                "attention.csv",
                "all_hbm",
                30,
                n_decode=2,
                kv_decode=20,
            ),
            _row(
                "attention.csv",
                "input_hbf",
                10,
                n_decode=2,
                kv_decode=10,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            profiler_root = Path(tmp) / "profiler"
            _write_bundle(
                _bundle_path(profiler_root, "cli-a"),
                "cli-a",
                {"attention.csv": attention_rows},
            )
            with mock.patch.object(
                trace_generator,
                "_PROFILER_ROOT_REL",
                str(profiler_root),
            ):
                perf_db = _load_v2(profiler_root, "cli-a")

        latency = trace_generator._lookup_attention_with_skew(
            perf_db,
            1,
            0,
            0,
            2,
            10,
            20,
            0,
            memory_scenario="all_hbm",
        )
        self.assertEqual(latency, 10000)

    def test_enabled_v2_skew_without_scenario_dimension_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiler_root = Path(tmp) / "profiler"
            _write_bundle(
                _bundle_path(profiler_root, "cli-a"),
                "cli-a",
                {
                    "attention.csv": [
                        _row("attention.csv", "all_hbm", 10),
                    ],
                },
                skew_enabled=True,
            )
            with mock.patch.object(
                trace_generator,
                "_PROFILER_ROOT_REL",
                str(profiler_root),
            ):
                with self.assertRaisesRegex(
                    ProfileV2RuntimeNotReadyError,
                    "memory_scenario",
                ):
                    _load_v2(profiler_root, "cli-a")

    def test_real_v1_profiles_keep_original_lookup_values(self):
        profiler_root = (
            Path(__file__).resolve().parents[1]
            / "profiler"
        )
        with mock.patch.object(
            trace_generator,
            "_PROFILER_ROOT_REL",
            str(profiler_root),
        ):
            dense_db = trace_generator._load_perf_db(
                "RTXPRO6000",
                "meta-llama/Llama-3.1-8B",
                "bf16",
                {1},
                "llama",
            )
            moe_db = trace_generator._load_perf_db(
                "RTXPRO6000",
                "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "bf16",
                {1},
                "qwen3_moe",
            )

        self.assertEqual(
            trace_generator._lookup_dense(dense_db, "act_fn", 1, 1),
            2677,
        )
        self.assertEqual(
            trace_generator._lookup_per_sequence(dense_db, "lm_head", 1, 1),
            714006,
        )
        self.assertEqual(
            trace_generator._lookup_attention(dense_db, 1, 0, 0, 1, 16),
            8352,
        )
        self.assertEqual(
            trace_generator._lookup_moe(moe_db, 1, 8),
            50230,
        )
        self.assertEqual(
            trace_generator._skew_alpha({}, 1, 0, 2, 0.5, 16, 0),
            trace_generator._ATTN_SKEW_ALPHA_FALLBACK,
        )


if __name__ == "__main__":
    unittest.main()
