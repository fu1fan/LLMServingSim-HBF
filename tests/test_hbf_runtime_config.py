import unittest
import sys
import types
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

pyinstrument = types.ModuleType("pyinstrument")
pyinstrument.Profiler = object
sys.modules.setdefault("pyinstrument", pyinstrument)

import serving.__main__ as serving_main


def _args():
    return SimpleNamespace(
        dtype=None,
        kv_cache_dtype="auto",
        enable_attn_offloading=False,
        enable_sub_batch_interleaving=False,
        enable_local_offloading=False,
        max_num_seqs=128,
        max_num_batched_tokens=2048,
        long_prefill_token_threshold=0,
        block_size=16,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        prioritize_prefill=False,
        enable_block_copy=True,
    )


def _placement():
    return {
        "default": {"weights": "LOCAL", "kv_loc": "LOCAL"},
        "block": [],
        "layer": {},
    }


def _instance(*, hbf=False):
    value = {"model_name": "test-model", "block_size": 32}
    if hbf:
        value.update(
            {
                "hbf_mem": {"mem_size": 64},
                "performance_profile": {
                    "mode": "memory_scenario_v2",
                    "memory_profile_id": "hbf-four-way",
                    "scenario_selection": "residency_derived",
                },
                "memory_tiering": {
                    "weights": {
                        "policy": "static_map",
                        "default_tier": "hbf",
                    }
                },
            }
        )
    return value


class HbfRuntimeConfigTest(unittest.TestCase):
    def test_hbf_memory_validator_is_explicitly_imported(self):
        self.assertTrue(callable(serving_main.validate_memory_config))

    def test_shutdown_checks_every_instance_memory(self):
        memories = [
            SimpleNamespace(
                free_prefix_cache=Mock(),
                free_weight=Mock(),
                is_free=Mock(return_value=True),
            )
            for _ in range(2)
        ]

        serving_main._release_instance_memory(
            [SimpleNamespace(memory=item) for item in memories]
        )

        for memory in memories:
            memory.free_prefix_cache.assert_called_once_with()
            memory.free_weight.assert_called_once_with()
            memory.is_free.assert_called_once_with()

        memories[0].is_free.return_value = False
        with self.assertRaisesRegex(RuntimeError, r"\[0\]"):
            serving_main._release_instance_memory(
                [SimpleNamespace(memory=item) for item in memories]
            )

    def test_tiering_stats_collection_skips_ordinary_instances(self):
        snapshot = SimpleNamespace(
            to_dict=lambda: {
                "schema": "llmservingsim_memory_tiering_stats_v1"
            }
        )
        schedulers = [
            SimpleNamespace(
                memory=SimpleNamespace(
                    tiering_stats_snapshot=lambda: None
                )
            ),
            SimpleNamespace(
                memory=SimpleNamespace(
                    tiering_stats_snapshot=lambda: snapshot
                )
            ),
        ]

        payload = serving_main._collect_memory_tiering_stats(schedulers)

        self.assertEqual(len(payload["instances"]), 1)
        self.assertEqual(payload["instances"][0]["instance_id"], 1)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "stats.json"
            serving_main._write_memory_tiering_stats(path, payload)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                payload,
            )

    @patch.object(
        serving_main,
        "get_config",
        return_value={
            "torch_dtype": "bfloat16",
            "num_hidden_layers": 32,
        },
    )
    def test_runtime_config_carries_verified_tiering_contract(self, _):
        configs = serving_main._build_instance_runtime_configs(
            [_instance(hbf=True)],
            _args(),
            {"bfloat16": 16},
            placements=[_placement()],
        )

        self.assertEqual(configs[0]["block_size"], 32)
        self.assertTrue(configs[0]["memory_tiering"].enabled)
        self.assertTrue(
            configs[0]["memory_scenario_policy"].is_residency_derived
        )

    @patch.object(
        serving_main,
        "get_config",
        return_value={
            "torch_dtype": "bfloat16",
            "num_hidden_layers": 32,
        },
    )
    def test_runtime_rejects_mixed_gpu_families_before_scheduling(self, _):
        with self.assertRaisesRegex(ValueError, "不能混用"):
            serving_main._build_instance_runtime_configs(
                [_instance(hbf=True), _instance(hbf=False)],
                _args(),
                {"bfloat16": 16},
                placements=[_placement(), _placement()],
            )

    @patch.object(
        serving_main,
        "get_config",
        return_value={
            "torch_dtype": "bfloat16",
            "num_hidden_layers": 32,
        },
    )
    def test_runtime_rejects_policy_interfaces_not_in_main_loop(self, _):
        cases = (
            (
                {"weights": {"policy": "hbf_backed_hbm_cache"}},
                "自适应 HBF 权重",
            ),
            (
                {"kv": {"policy": "watermark_lru"}},
                "watermark_lru",
            ),
            (
                {"transfer": {"capacity_fallback": "cpu"}},
                "capacity_fallback=reject",
            ),
            (
                {
                    "communication_buffers": {
                        "tier": "hbf",
                        "allow_hbf_staging": True,
                    }
                },
                "communication buffers",
            ),
        )
        for tiering, message in cases:
            instance = _instance(hbf=True)
            instance["memory_tiering"] = tiering
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    serving_main._build_instance_runtime_configs(
                        [instance],
                        _args(),
                        {"bfloat16": 16},
                        placements=[_placement()],
                    )

    @patch.object(
        serving_main,
        "get_config",
        return_value={
            "torch_dtype": "bfloat16",
            "num_hidden_layers": 32,
        },
    )
    def test_runtime_rejects_unmodeled_pd_and_uneven_prefix_paths(self, _):
        pd_instance = _instance(hbf=True)
        pd_instance["pd_type"] = "decode"
        pd_instance["memory_tiering"]["kv"] = {"policy": "hbf_only"}
        with self.assertRaisesRegex(RuntimeError, "tier-aware"):
            serving_main._build_instance_runtime_configs(
                [pd_instance],
                _args(),
                {"bfloat16": 16},
                placements=[_placement()],
            )

        prefix_instance = _instance(hbf=True)
        prefix_instance["pp_size"] = 3
        with self.assertRaisesRegex(RuntimeError, "能整除"):
            serving_main._build_instance_runtime_configs(
                [prefix_instance],
                _args(),
                {"bfloat16": 16},
                placements=[_placement()],
            )


if __name__ == "__main__":
    unittest.main()
