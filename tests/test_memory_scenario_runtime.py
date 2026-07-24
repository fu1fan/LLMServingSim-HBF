import importlib
import inspect
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serving.core import trace_generator
from serving.core.memory_scenario import (
    MemoryScenarioCompatibilityError,
    parse_instance_performance_profile,
)


def _profile(*, default="all_hbm", layers=None, blocks=None):
    return {
        "performance_profile": {
            "mode": "memory_scenario_v2",
            "memory_profile_id": "cli-a",
            "scenario_policy": {
                "default": default,
                "layers": layers or {},
                "blocks": blocks or [],
            },
        },
    }


def _policy(*, default="all_hbm", layers=None, blocks=None):
    return parse_instance_performance_profile(
        _profile(default=default, layers=layers, blocks=blocks),
        2,
    )


def _placement(*, weights="LOCAL", kv_loc="LOCAL"):
    return {
        "default": {
            "weights": weights,
            "kv_loc": kv_loc,
            "kv_evict_loc": "REMOTE:0",
        },
        "block": [],
        "layer": {},
    }


def _model_config():
    return {
        "model_type": "toy",
        "num_hidden_layers": 2,
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "torch_dtype": "bfloat16",
    }


def _perf_db(*, version=2):
    return {
        "meta": {},
        "profile_schema_version": version,
        "memory_profile_id": "cli-a" if version == 2 else None,
        "scenario_catalog": {
            "all_hbm": {},
            "layer_hbf": {},
            "block_hbf": {},
        } if version == 2 else {},
        "architecture": {
            "sequence": {
                "prologue": [],
                "pre_attn": ["qkv_proj"],
                "post_attn": [],
                "mlp_dense": [],
                "mlp_moe": [],
                "head": [],
            },
            "catalog": {
                "dense": {"qkv_proj": {}},
                "per_sequence": {"lm_head": {}},
                "attention": {"attention": {}},
                "moe": {"moe": {}},
            },
        },
        "variant": "bf16",
        "hardware": "gpu",
        "model": "toy/model",
        "available_tps": [1],
        "tables": {1: {}},
    }


def _build_ctx(policy, perf_db):
    with (
        mock.patch.object(
            trace_generator,
            "_load_perf_db",
            return_value=perf_db,
        ) as loader,
        mock.patch.object(
            trace_generator,
            "warn_if_runtime_exceeds_profiled",
        ),
    ):
        ctx = trace_generator._build_trace_ctx(
            "gpu",
            "toy/model",
            _model_config(),
            1,
            1,
            1,
            1,
            0,
            2,
            _placement(),
            None,
            False,
            None,
            None,
            None,
            variant="bf16",
            memory_scenario_policy=policy,
        )
    return ctx, loader


def _import_serving_main():
    # pyinstrument 只用于 CLI 性能分析，测试配置解析时无需安装。
    if "pyinstrument" not in sys.modules:
        module = types.ModuleType("pyinstrument")
        module.Profiler = object
        sys.modules["pyinstrument"] = module
    return importlib.import_module("serving.__main__")


class MemoryScenarioRuntimeTest(unittest.TestCase):
    def test_context_loads_v2_bundle_and_legacy_keeps_loader_call(self):
        policy = _policy()
        ctx, loader = _build_ctx(policy, _perf_db())

        self.assertIs(ctx.memory_scenario_policy, policy)
        loader.assert_called_once_with(
            "gpu",
            "toy/model",
            "bf16",
            {1},
            "toy",
            memory_profile_id="cli-a",
            model_config=_model_config(),
        )

        legacy = parse_instance_performance_profile({}, 2)
        _, legacy_loader = _build_ctx(legacy, _perf_db(version=1))
        legacy_loader.assert_called_once_with(
            "gpu",
            "toy/model",
            "bf16",
            {1},
            "toy",
        )

    def test_context_rejects_unknown_scenario_and_layer_before_trace(self):
        unknown_scenario = _policy(default="not_declared")
        with self.assertRaisesRegex(KeyError, "未知 memory_scenario"):
            _build_ctx(unknown_scenario, _perf_db())

        unknown_layer = _policy(layers={"qkv_typo": "all_hbm"})
        with self.assertRaisesRegex(KeyError, "architecture catalog"):
            _build_ctx(unknown_layer, _perf_db())

    def test_all_runtime_lookup_sites_receive_the_selected_scenario(self):
        policy = _policy(
            layers={"lm_head": "layer_hbf"},
            blocks=[{"blocks": "1", "scenario": "block_hbf"}],
        )
        perf_db = _perf_db()
        ctx = SimpleNamespace(
            memory_scenario_policy=policy,
            perf_db=perf_db,
            tp_size=1,
            model="toy/model",
            fp=2,
            placement=_placement(),
            hardware="gpu",
            ep_total=1,
            local_ep=1,
            dp_sum_total_len=0,
            ep_dim=None,
            config={"hidden_size": 16, "num_experts": 4},
            gate=SimpleNamespace(
                route_ep=lambda *_: SimpleNamespace(
                    local_tokens=[8],
                    activated_experts=[2],
                )
            ),
        )
        bctx = SimpleNamespace(
            total_len=8,
            lm_head_len=2,
            prefill_chunk=8,
            kv_prefill=0,
            n_decode=0,
            kv_decode_mean=0,
            kv_decode_max=0,
            kv_decode_min=0,
        )

        with (
            mock.patch.object(
                trace_generator,
                "_lookup_dense",
                return_value=10,
            ) as dense,
            mock.patch.object(
                trace_generator,
                "_lookup_per_sequence",
                return_value=11,
            ) as per_sequence,
            mock.patch.object(
                trace_generator,
                "_lookup_attention_with_skew",
                return_value=12,
            ) as attention,
            mock.patch.object(
                trace_generator,
                "_lookup_moe",
                return_value=13,
            ) as moe,
            mock.patch.object(
                trace_generator,
                "calculate_sizes",
                return_value=(1, 2, 3),
            ),
            mock.patch.object(
                trace_generator,
                "get_device",
                return_value="LOCAL",
            ),
            mock.patch.object(
                trace_generator,
                "formatter",
                return_value="layer\n",
            ),
        ):
            trace_generator._emit_layer(
                ctx, bctx, "qkv_proj", [], None, layer_num=1,
            )
            trace_generator._emit_layer(
                ctx, bctx, "lm_head", [], None, layer_num=1,
            )
            trace_generator._emit_layer(
                ctx, bctx, "attention", [], None, layer_num=1,
            )
            trace_generator._emit_moe_block(
                ctx, bctx, [], None, 1, "0",
            )
            trace_generator._layer_latency_for_power(
                ctx, bctx, "qkv_proj", layer_num=1,
            )

        self.assertTrue(
            all(
                call.kwargs["memory_scenario"] == "block_hbf"
                for call in dense.call_args_list
            )
        )
        self.assertEqual(
            per_sequence.call_args.kwargs["memory_scenario"],
            "layer_hbf",
        )
        self.assertEqual(
            attention.call_args.kwargs["memory_scenario"],
            "block_hbf",
        )
        self.assertEqual(
            moe.call_args.kwargs["memory_scenario"],
            "block_hbf",
        )

        with (
            mock.patch.object(
                trace_generator,
                "_layer_available",
                return_value=False,
            ) as available,
            mock.patch.object(trace_generator.logger, "warning"),
        ):
            trace_generator._emit_sequence(
                ctx, bctx, 1, ["qkv_proj"], [], None, "NONE",
            )
        self.assertEqual(
            available.call_args.kwargs["memory_scenario"],
            "block_hbf",
        )

    def test_block_override_forces_per_block_toy_trace(self):
        policy = _policy(
            blocks=[{"blocks": "1", "scenario": "block_hbf"}],
        )
        perf_db = _perf_db()
        batch = SimpleNamespace(
            model="toy/model",
            batch_id=7,
            load=0,
            evict=0,
            requests=[SimpleNamespace(id=1)],
            total_len=1,
            num_prefill=1,
            num_decode=0,
            prefill_q_list=[1],
            prefill_k_list=[0],
            decode_k_list=[],
        )

        def lookup(_db, _name, _tp, _tokens, *, memory_scenario):
            return {
                "all_hbm": 111,
                "block_hbf": 222,
            }[memory_scenario]

        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.object(
                    trace_generator,
                    "get_config",
                    return_value=_model_config(),
                ),
                mock.patch.object(
                    trace_generator,
                    "_load_perf_db",
                    return_value=perf_db,
                ) as loader,
                mock.patch.object(
                    trace_generator,
                    "warn_if_runtime_exceeds_profiled",
                ),
                mock.patch.object(
                    trace_generator,
                    "_layer_available",
                    return_value=True,
                ),
                mock.patch.object(
                    trace_generator,
                    "_lookup_dense",
                    side_effect=lookup,
                ),
                mock.patch.object(
                    trace_generator,
                    "calculate_sizes",
                    return_value=(1, 2, 3),
                ),
            ):
                trace_generator.generate_trace(
                    batch,
                    "gpu",
                    1,
                    1,
                    1,
                    1,
                    placement=_placement(),
                    block_mode_on=False,
                    dtype="bfloat16",
                    inputs_root=tmp,
                    memory_scenario_policy=policy,
                )

            output = (
                Path(tmp)
                / "trace"
                / "gpu"
                / "toy"
                / "model"
                / "instance0_batch7.txt"
            )
            qkv_times = [
                int(line.split()[1])
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.startswith("qkv_proj_")
            ]

        self.assertEqual(qkv_times, [111, 222])
        loader.assert_called_once_with(
            "gpu",
            "toy/model",
            "bf16",
            {1},
            "toy",
            memory_profile_id="cli-a",
            model_config=_model_config(),
        )

    def test_layer_only_policy_preserves_block_copy(self):
        policy = _policy(layers={"qkv_proj": "layer_hbf"})
        self.assertFalse(
            trace_generator._effective_block_mode(False, policy)
        )
        self.assertTrue(
            trace_generator._effective_block_mode(True, policy)
        )

    def test_main_runtime_gate_uses_final_flags_and_all_trace_calls_propagate(self):
        serving_main = _import_serving_main()
        args = SimpleNamespace(
            dtype="bfloat16",
            kv_cache_dtype="auto",
            enable_attn_offloading=False,
            enable_sub_batch_interleaving=False,
            enable_local_offloading=False,
            max_num_seqs=16,
            max_num_batched_tokens=128,
            long_prefill_token_threshold=0,
            block_size=16,
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
            prioritize_prefill=False,
            enable_block_copy=True,
        )
        instance = {
            "model_name": "toy/model",
            **_profile(),
        }
        with mock.patch.object(
            serving_main,
            "get_config",
            return_value=_model_config(),
        ):
            runtime = serving_main._build_instance_runtime_configs(
                [instance],
                args,
                {"bfloat16": 16},
                placements=[_placement()],
            )
            self.assertTrue(runtime[0]["memory_scenario_policy"].is_v2)

            with self.assertRaises(MemoryScenarioCompatibilityError):
                serving_main._build_instance_runtime_configs(
                    [instance],
                    args,
                    {"bfloat16": 16},
                    placements=[_placement(weights="REMOTE:0")],
                )

        source = inspect.getsource(serving_main.main)
        self.assertEqual(
            source.count(
                'memory_scenario_policy=inst_cfg["memory_scenario_policy"]'
            ),
            3,
        )


if __name__ == "__main__":
    unittest.main()
