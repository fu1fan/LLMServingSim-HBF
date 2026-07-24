import copy
import unittest

from serving.core.memory_scenario import (
    LEGACY_HBM_ONLY,
    MEMORY_SCENARIO_V2,
    MemoryScenarioCompatibilityError,
    MemoryScenarioConfigError,
    parse_instance_performance_profile,
    validate_memory_scenario_compatibility,
)


def _v2_instance(*, profile_id="cli-a", policy=None, **extra_profile_fields):
    if policy is None:
        policy = {"default": "all_hbm"}
    profile = {
        "mode": MEMORY_SCENARIO_V2,
        "memory_profile_id": profile_id,
        "scenario_policy": policy,
        **extra_profile_fields,
    }
    return {"performance_profile": profile}


def _local_placement():
    return {
        "default": {
            "weights": "LOCAL",
            "kv_loc": "LOCAL",
            "kv_evict_loc": "REMOTE:0",
        },
        "block": [
            {
                "weights": "LOCAL:0",
                "kv_loc": "LOCAL",
                "kv_evict_loc": "CXL:1",
            },
        ],
        "layer": {
            "qkv_proj": {
                "weights": "LOCAL",
                "kv_loc": "LOCAL:0",
                "kv_evict_loc": "REMOTE:1",
            },
        },
    }


_DEFAULT_PLACEMENT = object()


def _validate_gate(
    policy,
    placement=_DEFAULT_PLACEMENT,
    **flag_overrides,
):
    flags = {
        "enable_local_offloading": False,
        "enable_attn_offloading": False,
        "enable_sub_batch_interleaving": False,
    }
    flags.update(flag_overrides)
    return validate_memory_scenario_compatibility(
        policy,
        placement=(
            _local_placement()
            if placement is _DEFAULT_PLACEMENT
            else placement
        ),
        **flags,
    )


class MemoryScenarioPolicyTest(unittest.TestCase):
    def test_missing_or_explicit_legacy_profile_keeps_compatibility_mode(self):
        missing = parse_instance_performance_profile({}, 32)
        explicit = parse_instance_performance_profile(
            {"performance_profile": {"mode": LEGACY_HBM_ONLY}},
            32,
        )

        for policy in (missing, explicit):
            self.assertFalse(policy.is_v2)
            self.assertEqual(policy.mode, LEGACY_HBM_ONLY)
            self.assertIsNone(policy.memory_profile_id)
            self.assertIsNone(policy.scenario_for("qkv_proj", 0))
            self.assertFalse(policy.requires_per_block_trace)

    def test_v2_priority_is_layer_then_block_then_default(self):
        policy = parse_instance_performance_profile(
            _v2_instance(
                policy={
                    "default": "all_hbm",
                    "layers": {
                        "qkv_proj": "qkv_hbf",
                        "embedding": "embedding_hbf",
                    },
                    "blocks": [
                        {"blocks": "0-2,4", "scenario": "early_hbf"},
                        {"blocks": "6-5", "scenario": "late_hbf"},
                    ],
                }
            ),
            8,
        )

        self.assertTrue(policy.is_v2)
        self.assertEqual(policy.memory_profile_id, "cli-a")
        self.assertTrue(policy.requires_per_block_trace)
        self.assertEqual(policy.scenario_for("qkv_proj", 1), "qkv_hbf")
        self.assertEqual(policy.scenario_for("attention", 1), "early_hbf")
        self.assertEqual(policy.scenario_for("attention", 5), "late_hbf")
        self.assertEqual(policy.scenario_for("attention", 3), "all_hbm")
        self.assertEqual(policy.scenario_for("embedding", None), "embedding_hbf")
        self.assertEqual(policy.scenario_for("lm_head", None), "all_hbm")

    def test_layers_without_blocks_do_not_require_per_block_trace(self):
        policy = parse_instance_performance_profile(
            _v2_instance(
                policy={
                    "default": "all_hbm",
                    "layers": {"attention": "attention_hbf"},
                    "blocks": [],
                }
            ),
            4,
        )

        self.assertFalse(policy.requires_per_block_trace)
        self.assertEqual(policy.scenario_for("attention", 2), "attention_hbf")

    def test_v2_requires_mode_profile_id_and_policy_default(self):
        cases = {
            "null profile": {"performance_profile": None},
            "missing mode": {"performance_profile": {}},
            "unknown mode": {"performance_profile": {"mode": "automatic"}},
            "missing id": {
                "performance_profile": {
                    "mode": MEMORY_SCENARIO_V2,
                    "scenario_policy": {"default": "all_hbm"},
                }
            },
            "invalid id": _v2_instance(profile_id="../cli-a"),
            "missing policy": {
                "performance_profile": {
                    "mode": MEMORY_SCENARIO_V2,
                    "memory_profile_id": "cli-a",
                }
            },
            "missing default": _v2_instance(policy={"layers": {}}),
            "empty default": _v2_instance(policy={"default": ""}),
        }
        for name, instance in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(MemoryScenarioConfigError):
                    parse_instance_performance_profile(instance, 8)

    def test_unknown_fields_are_rejected_at_each_schema_level(self):
        cases = {
            "legacy profile": {
                "performance_profile": {
                    "mode": LEGACY_HBM_ONLY,
                    "memory_profile_id": "unused",
                }
            },
            "v2 profile": _v2_instance(unexpected=True),
            "policy": _v2_instance(
                policy={"default": "all_hbm", "fallback": "all_hbm"}
            ),
            "block rule": _v2_instance(
                policy={
                    "default": "all_hbm",
                    "blocks": [
                        {
                            "blocks": "0",
                            "scenario": "hbf",
                            "priority": 1,
                        }
                    ],
                }
            ),
        }
        for name, instance in cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    MemoryScenarioConfigError,
                    "未知字段",
                ):
                    parse_instance_performance_profile(instance, 8)

    def test_layer_overrides_require_exact_canonical_names_and_scenarios(self):
        invalid_layers = {
            "wildcard": {"qkv_*": "hbf"},
            "uppercase": {"QKV": "hbf"},
            "leading space": {" qkv_proj": "hbf"},
            "empty name": {"": "hbf"},
            "empty scenario": {"qkv_proj": ""},
            "invalid scenario": {"qkv_proj": "../hbf"},
        }
        for name, layers in invalid_layers.items():
            with self.subTest(name=name):
                with self.assertRaises(MemoryScenarioConfigError):
                    parse_instance_performance_profile(
                        _v2_instance(
                            policy={
                                "default": "all_hbm",
                                "layers": layers,
                            }
                        ),
                        8,
                    )

        with self.assertRaisesRegex(MemoryScenarioConfigError, "必须是 mapping"):
            parse_instance_performance_profile(
                _v2_instance(
                    policy={"default": "all_hbm", "layers": []}
                ),
                8,
            )

    def test_block_expressions_are_strict_and_bounds_checked(self):
        invalid_expressions = (
            3,
            "",
            "0,,1",
            "bad",
            "-1",
            "0-4",
            "1,1",
            "0-2,2-3",
        )
        for expression in invalid_expressions:
            with self.subTest(expression=expression):
                with self.assertRaises(MemoryScenarioConfigError):
                    parse_instance_performance_profile(
                        _v2_instance(
                            policy={
                                "default": "all_hbm",
                                "blocks": [
                                    {
                                        "blocks": expression,
                                        "scenario": "hbf",
                                    }
                                ],
                            }
                        ),
                        4,
                    )

    def test_block_rules_reject_missing_fields_and_overlap(self):
        cases = (
            [{"blocks": "0"}],
            [{"scenario": "hbf"}],
            [{"blocks": "0", "scenario": ""}],
            ["0-1"],
            [
                {"blocks": "0-2", "scenario": "hbf_a"},
                {"blocks": "2-3", "scenario": "hbf_b"},
            ],
        )
        for rules in cases:
            with self.subTest(rules=rules):
                with self.assertRaises(MemoryScenarioConfigError):
                    parse_instance_performance_profile(
                        _v2_instance(
                            policy={
                                "default": "all_hbm",
                                "blocks": rules,
                            }
                        ),
                        4,
                    )

        with self.assertRaisesRegex(MemoryScenarioConfigError, "必须是列表"):
            parse_instance_performance_profile(
                _v2_instance(
                    policy={"default": "all_hbm", "blocks": {}}
                ),
                4,
            )

    def test_num_layers_and_scenario_query_indices_are_strict(self):
        for num_layers in (0, -1, True, 3.5):
            with self.subTest(num_layers=num_layers):
                with self.assertRaises(MemoryScenarioConfigError):
                    parse_instance_performance_profile({}, num_layers)

        policy = parse_instance_performance_profile(_v2_instance(), 4)
        for block_index in (-1, 4, True, 1.5):
            with self.subTest(block_index=block_index):
                with self.assertRaises(MemoryScenarioConfigError):
                    policy.scenario_for("qkv_proj", block_index)
        with self.assertRaises(MemoryScenarioConfigError):
            policy.scenario_for("qkv_*", 0)

    def test_policy_mappings_are_read_only(self):
        policy = parse_instance_performance_profile(
            _v2_instance(
                policy={
                    "default": "all_hbm",
                    "layers": {"qkv_proj": "hbf"},
                    "blocks": [{"blocks": "0", "scenario": "hbf"}],
                }
            ),
            4,
        )

        with self.assertRaises(TypeError):
            policy.layer_scenarios["attention"] = "hbf"
        with self.assertRaises(TypeError):
            policy.block_scenarios[1] = "hbf"


class MemoryScenarioCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.policy = parse_instance_performance_profile(_v2_instance(), 4)

    def test_v2_accepts_local_demand_access_and_remote_eviction(self):
        self.assertIsNone(_validate_gate(self.policy))

    def test_v2_rejects_each_legacy_runtime_path(self):
        flags = (
            "enable_local_offloading",
            "enable_attn_offloading",
            "enable_sub_batch_interleaving",
        )
        for flag in flags:
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(
                    MemoryScenarioCompatibilityError,
                    flag,
                ):
                    _validate_gate(self.policy, **{flag: True})

        with self.assertRaisesRegex(
            MemoryScenarioCompatibilityError,
            "最终解析后的布尔值",
        ):
            _validate_gate(self.policy, enable_local_offloading=1)

    def test_v2_rejects_nonlocal_weights_and_kv_at_every_scope(self):
        cases = (
            ("default weights", ("default", None, "weights")),
            ("default kv", ("default", None, "kv_loc")),
            ("block weights", ("block", 0, "weights")),
            ("block kv", ("block", 0, "kv_loc")),
            ("layer weights", ("layer", "qkv_proj", "weights")),
            ("layer kv", ("layer", "qkv_proj", "kv_loc")),
        )
        for name, (scope, key, kind) in cases:
            with self.subTest(name=name):
                placement = copy.deepcopy(_local_placement())
                if scope == "default":
                    placement[scope][kind] = "REMOTE:0"
                else:
                    placement[scope][key][kind] = "CXL:0"
                with self.assertRaisesRegex(
                    MemoryScenarioCompatibilityError,
                    kind,
                ):
                    _validate_gate(self.policy, placement=placement)

    def test_v2_allows_nonlocal_kv_evict_at_every_scope(self):
        placement = _local_placement()
        placement["default"]["kv_evict_loc"] = "CXL:0"
        placement["block"][0]["kv_evict_loc"] = "REMOTE:0"
        placement["layer"]["qkv_proj"]["kv_evict_loc"] = "CXL:1"

        self.assertIsNone(_validate_gate(self.policy, placement=placement))

    def test_v2_requires_normalized_placement_shape(self):
        invalid_placements = (
            None,
            {},
            {"default": {}},
            {"default": {"weights": "LOCAL", "kv_loc": "LOCAL"}, "block": {}},
            {"default": {"weights": "LOCAL", "kv_loc": "LOCAL"}, "layer": []},
        )
        for placement in invalid_placements:
            with self.subTest(placement=placement):
                with self.assertRaises(MemoryScenarioCompatibilityError):
                    _validate_gate(self.policy, placement=placement)

    def test_legacy_mode_does_not_apply_v2_gate(self):
        legacy = parse_instance_performance_profile({}, 4)

        self.assertIsNone(
            validate_memory_scenario_compatibility(
                legacy,
                enable_local_offloading=True,
                enable_attn_offloading=True,
                enable_sub_batch_interleaving=True,
                placement=None,
            )
        )

    def test_gate_requires_policy_object(self):
        with self.assertRaises(TypeError):
            validate_memory_scenario_compatibility(
                {},
                enable_local_offloading=False,
                enable_attn_offloading=False,
                enable_sub_batch_interleaving=False,
                placement=_local_placement(),
            )


if __name__ == "__main__":
    unittest.main()
