import unittest

from serving.core.memory_tiering import MemoryTier
from serving.core.memory_tiering_config import (
    MemoryTieringConfigError,
    parse_instance_memory_tiering,
    validate_homogeneous_hbf_instances,
)


def _hbf_instance(tiering=None):
    instance = {
        "hbf_mem": {"mem_size": 1024},
        "performance_profile": {
            "mode": "memory_scenario_v2",
            "memory_profile_id": "hbf-cli-a",
            "scenario_selection": "residency_derived",
        },
    }
    if tiering is not None:
        instance["memory_tiering"] = tiering
    return instance


class MemoryTieringConfigTest(unittest.TestCase):
    def test_missing_hbf_config_uses_legacy_hbm_defaults(self):
        config = parse_instance_memory_tiering({}, 32)

        self.assertFalse(config.enabled)
        self.assertEqual(config.weights.default_tier, MemoryTier.HBM)
        self.assertEqual(config.kv.admission_tier, MemoryTier.HBM)
        self.assertEqual(config.communication_buffers.tier, MemoryTier.HBM)

    def test_hbf_defaults_remain_hbm_until_policy_is_explicit(self):
        config = parse_instance_memory_tiering(_hbf_instance(), 32)

        self.assertTrue(config.enabled)
        self.assertEqual(config.hbf_capacity_bytes, 1024 * 1024**3)
        self.assertEqual(config.weights.policy, "hbm_only")
        self.assertEqual(config.kv.policy, "hbm_only")

    def test_static_weight_map_uses_layer_then_block_then_default(self):
        config = parse_instance_memory_tiering(
            _hbf_instance(
                {
                    "weights": {
                        "policy": "static_map",
                        "default_tier": "hbm",
                        "layers": {"lm_head": "hbf"},
                        "blocks": [{"blocks": "0-2,5", "tier": "hbf"}],
                    }
                }
            ),
            8,
        )

        self.assertEqual(config.weights.tier_for("lm_head", None), MemoryTier.HBF)
        self.assertEqual(config.weights.tier_for("qkv_proj", 1), MemoryTier.HBF)
        self.assertEqual(config.weights.tier_for("qkv_proj", 4), MemoryTier.HBM)

    def test_length_threshold_requires_positive_threshold(self):
        with self.assertRaisesRegex(
            MemoryTieringConfigError,
            "threshold_tokens",
        ):
            parse_instance_memory_tiering(
                _hbf_instance({"kv": {"policy": "length_threshold"}}),
                32,
            )

    def test_watermark_order_is_strict(self):
        with self.assertRaisesRegex(
            MemoryTieringConfigError,
            "low_watermark",
        ):
            parse_instance_memory_tiering(
                _hbf_instance(
                    {
                        "kv": {
                            "policy": "watermark_lru",
                            "hbm_high_watermark": 0.8,
                            "hbm_low_watermark": 0.9,
                        }
                    }
                ),
                32,
            )

    def test_hbf_communication_staging_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(
            MemoryTieringConfigError,
            "allow_hbf_staging",
        ):
            parse_instance_memory_tiering(
                _hbf_instance(
                    {"communication_buffers": {"tier": "hbf"}}
                ),
                32,
            )

    def test_residency_mode_rejects_manual_scenario_policy(self):
        instance = _hbf_instance()
        instance["performance_profile"]["scenario_policy"] = {
            "default": "all_hbm"
        }

        with self.assertRaisesRegex(
            MemoryTieringConfigError,
            "互斥",
        ):
            parse_instance_memory_tiering(instance, 32)

    def test_hbf_instances_must_be_homogeneous(self):
        with self.assertRaisesRegex(
            MemoryTieringConfigError,
            "不能混用",
        ):
            validate_homogeneous_hbf_instances([_hbf_instance(), {}])

        self.assertFalse(validate_homogeneous_hbf_instances([{}, {}]))
        self.assertTrue(
            validate_homogeneous_hbf_instances(
                [_hbf_instance(), _hbf_instance()]
            )
        )

    def test_unknown_fields_fail_closed(self):
        with self.assertRaisesRegex(
            MemoryTieringConfigError,
            "未知字段",
        ):
            parse_instance_memory_tiering(
                _hbf_instance({"kv": {"policy": "hbm_only", "typo": 1}}),
                32,
            )


if __name__ == "__main__":
    unittest.main()
