import unittest

from serving.core.memory_tiering import MemoryTier
from serving.core.residency_scenario import (
    BatchMemoryView,
    ResidencyScenarioError,
    ResidencyScenarioResolver,
)


def _access_catalog():
    return {
        "qkv_proj/input": {
            "semantic": "activation",
            "access_type": "read",
            "lifetime": "iteration",
        },
        "qkv_proj/weight": {
            "semantic": "weight",
            "access_type": "read",
            "lifetime": "model",
        },
        "qkv_proj/output": {
            "semantic": "activation",
            "access_type": "write",
            "lifetime": "iteration",
        },
        "attention/query": {
            "semantic": "activation",
            "access_type": "read",
            "lifetime": "iteration",
        },
        "attention/key_cache_history": {
            "semantic": "kv_cache",
            "access_type": "read",
            "lifetime": "request",
        },
        "attention/key_cache_append": {
            "semantic": "kv_cache",
            "access_type": "write",
            "lifetime": "request",
        },
    }


def _scenario_catalog():
    base = {key: "hbm" for key in _access_catalog()}
    weights = dict(base)
    weights["qkv_proj/weight"] = "hbf"
    kv = dict(base)
    kv["attention/key_cache_history"] = "hbf"
    kv["attention/key_cache_append"] = "hbf"
    return {
        "all_hbm": {"accesses": base},
        "weights_hbf": {"accesses": weights},
        "kv_hbf": {"accesses": kv},
    }


class ResidencyScenarioResolverTest(unittest.TestCase):
    def test_batch_view_uses_exact_weight_and_request_layer_kv(self):
        view = BatchMemoryView(
            snapshot_version=7,
            weight_tiers={
                ("qkv_proj", None): MemoryTier.HBM,
                ("qkv_proj", 3): MemoryTier.HBF,
            },
            kv_tiers={
                ("10", 3): MemoryTier.HBF,
                ("11", 3): MemoryTier.HBM,
            },
        )

        self.assertEqual(
            view.weight_tier("qkv_proj", 3),
            MemoryTier.HBF,
        )
        self.assertEqual(
            view.weight_tier("qkv_proj", 2),
            MemoryTier.HBM,
        )
        self.assertEqual(
            view.kv_groups([10, 11], 3),
            {
                MemoryTier.HBF: ("10",),
                MemoryTier.HBM: ("11",),
            },
        )

    def test_batch_view_rejects_missing_request_layer_binding(self):
        view = BatchMemoryView(
            snapshot_version=0,
            weight_tiers={("attention", None): MemoryTier.HBM},
            kv_tiers={},
        )

        with self.assertRaisesRegex(
            ResidencyScenarioError,
            "request=1.*layer=0",
        ):
            view.kv_tier(1, 0)

    def test_weight_scenario_comes_from_actual_residency(self):
        resolver = ResidencyScenarioResolver(
            _access_catalog(),
            _scenario_catalog(),
        )

        binding = resolver.resolve(
            "qkv_proj",
            weight_tier=MemoryTier.HBF,
        )

        self.assertEqual(binding.scenario_id, "weights_hbf")
        self.assertEqual(
            binding.access_tiers["qkv_proj/input"],
            MemoryTier.HBM,
        )

    def test_kv_history_and_append_move_as_one_unit(self):
        resolver = ResidencyScenarioResolver(
            _access_catalog(),
            _scenario_catalog(),
        )

        binding = resolver.resolve(
            "attention",
            kv_tier=MemoryTier.HBF,
        )

        self.assertEqual(binding.scenario_id, "kv_hbf")
        self.assertEqual(
            binding.access_tiers["attention/key_cache_history"],
            MemoryTier.HBF,
        )
        self.assertEqual(
            binding.access_tiers["attention/key_cache_append"],
            MemoryTier.HBF,
        )

    def test_nonpersistent_access_cannot_follow_hbf_weight(self):
        resolver = ResidencyScenarioResolver(
            _access_catalog(),
            _scenario_catalog(),
        )

        binding = resolver.resolve(
            "qkv_proj",
            weight_tier=MemoryTier.HBF,
        )

        self.assertEqual(binding.access_tiers["qkv_proj/output"], MemoryTier.HBM)

    def test_missing_reachable_scenario_fails_preflight(self):
        catalog = _scenario_catalog()
        del catalog["kv_hbf"]
        resolver = ResidencyScenarioResolver(_access_catalog(), catalog)

        with self.assertRaisesRegex(ResidencyScenarioError, "没有 Profile 场景"):
            resolver.preflight(
                allow_weight_hbf=True,
                allow_kv_hbf=True,
            )

    def test_all_scenarios_must_cover_exact_access_catalog(self):
        catalog = _scenario_catalog()
        del catalog["all_hbm"]["accesses"]["attention/query"]

        with self.assertRaisesRegex(ResidencyScenarioError, "覆盖不完整"):
            ResidencyScenarioResolver(_access_catalog(), catalog)

    def test_unknown_operator_is_rejected(self):
        resolver = ResidencyScenarioResolver(
            _access_catalog(),
            _scenario_catalog(),
        )

        with self.assertRaisesRegex(ResidencyScenarioError, "不包含 operator"):
            resolver.resolve("lm_head")


if __name__ == "__main__":
    unittest.main()
