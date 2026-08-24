import unittest
from types import SimpleNamespace

import numpy as np

from serving.core.gate_function import GateRouter, local_ep_rank_indices


def _profile():
    return SimpleNamespace(
        profile_id="custom-fixture-v1",
        sha256="4" * 64,
        selection_weights=(
            1.0,
            0.8,
            0.6,
            0.4,
            0.2,
            0.1,
            0.05,
            0.025,
        ),
    )


def _gate(seed=42, instance_id=0):
    return GateRouter(
        node_id=0,
        instance_id=instance_id,
        num_local_experts=8,
        num_experts_per_tok=2,
        routing_policy="CUSTOM",
        seed=seed,
        block_copy=False,
        custom_profile=_profile(),
    )


class CustomExpertRoutingTest(unittest.TestCase):
    def test_gumbel_top_k_selects_distinct_experts(self):
        selected = _gate()._sample_custom_experts(
            layer_num=3,
            routing_key="wave-7",
            total_len=128,
        )

        self.assertEqual(selected.shape, (128, 2))
        self.assertTrue(np.all(selected[:, 0] != selected[:, 1]))
        self.assertTrue(np.all(selected >= 0))
        self.assertTrue(np.all(selected < 8))

    def test_same_seed_and_routing_key_are_deterministic(self):
        first = _gate(instance_id=0).route_ep(
            5, "local-a", 256, 4, routing_key="shared-wave"
        )
        second = _gate(instance_id=1).route_ep(
            5, "local-b", 256, 4, routing_key="shared-wave"
        )

        self.assertEqual(first, second)

    def test_seed_changes_expert_to_rank_mapping(self):
        first = _gate(seed=42).route_ep(7, "batch", 512, 4)
        second = _gate(seed=101).route_ep(7, "batch", 512, 4)

        self.assertNotEqual(first.local_tokens, second.local_tokens)

    def test_route_result_obeys_conservation_bounds(self):
        total_len = 192
        result = _gate().route_ep(2, "batch", total_len, 4)

        self.assertEqual(sum(result.source_tokens), total_len)
        self.assertEqual(len(result.local_tokens), 4)
        self.assertEqual(len(result.activated_experts), 4)
        self.assertTrue(all(0 <= value <= total_len for value in result.local_tokens))
        self.assertTrue(all(0 <= value <= 2 for value in result.activated_experts))
        self.assertGreaterEqual(sum(result.local_tokens), total_len)
        self.assertLessEqual(sum(result.local_tokens), total_len * 2)

    def test_flat_route_assigns_top_k_per_token(self):
        counts = _gate().route(1, "batch", 64)
        self.assertEqual(len(counts), 8)
        self.assertEqual(sum(counts), 64 * 2)

    def test_dp_group_members_take_disjoint_global_rank_slices(self):
        routing = _gate().route_ep(
            4, "local", 256, 4, routing_key="shared-wave"
        )
        first = list(local_ep_rank_indices(2, 4, 0))
        second = list(local_ep_rank_indices(2, 4, 2))

        first_loads = [routing.local_tokens[idx] for idx in first]
        second_loads = [routing.local_tokens[idx] for idx in second]
        self.assertEqual(first + second, [0, 1, 2, 3])
        self.assertEqual(len(first_loads), 2)
        self.assertEqual(len(second_loads), 2)


if __name__ == "__main__":
    unittest.main()
