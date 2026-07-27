import unittest

from serving.core.gate_function import GateRouter


class MoERoutingRegressionTest(unittest.TestCase):
    def _route(self, policy):
        gate = GateRouter(
            node_id=0,
            instance_id=0,
            num_local_experts=16,
            num_experts_per_tok=2,
            routing_policy=policy,
            seed=7,
            block_copy=False,
        )
        return gate.route_ep(
            layer_num=3,
            batch_id="batch-4",
            total_len=13,
            ep_size=4,
        )

    def test_balanced_routing_golden(self):
        result = self._route("BALANCED")
        self.assertEqual(result.local_tokens, [6, 6, 6, 6])
        self.assertEqual(result.activated_experts, [4, 4, 4, 4])
        self.assertEqual(result.source_tokens, [4, 3, 3, 3])

    def test_round_robin_routing_golden(self):
        result = self._route("RR")
        self.assertEqual(result.local_tokens, [13, 0, 0, 0])
        self.assertEqual(result.activated_experts, [2, 0, 0, 0])
        self.assertEqual(result.source_tokens, [4, 3, 3, 3])

    def test_seeded_random_routing_golden(self):
        result = self._route("RAND")
        self.assertEqual(result.local_tokens, [12, 4, 3, 7])
        self.assertEqual(result.activated_experts, [4, 3, 2, 4])
        self.assertEqual(result.source_tokens, [4, 3, 3, 3])


if __name__ == "__main__":
    unittest.main()
