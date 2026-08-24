import unittest

from serving.core.config_builder import _resolve_dp_groups
from serving.core.gate_function import local_ep_rank_indices


class MoEDPGroupRoutingTest(unittest.TestCase):
    def test_dp_group_assigns_disjoint_global_ep_slices(self):
        instances = [
            {
                "tp_size": 2,
                "pp_size": 1,
                "ep_size": 4,
                "dp_group": "experts",
            },
            {
                "tp_size": 2,
                "pp_size": 1,
                "ep_size": 4,
                "dp_group": "experts",
            },
        ]

        _resolve_dp_groups(instances)

        self.assertEqual(instances[0]["dp_group_rank"], 0)
        self.assertEqual(instances[0]["ep_rank_offset"], 0)
        self.assertEqual(instances[1]["dp_group_rank"], 1)
        self.assertEqual(instances[1]["ep_rank_offset"], 2)
        self.assertEqual(instances[0]["local_ep"], 2)
        self.assertEqual(instances[1]["local_ep"], 2)

    def test_local_ep_rank_ranges_do_not_overlap(self):
        first = list(local_ep_rank_indices(2, 4, 0))
        second = list(local_ep_rank_indices(2, 4, 2))

        self.assertEqual(first, [0, 1])
        self.assertEqual(second, [2, 3])
        self.assertTrue(set(first).isdisjoint(second))

    def test_invalid_ep_slice_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid EP rank slice"):
            list(local_ep_rank_indices(2, 4, 3))

    def test_non_dp_instance_uses_zero_offset(self):
        instance = {
            "num_npus": 4,
            "tp_size": 4,
            "pp_size": 1,
            "ep_size": 4,
            "dp_group": None,
        }

        _resolve_dp_groups([instance])

        self.assertEqual(instance["dp_group_rank"], 0)
        self.assertEqual(instance["ep_rank_offset"], 0)
        self.assertEqual(instance["local_ep"], 4)


if __name__ == "__main__":
    unittest.main()
