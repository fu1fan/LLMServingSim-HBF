import unittest
from types import SimpleNamespace

from serving.core.trace_generator import _trace_copy_plan


def context(is_moe, block_copy):
    return SimpleNamespace(
        is_moe=is_moe,
        gate=SimpleNamespace(block_copy=block_copy),
    )


class MoeTraceCopyPlanTests(unittest.TestCase):
    def test_balanced_moe_keeps_existing_block_copy(self):
        self.assertEqual(
            _trace_copy_plan(context(True, True), 94, False),
            (1, 94, True),
        )

    def test_custom_moe_emits_every_layer(self):
        self.assertEqual(
            _trace_copy_plan(context(True, False), 94, False),
            (94, 1, False),
        )

    def test_explicit_block_placement_emits_every_layer(self):
        self.assertEqual(
            _trace_copy_plan(context(False, True), 126, True),
            (126, 1, False),
        )

    def test_sub_batch_middle_layers_use_same_contract(self):
        self.assertEqual(
            _trace_copy_plan(context(True, False), 93, False),
            (93, 1, False),
        )


if __name__ == "__main__":
    unittest.main()
