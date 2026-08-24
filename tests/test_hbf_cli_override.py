import unittest

from serving.core.hbf_performance import (
    apply_hbf_latency_scale_override,
)


def _instance(source="scale"):
    performance = {"source": source}
    if source == "scale":
        performance["latency_scale"] = 1.0
    else:
        performance.update({
            "profile_root": "/tmp/hbf",
            "profile_hardware": "B200_HBF",
        })
    return {
        "hbf_mem": {
            "schema_version": 1,
            "num_stacks": 8,
            "stack_capacity_gb": 512,
            "performance": performance,
        }
    }


class HBFCliOverrideTest(unittest.TestCase):
    def test_override_updates_every_scale_instance(self):
        instances = [_instance(), _instance()]

        apply_hbf_latency_scale_override(instances, 2.5)

        self.assertEqual(
            [item["hbf_mem"]["performance"]["latency_scale"]
             for item in instances],
            [2.5, 2.5],
        )

    def test_override_requires_hbf_and_scale_source(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            apply_hbf_latency_scale_override([{}], 2.0)

        with self.assertRaisesRegex(ValueError, "profile source"):
            apply_hbf_latency_scale_override([_instance("profile")], 2.0)

    def test_absent_override_is_a_noop(self):
        instances = [_instance()]

        apply_hbf_latency_scale_override(instances, None)

        self.assertEqual(
            instances[0]["hbf_mem"]["performance"]["latency_scale"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
