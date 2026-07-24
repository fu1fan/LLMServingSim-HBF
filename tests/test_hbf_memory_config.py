import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from serving.core.config_builder import _mem_str
from serving.core.hbf_memory_config import (
    HbfMemoryConfigError,
    astra_memory_spec_from_integration,
    install_hbf_memory_resources,
)


def _integration(mode="cli", fabric=None):
    return {
        "mode": mode,
        "parameters": {
            "schema_version": 1,
            "timing_model": "directional_v1",
            "integration_mode": mode,
            "bandwidth_scope": "pure_direction_effective",
            "fabric_model": (
                "none" if fabric is None else "distinct_shared_resource"
            ),
            "gpu_memory_fabric_bandwidth_byte_per_second": fabric,
            "tiers": {
                "hbm": {
                    "read_write_service": "time_shared",
                    "read": {
                        "bandwidth_byte_per_second": 2e12,
                        "fixed_latency_second": 1e-6,
                        "latency_scope": "per_stream",
                        "request_granularity_byte": 32,
                        "max_inflight_requests": 1,
                    },
                    "write": {
                        "bandwidth_byte_per_second": 1e12,
                        "fixed_latency_second": 2e-6,
                        "latency_scope": "per_stream",
                        "request_granularity_byte": 32,
                        "max_inflight_requests": 1,
                    },
                },
                "hbf": {
                    "read_write_service": "time_shared",
                    "read": {
                        "bandwidth_byte_per_second": 4e12,
                        "fixed_latency_second": 5e-6,
                        "latency_scope": "per_stream",
                        "request_granularity_byte": 4096,
                        "max_inflight_requests": 1,
                    },
                    "write": {
                        "bandwidth_byte_per_second": 2.5e11,
                        "fixed_latency_second": 2e-4,
                        "latency_scope": "per_stream",
                        "request_granularity_byte": 4096,
                        "max_inflight_requests": 1,
                    },
                },
            },
        },
    }


class HbfMemoryConfigTest(unittest.TestCase):
    def test_cli_uses_independent_hbm_and_hbf_services(self):
        spec = astra_memory_spec_from_integration(_integration("cli"))

        self.assertEqual(spec.local_mem["read-mem-bw"], 2000)
        self.assertEqual(spec.hbf_mem["write-mem-bw"], 250)
        self.assertEqual(spec.hbf_mem["write-mem-latency"], 200_000)
        self.assertNotEqual(
            spec.local_mem["service-group"],
            spec.hbf_mem["service-group"],
        )

    def test_csi_uses_one_shared_service(self):
        spec = astra_memory_spec_from_integration(_integration("csi"))

        self.assertEqual(
            spec.local_mem["service-group"],
            spec.hbf_mem["service-group"],
        )
        self.assertNotIn("service-group-bw", spec.local_mem)

    def test_mode_mismatch_is_rejected(self):
        value = _integration("cli")
        value["parameters"]["integration_mode"] = "csi"

        with self.assertRaisesRegex(HbfMemoryConfigError, "integration_mode"):
            astra_memory_spec_from_integration(value)

    def test_zero_fixed_latency_is_preserved(self):
        value = _integration("cli")
        value["parameters"]["tiers"]["hbm"]["read"][
            "fixed_latency_second"
        ] = 0

        spec = astra_memory_spec_from_integration(value)

        self.assertEqual(spec.local_mem["read-mem-latency"], 0)

    def test_unsupported_latency_and_fabric_models_fail_closed(self):
        value = _integration("cli")
        value["parameters"]["tiers"]["hbf"]["read"][
            "latency_scope"
        ] = "per_request_batch"
        with self.assertRaisesRegex(HbfMemoryConfigError, "per_stream"):
            astra_memory_spec_from_integration(value)

        with self.assertRaisesRegex(HbfMemoryConfigError, "fabric_model"):
            astra_memory_spec_from_integration(
                _integration("csi", fabric=1.5e12)
            )

    def test_runtime_mapping_requires_complete_direction_contract(self):
        value = _integration("cli")
        del value["parameters"]["tiers"]["hbf"]["write"][
            "request_granularity_byte"
        ]

        with self.assertRaisesRegex(HbfMemoryConfigError, "字段不完整"):
            astra_memory_spec_from_integration(value)

    def test_installer_writes_only_explicit_migration_resources(self):
        runtime = {
            "dtype": "bfloat16",
            "kv_cache_dtype": "auto",
            "memory_tiering": SimpleNamespace(enabled=True),
            "memory_scenario_policy": SimpleNamespace(
                memory_profile_id="profile-a"
            ),
        }
        contract = SimpleNamespace(
            memory_integration=_integration("cli"),
            performance_identity={"digest": "a" * 64},
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "memory.json"
            path.write_text(
                json.dumps({"remote_mem": {"memory-type": "MEMORY_POOL"}}),
                encoding="utf-8",
            )
            with patch(
                "serving.core.hbf_memory_config.load_profile_contract",
                return_value=contract,
            ):
                install_hbf_memory_resources(
                    path,
                    [{"hardware": "gpu", "model_name": "model"}],
                    [runtime],
                    profiler_root=td,
                    variant_resolver=lambda *_: "bf16",
                )

            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("remote_mem", value)
            self.assertEqual(
                value["local_mem"]["memory-location"],
                "LOCAL_MEMORY",
            )
            self.assertEqual(
                value["hbf_mem"]["memory-location"],
                "HBF_MEMORY",
            )

    def test_hbf_placement_is_a_first_class_astra_location(self):
        self.assertEqual(_mem_str("hbf", 0), "HBF")
        self.assertEqual(_mem_str("hbf:2", 0), "HBF:2")


if __name__ == "__main__":
    unittest.main()
