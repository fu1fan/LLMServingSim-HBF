import json
import unittest
from pathlib import Path

from serving.core.hbf_model import GB_TO_BYTE, parse_hbf_config


REPO_ROOT = Path(__file__).resolve().parents[1]


class HBFTargetConfigTest(unittest.TestCase):
    def test_target_model_shapes(self):
        expected = {
            "configs/model/meta-llama/Llama-3.1-405B-Instruct.json": {
                "model_type": "llama",
                "hidden_size": 16384,
                "num_hidden_layers": 126,
                "num_attention_heads": 128,
                "num_key_value_heads": 8,
            },
            "configs/model/Qwen/Qwen3-235B-A22B.json": {
                "model_type": "qwen3_moe",
                "hidden_size": 4096,
                "num_hidden_layers": 94,
                "num_attention_heads": 64,
                "num_key_value_heads": 4,
                "num_experts": 128,
                "num_experts_per_tok": 8,
            },
        }
        for relative_path, fields in expected.items():
            with open(REPO_ROOT / relative_path, encoding="utf-8") as stream:
                config = json.load(stream)
            for key, value in fields.items():
                self.assertEqual(config[key], value)
            self.assertEqual(config["torch_dtype"], "bfloat16")

    def test_hardware_templates_use_per_gpu_4096_gib_hbf(self):
        templates = sorted(
            (REPO_ROOT / "configs" / "cluster").glob(
                "hbf_*_llama405b_tp8.json"
            )
        ) + sorted(
            (REPO_ROOT / "configs" / "cluster").glob(
                "hbf_*_qwen3_235b_tp4_ep4.json"
            )
        )
        self.assertEqual(len(templates), 4)
        for path in templates:
            with open(path, encoding="utf-8") as stream:
                cluster = json.load(stream)
            instance = cluster["nodes"][0]["instances"][0]
            hbf = parse_hbf_config(instance["hbf_mem"])
            self.assertEqual(hbf.capacity_bytes, 4096 * GB_TO_BYTE)
            self.assertEqual(
                instance["placement"]["default"]["weights"], "hbf"
            )
            self.assertEqual(
                instance["placement"]["default"]["kv_loc"], "npu"
            )


if __name__ == "__main__":
    unittest.main()
