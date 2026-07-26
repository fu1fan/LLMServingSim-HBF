import os
import tempfile
import unittest
from pathlib import Path

from serving.core.request import Batch, Request
from serving.core.trace_generator import generate_trace


MODEL = "meta-llama/Llama-3.1-8B"
HARDWARE = "RTXPRO6000"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _batch():
    request = Request(
        id=0,
        model=MODEL,
        input=4,
        output=6,
        arrival=0,
        instance_id=0,
        is_init=True,
    )
    batch = Batch(
        0,
        MODEL,
        4,
        0,
        [4],
        [0],
        1,
        0,
        [4],
        [0],
        [],
        0,
        0,
    )
    batch.requests.append(request)
    return batch


def _placement(weight_location):
    return {
        "default": {
            "weights": weight_location,
            "kv_loc": "LOCAL",
            "kv_evict_loc": "REMOTE:0",
        },
        "block": [],
        "layer": {},
    }


def _hbf(scale):
    return {
        "schema_version": 1,
        "num_stacks": 8,
        "stack_capacity_gb": 512,
        "mem_size": 4096,
        "performance": {
            "source": "scale",
            "latency_scale": scale,
        },
    }


def _generate(root, placement, hbf_mem=None):
    generate_trace(
        _batch(),
        HARDWARE,
        tp_size=1,
        pp_size=1,
        local_ep=1,
        ep_total=1,
        instance_id=0,
        max_num_batched_tokens=2048,
        max_num_seqs=128,
        placement=placement,
        fp=16,
        dtype="bfloat16",
        inputs_root=root,
        hbf_mem=hbf_mem,
    )
    path = (
        Path(root)
        / "trace"
        / HARDWARE
        / MODEL
        / "instance0_batch0.txt"
    )
    return path.read_text(encoding="utf-8")


def _operator_rows(trace):
    lines = trace.splitlines()[3:]
    return [line.split() for line in lines]


class HBFTraceEndToEndTest(unittest.TestCase):
    def test_identity_trace_matches_hbm_and_scale_is_weight_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous_cwd = Path.cwd()
            try:
                os.chdir(REPO_ROOT / "astra-sim")
                baseline = _generate(
                    root / "baseline", _placement("LOCAL")
                )
                identity = _generate(
                    root / "identity", _placement("HBF"), _hbf(1.0)
                )
                scaled = _generate(
                    root / "scaled", _placement("HBF"), _hbf(2.0)
                )
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(identity, baseline)
        self.assertNotIn("HBF", scaled)

        baseline_rows = _operator_rows(baseline)
        scaled_rows = _operator_rows(scaled)
        self.assertEqual(len(scaled_rows), len(baseline_rows))
        changed = 0
        unchanged_zero_weight = 0
        for base, hbf in zip(baseline_rows, scaled_rows):
            self.assertEqual(base[0], hbf[0])
            self.assertEqual(base[2:], hbf[2:])
            weight_bytes = int(base[5])
            baseline_latency = int(base[1])
            hbf_latency = int(hbf[1])
            if weight_bytes > 0:
                self.assertEqual(hbf_latency, baseline_latency * 2)
                changed += 1
            else:
                self.assertEqual(hbf_latency, baseline_latency)
                unchanged_zero_weight += 1
        self.assertGreater(changed, 0)
        self.assertGreater(unchanged_zero_weight, 0)


if __name__ == "__main__":
    unittest.main()
