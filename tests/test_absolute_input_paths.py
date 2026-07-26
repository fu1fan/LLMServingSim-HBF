import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from serving.core.router import Router
from serving.core.run_paths import resolve_user_input_path


class AbsoluteInputPathTest(unittest.TestCase):
    def test_absolute_paths_are_not_prefixed(self):
        self.assertEqual(
            resolve_user_input_path("/tmp/cluster.json"),
            "/tmp/cluster.json",
        )
        self.assertEqual(
            resolve_user_input_path("configs/cluster/test.json"),
            "../configs/cluster/test.json",
        )

    def test_router_loads_absolute_sweep_dataset(self):
        scheduler = SimpleNamespace(pd_type=None)
        router = Router(1, [scheduler], req_num=0, routing_policy="LOAD")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "requests.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "input_toks": 4,
                        "output_toks": 2,
                        "arrival_time_ns": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            router.load_requests(path)
        self.assertEqual(len(router._pending_requests), 1)
        self.assertEqual(router._pending_requests[0]["input_toks"], 4)


if __name__ == "__main__":
    unittest.main()
