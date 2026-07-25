import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from serving.core.graph_generator import generate_graph


class GraphGeneratorTest(unittest.TestCase):
    def test_chakra_uses_current_python_interpreter(self):
        batch = SimpleNamespace(model="model", batch_id=1)
        with tempfile.TemporaryDirectory() as td:
            with patch(
                "serving.core.graph_generator.subprocess.run"
            ) as run:
                generate_graph(
                    batch,
                    "gpu",
                    1,
                    inputs_root=str(Path(td) / "inputs"),
                    cleanup_trace=False,
                )

        command = run.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
