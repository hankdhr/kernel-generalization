import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kernel_generalization.core import pilot_tasks

ROOT = Path(__file__).resolve().parents[1]


class EntrypointTests(unittest.TestCase):
    def test_gpu_plan_does_not_import_torch_or_emit_timings(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "probe"
            # -S hides site packages: the dry run must work without torch/numpy.
            result = subprocess.run([sys.executable, "-S", "scripts/run_probe.py", "--plan-only",
                                     "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads((out / "plan.json").read_text())
            self.assertEqual(plan["status"], "planned")
            self.assertFalse((out / "completion.json").exists())
            self.assertEqual(len(list(out.iterdir())), 1)

    def test_generation_plan_and_holdout_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "generation"
            command = [sys.executable, "-S", "scripts/generate_one.py", "--plan-only", "--task-id",
                       pilot_tasks()[0]["task_id"], "--out", str(out)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((out / "response.txt").exists())
            self.assertIn("run(torch::Tensor x)", (out / "prompt.txt").read_text())
            holdout = next(t for t in pilot_tasks() if t["split"] == "test")
            command[command.index("--task-id") + 1] = holdout["task_id"]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("deferred", result.stderr)

    def test_missing_gpu_dependency_is_saved_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "failed-probe"
            result = subprocess.run([sys.executable, "-S", "scripts/run_probe.py", "--out", str(out)],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads((out / "completion.json").read_text())["status"], "failed")
            self.assertTrue((out / "plan.json").exists())


if __name__ == "__main__":
    unittest.main()
