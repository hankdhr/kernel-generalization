import copy
import tempfile
import unittest
from pathlib import Path

from kernel_generalization.core import digest, pilot_tasks, summarize, validate_tasks, write_new


class ManifestTests(unittest.TestCase):
    def test_manifest_determinism_and_expected_coverage(self):
        tasks = pilot_tasks()
        self.assertEqual(digest(tasks), digest(pilot_tasks()))
        self.assertEqual(validate_tasks(tasks), {"train": 54, "dev": 18, "test": 36})
        self.assertEqual(len({t["task_id"] for t in tasks}), 108)

    def test_rehashed_split_leakage_rejected(self):
        tasks = pilot_tasks()
        tasks[0]["split"] = "test"
        tasks[0]["task_id"] = digest({k: v for k, v in tasks[0].items() if k != "task_id"})[:20]
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            validate_tasks(tasks)

    def test_tampering_and_overwrite_rejected(self):
        tasks = pilot_tasks()
        tasks[0]["rows"] = 99
        with self.assertRaises(ValueError):
            validate_tasks(tasks)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_new(path, "original")
            with self.assertRaises(FileExistsError):
                write_new(path, "replacement")
            self.assertEqual(path.read_text(), "original")


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.tasks = pilot_tasks()[:2]
        self.record = dict(task_id=self.tasks[0]["task_id"], seed=101, run_id="unit-fixture",
                           method="fixture", attempt=1, tokens_total=10, wall_seconds_total=1,
                           status="ok", baseline_ms=2, candidate_ms=1)

    def report(self, records, **kwargs):
        return summarize(self.tasks, records, **dict(seed=101, attempts=4, max_tokens=100,
                         max_wall=10, threshold=1.05, **kwargs))

    def test_missing_and_failed_tasks_remain_in_denominator(self):
        failure = {**self.record, "task_id": self.tasks[1]["task_id"], "status": "compile_error"}
        for records in ([self.record], [self.record, failure]):
            result = self.report(records)
            self.assertEqual(result["fast_rate"], .5)
            self.assertEqual(result["correct_rate"], .5)
            self.assertEqual(result["speedup_geomean_correct_only"], 2)

    def test_each_budget_independently_excludes_candidates(self):
        for update in ({"tokens_total": 101}, {"wall_seconds_total": 11}):
            self.assertEqual(self.report([{**self.record, **update}])["fast_rate"], 0)
        records = [{**self.record, "attempt": i, "status": "wrong"} for i in range(1, 5)]
        records.append({**self.record, "attempt": 5})
        self.assertEqual(self.report(records)["correct_rate"], 0)

    def test_threshold_is_strict_and_equality_cost_is_included(self):
        result = self.report([{**self.record, "baseline_ms": 1.05, "tokens_total": 100,
                               "wall_seconds_total": 10}])
        self.assertEqual(result["correct_rate"], .5)
        self.assertEqual(result["fast_rate"], 0)

    def test_invalid_logs_fail_loudly(self):
        bad_groups = [
            [self.record, self.record],
            [{**self.record, "attempt": 2}],
            [{**self.record, "candidate_ms": 0}],
            [{**self.record, "candidate_ms": float("nan")}],
            [self.record, {**self.record, "attempt": 2, "tokens_total": 9}],
            [self.record, {**self.record, "attempt": 2, "baseline_ms": 3}],
            [{**self.record, "task_id": "unknown"}],
            [{**self.record, "seed": 202}],
            [self.record, {**self.record, "attempt": 2, "method": "another"}],
        ]
        for records in bad_groups:
            with self.subTest(records=records), self.assertRaises(ValueError):
                self.report(copy.deepcopy(records))

    def test_empty_run_has_zero_coverage_not_success(self):
        result = self.report([])
        self.assertEqual(result["correct_rate"], 0)
        self.assertIsNone(result["speedup_geomean_correct_only"])

    def test_nonfinite_budget_rejected(self):
        with self.assertRaises(ValueError):
            summarize(self.tasks, [], seed=101, attempts=4, max_tokens=100, max_wall=float("nan"))

    def test_failed_attempt_cost_is_kept_for_first_success(self):
        records = [{**self.record, "status": "wrong"},
                   {**self.record, "attempt": 2, "tokens_total": 20, "wall_seconds_total": 3}]
        outcome = self.report(records)["outcomes"][0]
        self.assertEqual(outcome["first_correct_attempt"], 2)
        self.assertEqual(outcome["first_fast_tokens"], 20)
        self.assertEqual(outcome["first_correct_wall_seconds"], 3)


if __name__ == "__main__":
    unittest.main()
