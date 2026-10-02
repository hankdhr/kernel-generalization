"""CPU-only manifests, environment inventory, and budgeted metrics."""
import argparse
import hashlib
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

FAMILIES = {
    "row_sum": "train",
    "row_sumsq": "train",
    "row_relu_sum": "train",
    "row_mean": "dev",
    "row_rmsnorm": "test",
    "row_softmax": "test",
}
SEMANTICS = {
    "row_sum": "y[r] = sum_c x[r,c]",
    "row_sumsq": "y[r] = sum_c x[r,c]**2",
    "row_relu_sum": "y[r] = sum_c max(x[r,c], 0)",
    "row_mean": "y[r] = sum_c x[r,c] / C",
    "row_rmsnorm": "y[r,c] = x[r,c] / sqrt(mean_c(x[r,c]**2) + 1e-5); no weight",
    "row_softmax": "y[r,c] = exp(x[r,c]-max_c(x[r,c])) / sum_c exp(x[r,c]-max_c(x[r,c]))",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def pilot_tasks():
    tasks = []
    for family, split in FAMILIES.items():
        for rows in (1, 32, 512):
            for cols in (127, 1024, 4097):
                for stride in (1, 2):
                    spec = {
                        "suite": "pilot-v0", "family": family, "split": split,
                        "rows": rows, "cols": cols, "dtype": "float32",
                        "stride": [cols * stride, stride], "semantics": SEMANTICS[family],
                        "input_seeds": [17, 29, 43],
                        "distributions": ["normal", "uniform", "zero", "impulse", "alternating"],
                        "rtol": 1e-4, "atol": 2e-4,
                    }
                    tasks.append({"task_id": digest(spec)[:20], **spec})
    return tasks


def validate_tasks(tasks):
    if not tasks:
        raise ValueError("empty task manifest")
    ids, family_splits = set(), {}
    for task in tasks:
        task_id = task["task_id"]
        spec = {k: v for k, v in task.items() if k != "task_id"}
        if task_id != digest(spec)[:20] or task_id in ids:
            raise ValueError("task hash mismatch or duplicate task")
        ids.add(task_id)
        family, split = task["family"], task["split"]
        if split not in {"train", "dev", "test"}:
            raise ValueError("unknown split")
        if family in family_splits and family_splits[family] != split:
            raise ValueError("family appears in multiple splits")
        family_splits[family] = split
    return {split: sum(t["split"] == split for t in tasks) for split in ("train", "dev", "test")}


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_new(path, text):
    """Never silently replace a manifest or result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def numeric(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"invalid {name}")
    return value


def summarize(tasks, records, *, seed, attempts, max_tokens, max_wall, threshold=1.05):
    """One method/run/seed only. Missing tasks stay in the denominator.

    Each record uses cumulative costs from the start of that task's run.
    Exceeding ANY resource limit excludes a candidate. Correctness must cover
    all prescribed inputs before status='ok' is emitted by a trusted worker.
    """
    validate_tasks(tasks)
    if type(attempts) is not int or type(max_tokens) is not int:
        raise ValueError("attempt and token budgets must be integers")
    for name, value in (("attempts", attempts), ("max_tokens", max_tokens),
                        ("max_wall", max_wall), ("threshold", threshold)):
        numeric(value, name, positive=True)
    groups = {t["task_id"]: [] for t in tasks}
    run_keys = set()
    for record in records:
        if record["task_id"] not in groups:
            raise ValueError("record references task outside the selected manifest")
        if record["seed"] != seed:
            raise ValueError("select exactly one seed before aggregation")
        run_keys.add((record["run_id"], record["method"]))
        if len(run_keys) > 1:
            raise ValueError("cannot mix methods or runs")
        if record["status"] not in {"ok", "compile_error", "runtime_error", "wrong", "timeout", "generation_error"}:
            raise ValueError("unknown candidate status")
        if type(record["attempt"]) is not int or record["attempt"] < 1:
            raise ValueError("attempt must be a positive integer")
        if type(record["tokens_total"]) is not int:
            raise ValueError("tokens_total must be an integer")
        numeric(record["tokens_total"], "tokens_total")
        numeric(record["wall_seconds_total"], "wall_seconds_total")
        if record["status"] == "ok":
            numeric(record["baseline_ms"], "baseline_ms", positive=True)
            numeric(record["candidate_ms"], "candidate_ms", positive=True)
        groups[record["task_id"]].append(record)

    outcomes = []
    for task_id, group in groups.items():
        group.sort(key=lambda r: r["attempt"])
        last_attempt, last_tokens, last_wall = 0, 0, 0
        valid, in_budget = [], []
        baseline = None
        for r in group:
            if r["attempt"] != last_attempt + 1:
                raise ValueError("attempts must be unique and contiguous, including failures")
            if r["tokens_total"] < last_tokens or r["wall_seconds_total"] < last_wall:
                raise ValueError("cumulative costs decreased")
            last_attempt, last_tokens, last_wall = r["attempt"], r["tokens_total"], r["wall_seconds_total"]
            if r["status"] == "ok":
                if baseline is not None and r["baseline_ms"] != baseline:
                    raise ValueError("baseline changed within task; use a frozen evaluation baseline")
                baseline = r["baseline_ms"]
            if r["attempt"] <= attempts and r["tokens_total"] <= max_tokens and r["wall_seconds_total"] <= max_wall:
                in_budget.append(r)
                if r["status"] == "ok":
                    valid.append(r)
        best = max((r["baseline_ms"] / r["candidate_ms"] for r in valid), default=None)
        reached = next((r for r in valid if r["baseline_ms"] / r["candidate_ms"] > threshold), None)
        outcomes.append({
            "task_id": task_id, "attempted_in_budget": bool(in_budget),
            "correct": bool(valid), "best_speedup": best,
            "fast": best is not None and best > threshold,
            "first_correct_attempt": valid[0]["attempt"] if valid else None,
            "first_correct_tokens": valid[0]["tokens_total"] if valid else None,
            "first_correct_wall_seconds": valid[0]["wall_seconds_total"] if valid else None,
            "first_fast_attempt": reached["attempt"] if reached else None,
            "first_fast_tokens": reached["tokens_total"] if reached else None,
            "first_fast_wall_seconds": reached["wall_seconds_total"] if reached else None,
        })
    n = len(outcomes)
    speeds = [o["best_speedup"] for o in outcomes if o["correct"]]
    run_id, method = next(iter(run_keys)) if run_keys else (None, None)
    return {
        "schema_version": 1, "seed": seed, "task_count": n,
        "run_id": run_id, "method": method, "records_sha256": digest(records),
        "manifest_sha256": digest(tasks),
        "budget": {"attempts": attempts, "tokens": max_tokens, "wall_seconds": max_wall},
        "threshold_strictly_greater_than": threshold,
        "attempted_rate": sum(o["attempted_in_budget"] for o in outcomes) / n,
        "correct_rate": len(speeds) / n,
        "fast_rate": sum(o["fast"] for o in outcomes) / n,
        "speedup_geomean_correct_only": math.exp(statistics.mean(map(math.log, speeds))) if speeds else None,
        "outcomes": outcomes,
    }


def inventory():
    checks = {}
    for executable, args in {
        "nvidia-smi": ["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        "nvcc": ["--version"], "ncu": ["--version"], "git": ["rev-parse", "HEAD"],
    }.items():
        if not shutil.which(executable):
            checks[executable] = {"available": False}
            continue
        try:
            proc = subprocess.run([executable, *args], capture_output=True, text=True, timeout=15)
            checks[executable] = {"available": True, "exit_code": proc.returncode,
                                  "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
        except subprocess.TimeoutExpired:
            checks[executable] = {"available": True, "status": "timeout"}
    return {"python": sys.version, "platform": platform.platform(), "checks": checks,
            "gpu_execution_verified": False, "profiling_counters_verified": False,
            "note": "Inventory only. A successful CUDA run and an actual ncu capture are separate checks."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    suite = sub.add_parser("suite")
    suite.add_argument("--out", required=True)
    check = sub.add_parser("check")
    check.add_argument("manifest")
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--out")
    report = sub.add_parser("report")
    report.add_argument("--manifest", required=True)
    report.add_argument("--records", required=True)
    report.add_argument("--split", choices=["train", "dev", "test"], required=True)
    report.add_argument("--seed", type=int, required=True)
    report.add_argument("--attempts", type=int, required=True)
    report.add_argument("--max-tokens", type=int, required=True)
    report.add_argument("--max-wall", type=float, required=True)
    report.add_argument("--threshold", type=float, default=1.05)
    report.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "suite":
        tasks = pilot_tasks()
        write_new(args.out, "".join(canonical(t) + "\n" for t in tasks))
        print(json.dumps({"counts": validate_tasks(tasks), "sha256": digest(tasks)}))
    elif args.command == "check":
        tasks = read_jsonl(args.manifest)
        print(json.dumps({"counts": validate_tasks(tasks), "sha256": digest(tasks)}))
    elif args.command == "preflight":
        result = json.dumps(inventory(), indent=2) + "\n"
        if args.out:
            write_new(args.out, result)
        print(result)
    else:
        tasks = [t for t in read_jsonl(args.manifest) if t["split"] == args.split]
        records = read_jsonl(args.records)
        result = summarize(tasks, records, seed=args.seed, attempts=args.attempts,
                           max_tokens=args.max_tokens, max_wall=args.max_wall, threshold=args.threshold)
        write_new(args.out, json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "outcomes"}, indent=2))
