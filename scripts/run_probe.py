"""GPU measurement smoke test for TRUSTED handwritten probes, not agent evaluation.

Use an exclusive GPU. Compilation and torch.compile are excluded from latency,
but total elapsed setup is retained. Generated CUDA requires a separate isolated
worker before it can be executed; this script does not load generated code.
"""
import argparse
import datetime
import hashlib
import json
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kernel_generalization.core import digest, inventory, read_jsonl, validate_tasks, write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "data/pilot-v0.jsonl"))
    parser.add_argument("--split", choices=["train", "dev", "test"], default="train")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", required=True, help="New run directory")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.limit < 1 or args.repeats < 3 or args.warmup < 1:
        parser.error("limit >= 1, repeats >= 3 and warmup >= 1 required")
    tasks = read_jsonl(args.manifest)
    validate_tasks(tasks)
    selected = [t for t in tasks if t["split"] == args.split][:args.limit]
    if not selected:
        parser.error("no selected tasks")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    kernel = ROOT / "kernels/row_probe.cu"
    meta = {"kind": "handwritten_harness_probe", "status": "planned",
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "args": vars(args), "manifest_sha256": digest(tasks),
            "kernel_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
            "selected_task_ids": [t["task_id"] for t in selected], "environment": inventory(),
            "timing": "CUDA events, warm cache, output allocation included, no CUDA Graph",
            "precision": "FP32, float64 correctness reference; finite inputs only"}
    write_new(out / "plan.json", json.dumps(meta, indent=2) + "\n")
    if args.plan_only:
        print("Plan saved. No CUDA compiled, model called, or latency measured.")
        return
    started = time.monotonic()
    try:
        import torch
        from torch.utils.cpp_extension import load
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU unavailable; this is not a successful experiment")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        ext = load(name="kg_probe_" + meta["kernel_sha256"][:12], sources=[str(kernel)],
                   extra_cuda_cflags=["-O3"], verbose=False)
        meta.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    device=torch.cuda.get_device_name(0))
        operations = ["row_sum", "row_sumsq", "row_relu_sum", "row_mean", "row_rmsnorm", "row_softmax"]

        def reference(x, family):
            if family == "row_sum": return x.sum(1)
            if family == "row_sumsq": return (x * x).sum(1)
            if family == "row_relu_sum": return x.clamp_min(0).sum(1)
            if family == "row_mean": return x.mean(1)
            if family == "row_rmsnorm": return x * torch.rsqrt((x * x).mean(1, keepdim=True) + 1e-5)
            return torch.softmax(x, dim=1)

        def make_input(task, seed, distribution):
            rows, cols = task["rows"], task["cols"]
            stride = task["stride"][1]
            g = torch.Generator(device="cuda").manual_seed(seed)
            backing = torch.randn((rows, cols * stride), generator=g, device="cuda")
            x = backing[:, ::stride]
            if distribution == "uniform": x.uniform_(-1, 1, generator=g)
            elif distribution == "zero": x.zero_()
            elif distribution == "impulse":
                x.zero_()
                x[:, cols // 2] = 7
            elif distribution == "alternating":
                x.fill_(1)
                x[:, 1::2] = -1
            return x

        for task in selected:
            task_start = time.monotonic()
            family = task["family"]
            op = operations.index(family)
            eager = lambda x, family=family: reference(x, family)
            methods = {"eager": eager, "compile": torch.compile(eager, fullgraph=True),
                       "block128": lambda x, op=op: ext.row_probe(x, op, 128),
                       "block256": lambda x, op=op: ext.row_probe(x, op, 256)}
            row = {"task_id": task["task_id"], "family": family, "methods": {}}
            # Non-default stream also checks that the extension respects PyTorch's stream.
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                for name, fn in methods.items():
                    try:
                        for seed in task["input_seeds"]:
                            for distribution in task["distributions"]:
                                x = make_input(task, seed, distribution)
                                before = x.clone()
                                expected = reference(x.double(), family).float()
                                actual = fn(x)
                                if actual.shape != expected.shape or actual.dtype != expected.dtype:
                                    raise AssertionError("shape or dtype mismatch")
                                torch.testing.assert_close(actual, expected, rtol=task["rtol"], atol=task["atol"])
                                torch.testing.assert_close(x, before, rtol=0, atol=0)
                        row["methods"][name] = {"status": "correct", "correctness_cases": 15, "samples_ms": []}
                    except Exception as error:
                        row["methods"][name] = {"status": "failed", "error": str(error)}
                x = make_input(task, 17, "normal")
                good = [name for name in methods if row["methods"][name]["status"] == "correct"]
                for name in good:
                    for _ in range(args.warmup): methods[name](x)
                stream.synchronize()
                rng = random.Random(17)
                row["timing_order"] = []
                for _ in range(args.repeats):
                    order = good.copy()
                    rng.shuffle(order)
                    row["timing_order"].append(order)
                    for name in order:
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        result = methods[name](x)
                        end.record()
                        end.synchronize()
                        row["methods"][name]["samples_ms"].append(begin.elapsed_time(end))
                for name in good:
                    row["methods"][name]["median_ms"] = statistics.median(row["methods"][name]["samples_ms"])
            baseline_names = [n for n in ("eager", "compile") if n in good]
            row["baseline_complete"] = len(baseline_names) == 2
            row["baseline_name"] = min(baseline_names, key=lambda n: row["methods"][n]["median_ms"]) if baseline_names else None
            row["wall_seconds"] = time.monotonic() - task_start
            write_new(out / (task["task_id"] + ".json"), json.dumps(row, indent=2) + "\n")
        meta["status"] = "probe_completed_inspect_per_method_results"
    except Exception:
        meta["status"] = "failed"
        meta["error"] = traceback.format_exc()
        raise
    finally:
        meta["total_wall_seconds"] = time.monotonic() - started
        write_new(out / "completion.json", json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
