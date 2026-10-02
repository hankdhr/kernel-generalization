"""Save one raw model response. Does not execute generated code or constitute an agent."""
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kernel_generalization.core import read_jsonl, validate_tasks, write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["qwen3_8b", "qwen25_coder_7b"], default="qwen3_8b")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--out", required=True, help="New directory")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    tasks = read_jsonl(ROOT / "data/pilot-v0.jsonl")
    validate_tasks(tasks)
    task = next((t for t in tasks if t["task_id"] == args.task_id), None)
    if task is None: parser.error("unknown task")
    if task["split"] == "test": parser.error("pilot final-test generation is deferred until evaluation worker and protocol are frozen")
    source = json.loads((ROOT / "configs/sources.lock.json").read_text())["sources"][args.model]
    config = json.loads((ROOT / "configs/experiment.json").read_text())["generation"]
    prompt = (
        "Implement the following operation in CUDA C++ using a PyTorch C++ extension. "
        "Return one complete .cu source file with a PYBIND11_MODULE exposing run(torch::Tensor x). "
        "Use the current PyTorch CUDA stream and device. Return a fresh contiguous float32 output. "
        "Do not modify input, use high-level tensor operations for computation, or read files or the network. "
        "Support the specified positive input strides. Numerical contract: finite inputs, "
        "rtol=1e-4, atol=2e-4 against a float64 reference cast to float32. "
        "No timing or correctness test code is needed.\nTask:\n" + json.dumps({k: task[k] for k in
            ("family", "rows", "cols", "dtype", "stride", "semantics")}, indent=2)
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "prompt.txt", prompt)
    meta = {"kind": "single_generation", "status": "planned", "task_id": task["task_id"],
            "model": source, "generation": config, "seed": args.seed,
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    write_new(out / "plan.json", json.dumps(meta, indent=2) + "\n")
    if args.plan_only:
        print("Prompt saved. No model downloaded or called.")
        return
    started = time.monotonic()
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        if not torch.cuda.is_available(): raise RuntimeError("CUDA required by this initial generator")
        set_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(source["id"], revision=source["revision"], trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(source["id"], revision=source["revision"],
                   torch_dtype=torch.bfloat16, trust_remote_code=False).to("cuda").eval()
        chat_kwargs = {"enable_thinking": config["enable_thinking"]} if args.model == "qwen3_8b" else {}
        chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True, **chat_kwargs)
        write_new(out / "rendered_prompt.txt", chat)
        inputs = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to("cuda")
        torch.cuda.synchronize()
        generation_start = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=True, temperature=config["temperature"],
                     top_p=config["top_p"], top_k=config["top_k"], max_new_tokens=config["max_new_tokens"])
        torch.cuda.synchronize()
        tokens = output[0, inputs.input_ids.shape[1]:]
        response = tokenizer.decode(tokens, skip_special_tokens=False)
        write_new(out / "response.txt", response)
        write_new(out / "token_ids.json", json.dumps(tokens.tolist()) + "\n")
        meta.update(status="generated_not_evaluated", generation_seconds=time.monotonic() - generation_start,
                    prompt_tokens=inputs.input_ids.shape[1], generated_tokens=len(tokens),
                    reached_token_limit=len(tokens) == config["max_new_tokens"],
                    torch_version=torch.__version__, transformers_version=transformers.__version__,
                    device=torch.cuda.get_device_name(0),
                    response_sha256=hashlib.sha256(response.encode()).hexdigest())
    except Exception:
        meta.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        meta["total_wall_seconds"] = time.monotonic() - started
        write_new(out / "completion.json", json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
