"""Resolve public model/repository tips into a NEW proposed lock; never update runs.

Requires network access. Review the diff before adopting a new source revision.
"""
import argparse
import concurrent.futures
import datetime
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kernel_generalization.core import write_new


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "kernel-generalization-research"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def one(item):
    key, source = item
    kind, identifier = source["kind"], source["id"]
    url = ("https://huggingface.co/api/models/" + identifier if kind == "model"
           else "https://api.github.com/repos/" + identifier + "/commits/main")
    data = fetch(url)
    result = {"kind": kind, "id": identifier, "revision": data["sha"], "metadata_url": url}
    if kind == "model":
        result["license"] = data.get("cardData", {}).get("license", "unverified")
    else:
        info = fetch("https://api.github.com/repos/" + identifier)
        result["license"] = (info.get("license") or {}).get("spdx_id", "unverified")
    return key, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists(): parser.error("output exists; use a new path")
    current = json.loads((ROOT / "configs/sources.lock.json").read_text())["sources"]
    result = {"retrieved_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "sources": {}}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for key, source in pool.map(one, current.items()):
            result["sources"][key] = source
    write_new(args.out, json.dumps(result, indent=2) + "\n")
    print(f"Proposed source lock written to {args.out}")


if __name__ == "__main__":
    main()
