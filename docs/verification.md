# Local verification — 2026-10-02

Environment: macOS ARM64, Python 3.12.9. No local CUDA GPU/toolchain, PyTorch, or model weights were used. This is infrastructure verification, not an experimental result.

| Check | Observed result |
|---|---|
| `python -m unittest discover -s tests -v` | 13 tests passed |
| Python compileall for package, scripts and tests | Passed |
| Regenerated pilot manifest and validated hashes/splits | 108 tasks: train 54, dev 18, test 36 |
| GPU plan entrypoint with Python `-S` | Passed without third-party packages; no timing results produced |
| Model prompt planning with Python `-S` | Passed without loading/calling a model |
| Missing GPU dependency failure path | Nonzero exit; plan and failure completion retained |
| Final-test generation gate | Test task rejected before generation |
| Local inventory | nvidia-smi, nvcc and ncu unavailable |

Canonical manifest SHA256:

```text
7db277443abf66b685f21b297e13ed69d1805506ed1ecc7231ec106c2bba38f5
```

Tests cover: deterministic task construction; duplicate/tampered task rejection; family split leakage; output overwrite protection; missing/failed-task denominator; each resource budget; strict speed threshold; invalid/nonfinite measurements; mixed methods/seeds; cumulative cost accounting; and dependency-free planning/failure logging.

**Not verified:** CUDA compilation or numerical correctness, stream behavior on hardware, benchmark performance, model loading/generation, LoRA training, isolated candidate execution, public benchmark integration, remote CI execution. Scripts and plans for these are not evidence that they work on a GPU.
