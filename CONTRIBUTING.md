# Contributing

Before changing a hypothesis, split, metric, or budget, explain the reason in a new research note and update the experiment ledger. Preserve old run artifacts and protocol revisions. Do not change final-test criteria after inspecting results without labeling the follow-up as exploratory.

Run the CPU checks from the repository root:

```bash
python -m unittest discover -s tests -v
python -m kernel_generalization check data/pilot-v0.jsonl
```

CUDA changes need actual GPU compilation, correctness validation, and measurements in a documented environment. CPU tests do not establish CUDA correctness or performance. Record unverified changes explicitly.

Never commit credentials or private machine information. Keep large run artifacts in versioned research releases with checksums, linked from experiment-ledger.csv. Retain failed runs and all candidate costs.
