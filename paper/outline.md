# Working paper outline — proposal, no results yet

Working title: **Learning When Kernel Optimizations Work: Transfer from Controlled Program Transformations**

The title names the question; it does not claim a proven contribution. Revise it after novelty review and experiments.

## 1. Introduction

Problem: expensive repeated trial-and-error on unseen kernel tasks. Question: can measured applicability conditions become transferable model knowledge? Distinguish parameter learning, retrieved experience, and test-time search.

## 2. Related work

Kernel RL (CUDA Agent, Kevin, Dr. Kernel), task synthesis (DRTriton), profiling/search (KernelPro), experience-based optimization (AccelOpt), public evaluation (KernelBench, FlashInferBench). Identify precise overlap before claiming novelty.

## 3. Method

Task/implementation conditions; controlled source transformations; paired correctness and latency observations; uncertainty-aware labels; training objective. Document failed transformations and assumptions. Optional active collection belongs here only if it shows an independent effect.

## 4. Experimental design

Immutable model/source pins, data construction and splits, contamination limits, numerical contracts, measurement procedures, candidate/token/time budgets, equal-data and equal-compute controls, training and generation seeds.

## 5. Results — populate only after execution

- Main table: correctness and fast_p on frozen held-out tasks, same agent/tools/budget.
- Figure: performance versus search budget, plus total training/data cost.
- Mechanism: condition prediction, shuffled labels, failure-example removal, raw trajectory SFT.
- Transfer: graph composition, shape/layout, another model; hardware only if tested.
- Case study: real operator versus pinned expert implementation.
- Negative results and uncertainty.

## 6. Limitations

Public benchmark contamination, limited task families/hardware, measurement sensitivity, model and teacher dependence, scope of evidence. Kernel-domain improvement alone is not evidence of general intelligence growth or unbounded recursive self-improvement.

## 7. Reproducibility and release

Link source, environment lock, full experimental ledger, data/trajectory archives, models/adapters, evaluation and plotting commands. Every reported number should resolve to a run and an analysis script.

## Evidence ledger

| Intended claim | Required evidence | Current status |
|---|---|---|
| Conditions improve transfer | E03/E04 plus frozen held-out evaluation | Unmeasured |
| Failures carry useful information | E05 removal/shuffle controls | Unmeasured |
| Improvement resides partly in weights | Retrieval-off fixed-agent evaluation | Unmeasured |
| Practical kernel benefit | E08 repeated expert-baseline comparison | Unmeasured |
| Iterative self-improvement | E09 fixed-collector/equal-cost controls | Unmeasured |
