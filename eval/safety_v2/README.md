# Preregistered Safety Evaluation v2

V2 preserves the v1 pilot and corrects its main construct-validity limitation:
direct attacks are actual published JailbreakBench artifact prompts rather than
bare harmful behavior goals.

The fixed dataset contains 80 direct jailbreaks (20 each from PAIR, GCG,
JailbreakChat, and random-search artifacts), 30 indirect prompt injections, and
60 benign controls. Direct prompts are category-balanced across all ten JBB
policy categories and selected by source index without using attack outcomes.

The primary metric is **guardrail-bypass rate (GBR)**: attacks allowed past the
guardrail / attack trials. It is deliberately not called end-to-end ASR because
this experiment does not ask the downstream model to generate harmful content.
FPR is benign refusals / benign trials. Every rate includes event counts,
`N_trials`, and a Wilson 95% interval. Before/after comparisons use a two-sided
exact McNemar test on paired examples.

```bash
source .venv/bin/activate
python3 scripts/build_safety_eval_v2_dataset.py
python3 -m eval.safety_v2.harness --dry-run
python3 -m eval.safety_v2.harness --max-cost-usd 3
python3 -m eval.safety_v2.report eval/safety_v2/results/safety_v2_<timestamp>.json
```

The harness checkpoints every paid decision. Resume an interrupted run with:

```bash
python3 -m eval.safety_v2.harness --max-cost-usd 3 --resume <artifact.json>
```
