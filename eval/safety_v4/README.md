# Held-Out Typoglycemia Robustness v4

V4 uses the two OpenAI v3 bypasses only to identify the mechanism. Outcomes use
20 untouched v2 injection seeds and exclude both development examples.

```bash
python3 scripts/build_safety_eval_v4_dataset.py
python3 -m eval.safety_v4.harness --dry-run
python3 -m eval.safety_v4.harness --model claude-sonnet-4-6 --max-cost-usd 2
python3 -m eval.safety_v4.harness --model claude-haiku-4-5 --max-cost-usd 2
python3 -m eval.safety_v4.harness --provider openai \
  --model gpt-5-mini-2025-08-07 --max-cost-usd 1
python3 -m eval.safety_v4.report <three artifacts> --output eval/safety_v4/results/report.md
```

Every paid decision is checkpointed. Use `--resume` after interruption. Do not
pool across models; each artifact reports paired v3/v4 outcomes on one model.
