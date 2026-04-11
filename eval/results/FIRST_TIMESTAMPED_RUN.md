# First Timestamped Eval Run

Status: pending.

A real run artifact has not been committed yet because this environment does not currently have `ANTHROPIC_API_KEY` and `TAVILY_API_KEY` configured.

## When keys are available

Run:

```bash
python3 -m eval.harness
python3 -m eval.summarize_results --latest --export-json eval/results/latest_summary.json
```

Then:
- commit the generated timestamped file from `eval/results/` (for example `20260411T173000Z.json`),
- refresh the top results table in `README.md` from that artifact,
- update this file with the run id, date, model, and commit hash.
