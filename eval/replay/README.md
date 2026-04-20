# Replay Benchmarks

This folder is for frozen benchmark artifacts that can be replayed with:

```bash
python -m eval.harness --offline --replay-benchmark eval/replay/<benchmark>.json
```

Intended workflow:

1. Run a live benchmark once:
   `python -m eval.harness --profile core_live_28`
2. Freeze that run:
   `python -m eval.freeze_benchmark --run-file eval/results/<run>.json --output eval/replay/core_live_28.json --benchmark-name core_live_28`
3. Replay it deterministically:
   `python -m eval.harness --offline --replay-benchmark eval/replay/core_live_28.json`

Guardrails:

- A replay benchmark is benchmark-claimable only if it was frozen from a benchmark-claimable source artifact.
- Synthetic smoke runs can be frozen only with `--allow-nonlive`, and should never be presented as research results.
- Replay mode replays frozen answers and recomputes scores deterministically; it does not make web or model calls.
