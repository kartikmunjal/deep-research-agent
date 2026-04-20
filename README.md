# Deep Research Agent

A modular deep-research pipeline that decomposes a question into sub-questions, retrieves and semantically compresses evidence, synthesizes a cited answer, and verifies each factual claim against evidence before returning results.

## Current Evidence Status

This branch now separates three kinds of evidence clearly:
- `live_api`: one-shot live runs used to generate frozen benchmark artifacts,
- `replay_benchmark`: deterministic zero-cost reruns of frozen artifacts,
- `offline_fixture`: synthetic smoke validation only.

That architecture is the intended research path for this repo: freeze one live capture, then do all subsequent measurement and reporting on deterministic replay artifacts.

What is implemented:
- planning, retrieval/compression, synthesis, and claim-level verification,
- ablation configs in a single eval harness,
- named benchmark profiles for reproducible live runs,
- replay benchmark freezing and deterministic reruns,
- async parallel retrieval in the pipeline,
- failure-mode tagging in the result schema,
- a small GAIA L1 subset in the local task file.

What is **not** independently verified in the current branch:
- the earlier 28-task live benchmark snapshot,
- any official GAIA validation-set score,
- a checked-in frozen replay benchmark generated from a live run,
- verifier false-rejection rate on supported paraphrases,
- measured async-vs-serial latency deltas from frozen timing artifacts,
- any experiment showing verifier signals improve the agent the way a PRM-style training signal would.

For rigor, treat `replay_benchmark` artifacts as the preferred reportable benchmark surface once they are frozen from a claimable source run. `offline_fixture` remains smoke-test only.

## GAIA Status

The eval harness currently embeds 12 public GAIA Level-1 tasks under category `gaia_l1`. That subset is useful for local sanity checks, but it is not the official GAIA validation benchmark and should not be reported as such.

The repo does **not** currently report a reproducible score on the full GAIA Level-1 validation set (165 tasks). Running and reporting that official number is still open work.

Run the local subset only:

```bash
python -m eval.harness --category gaia_l1
```

If you cite GAIA results from this repo, distinguish clearly between:
- the 12-task local subset in `eval/tasks.json`, and
- the official GAIA validation set.

## Cost / Latency Status

The async implementation is real: sub-question retrieval is executed concurrently via `asyncio.gather`. What is missing in this branch is a published apples-to-apples benchmark comparing serial and async execution on the same prompts.

`cost_usd` is tracked per query in live artifacts. Async latency can now be measured from frozen timing fields in replayable artifacts, but the repo still needs one frozen benchmark file before publishing numbers.

## Result Integrity

- `live_api` artifacts are for capture.
- `replay_benchmark` artifacts are for reproducible reporting.
- `offline_fixture` artifacts are deterministic synthetic outputs for smoke validation only.

## Why This Matters

Most LLM demos optimize for fluent final answers. This project focuses on a harder engineering target: auditable answers where each claim is traceable, uncertainty is explicit, and failure modes are measurable.

## Portfolio Position

This repo is the agent-evaluation branch of the reward-methodology work in
[`rlhf-and-reward-modelling-alt`](https://github.com/kartikmunjal/rlhf-and-reward-modelling-alt).
The verifier is not claimed as a trained PRM; it is a process-style diagnostic
signal that labels individual factual claims as supported or unsupported.

Related repos:
- [`rl-env`](https://github.com/kartikmunjal/rl-env): controlled reward-hacking
  experiments in a schema-grounded SQL MDP.
- [`rlhf-and-reward-modelling-alt`](https://github.com/kartikmunjal/rlhf-and-reward-modelling-alt):
  reward model design, PPO/DPO, ensemble rewards, calibration, and agent benchmarks.

## Problem Statement

Complex research questions are multi-hop and high-risk for hallucination. A practical research agent must:
- preserve coverage across sub-topics,
- keep context manageable when many sources are retrieved,
- ground claims in evidence rather than surface-level plausibility,
- and fail safely when evidence is missing.

## System Architecture

```
Question
    -> Planner (3-5 independently searchable sub-questions)
    -> Searcher (web retrieval + semantic compression)
    -> Synthesizer (structured answer with inline [N] citations)
    -> Verifier (claim extraction + claim-level grounding)
    -> Final answer + unverified claim report + coverage gaps
```

Core modules:
- `planner`: decomposition strategy
- `searcher`: Tavily retrieval and per-source evidence extraction
- `synthesizer`: evidence-to-answer generation with citations
- `verifier`: claim-level verification against extracted evidence
- `pipeline`: orchestration, ablations, and failure recovery surfacing

## What Is Different Here

- Claim-level grounding, not only answer-level scoring.
- Semantic compression before synthesis to improve context efficiency.
- Explicit failure recovery path with retry/reformulation and coverage-gap reporting.
- Evaluation harness with ablations (`no_plan_no_verify`, `plan_no_verify`, `plan_verify`) to isolate component impact.
- Async parallel search: sub-questions are retrieved concurrently via `asyncio.gather`; a measured serial-vs-async benchmark still needs to be published.
- Failure taxonomy instrumentation: each result can be tagged with a failure mode (`none`, `partial_hallucination`, `genuine_hallucination`, `coverage_gap`, `retrieval_failure`) for diagnostic analysis.
- Local GAIA L1 subset support as a sanity-check harness, distinct from the official GAIA validation benchmark.

## Conceptual Connection to RLHF / Process Reward Models

The verifier in this pipeline is best understood as a conceptual analog of a process reward model (PRM), not as an empirical PRM result. Instead of assigning a single reward to a final answer, a PRM scores intermediate reasoning steps and helps localize where reasoning fails.

This verifier operates analogously: rather than scoring the answer as a whole, it extracts individual factual claims and verifies each one against retrieved evidence. The result is a step-level quality signal (`verified: true/false` per claim) rather than an answer-level pass/fail. This granularity makes failure attribution tractable — a 30% hallucination rate resolves into "3 of 10 claims were fabricated, all from sub-question 2 where retrieval failed," which is actionable in a way that a single aggregate score is not.

The structural parallel is: PRM intermediate scores -> verifier per-claim verdicts. Both replace a single terminal reward with a denser intermediate signal that is more informative for diagnosis.

What is not shown in this repo yet is the stronger empirical claim: using verifier verdicts as a training or selection signal to improve later agent behavior. Until that experiment exists, the PRM connection should be read as framing, not as a demonstrated result.

## Claim-Level Grounding vs Answer-Level Scoring (Plain English)

- Answer-level scoring asks: "Is the overall answer mostly right?"
- Claim-level grounding asks: "For each specific statement, can we point to supporting evidence?"

A response can look mostly correct while containing one fabricated detail. Claim-level verification is designed to catch that single bad claim instead of letting it hide inside a good-sounding paragraph.

## Semantic Compression and Failure Recovery

Semantic compression example:
- Without compression: 4 sub-questions x 4 results can produce many long articles, quickly flooding context.
- With compression: each source is reduced to a few directly relevant sentences; synthesis and verification operate on high-signal evidence.

Failure recovery example:
- If a sub-question search returns no usable extracts, the searcher reformulates the query and retries once.
- If retry still fails, the sub-question is explicitly marked unresolved and surfaced as a coverage gap (instead of being silently ignored or hallucinated).

## Evaluation Methodology

- Task set: `eval/tasks.json` with factual, multi-hop, unanswerable, conflicting-evidence, and local GAIA L1 prompts.
- Named benchmark profiles in `eval/benchmark_profiles.py` for reproducible filtered runs.
- Configurations:
  - `no_plan_no_verify`
  - `plan_no_verify`
  - `plan_verify`
- Primary metrics:
  - `citation_accuracy`
  - `completeness`
  - `hallucination_rate`
  - `uncertainty_reported` (unanswerable category)

See `eval/results/README.md` for metric definitions and interpretation.

For the two `*_no_verify` ablations, claim-level metrics are still computed during evaluation via a post-hoc verifier pass. That keeps hallucination/citation scoring comparable across configurations without changing the pipeline behavior of those ablations.

### Task taxonomy and selection

The internal task set is hand-authored and curated for diagnostic coverage; it is not a random sample from a benchmark distribution.

- `factual`: direct, well-scoped questions whose key facts should be recoverable from standard sources without nontrivial composition.
- `multi_hop`: questions whose expected answer requires connecting at least two linked facts, sources, or historical steps; several are explicitly marked adversarial.
- `unanswerable`: prompts intentionally lacking a reliable public answer, used to test abstention and uncertainty signaling.
- `conflicting_evidence`: prompts where credible sources disagree and the intended behavior is adjudication or explicit acknowledgement of disagreement.
- `gaia_l1`: a 12-task local subset copied into the harness for sanity checks only.

Because the tasks are curated, internal scores should be interpreted as controlled research probes into agent behavior, not as claims about general-purpose performance.

Named profiles currently included:
- `core_live_28`: budget-bounded live benchmark (10 factual, 10 multi-hop, 8 unanswerable).
- `internal_full_51`: all internal non-GAIA tasks.
- `gaia_local_12`: the local GAIA subset only.

## Key Quantitative Results

- The harness is set up to compare planning and verification ablations on the same tasks.
- The current checked-in results are not yet a frozen replay benchmark generated from a live run, so this branch still lacks publishable benchmark numbers.
- The intended reporting path is:
  1. capture one live run,
  2. freeze it with `eval.freeze_benchmark`,
  3. cite the resulting `replay_benchmark` artifact.

## Failure Cases / Limitations

- Weak or noisy retrieval can propagate to synthesis quality.
- Citation numbers indicate source mapping, but semantic support quality depends on extraction quality.
- Verification uses model judgments and can produce false positives/negatives on nuanced claims.
- API and web variance can shift results between runs.

## Reproducibility

### 1. Setup

```bash
git clone https://github.com/kartikmunjal/deep-research-agent
cd deep-research-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure keys

```bash
cp .env.example .env
# Fill ANTHROPIC_API_KEY and TAVILY_API_KEY
```

### 3. Run demo

```bash
python demo.py "How did RLHF change LLM alignment research?"
```

### 4. Capture a live run once

```bash
python -m eval.harness --profile core_live_28
```

### 5. Freeze that run into a replay benchmark

```bash
python -m eval.freeze_benchmark \
  --run-file eval/results/<live-run>.json \
  --output eval/replay/core_live_28.json \
  --benchmark-name core_live_28
```

### 6. Zero-cost replay benchmark (no API calls)

```bash
python3 -m eval.harness --offline --replay-benchmark eval/replay/core_live_28.json
```

This writes a timestamped artifact marked `result_mode=replay_benchmark`.

### 7. Summarize latest run into a table

```bash
python -m eval.summarize_results --latest
```

To enforce benchmark-claimable summaries:

```bash
python -m eval.summarize_results --latest --mode replay_benchmark --require-benchmarkable
```

### 8. Measure verifier rejection on supported paraphrases

Export a 20-claim reference template from a benchmark-claimable live or replay artifact:

```bash
python -m eval.measure_verifier_fpr \
  --run-file eval/results/<benchmarkable-run>.json \
  --export-template eval/results/verifier_fpr_template.json
```

Then fill `paraphrased_claim` for each example and measure:

```bash
python -m eval.measure_verifier_fpr \
  --reference-file eval/results/verifier_fpr_template.json \
  --output eval/results/verifier_fpr_results.json
```

### 9. Measure async search savings from frozen timings

```bash
python -m eval.measure_replay_latency \
  --file eval/results/<replay-run>.json \
  --output eval/results/latency_summary.json
```

### 10. Synthetic smoke mode for CI only

```bash
python -m eval.harness --synthetic-smoke
```

### 11. Optional Makefile shortcuts

```bash
make demo QUESTION="How did RLHF change LLM alignment research?"
make eval
make eval-offline
make eval-smoke
make summarize-latest
make summarize-benchmark
make test
```


## Zero-Cost Workflow

Use these commands when you want reproducibility checks without spending on API calls:

```bash
python3 -m pytest -q
python3 -m eval.harness --offline --replay-benchmark eval/replay/core_live_28.json
python3 -m eval.measure_verifier_fpr --reference-file eval/results/verifier_fpr_template.json
```

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env
python demo.py "What changed in post-training after RLHF?"
```

## Repo Structure

```
src/agent/
  planner.py
  searcher.py
  synthesizer.py
  verifier.py
  pipeline.py
  models.py

eval/
  harness.py
  scoring.py
  summarize_results.py
  tasks.json
  results/
    README.md
    result_template.json.example

tests/
  test_core_behavior.py

demo.py
Makefile
pyproject.toml
```

## Known Limitations and Failure Mode Taxonomy

The eval harness tags each result with a `failure_mode` field, enabling systematic analysis. The replay path is now implemented, but the repo still needs a frozen benchmark artifact before it can report a reproducible failure distribution.

Failure modes currently represented in the schema:
- `none`: no notable failure surfaced by the scorer.
- `partial_hallucination`: some claims are unsupported.
- `genuine_hallucination`: unsupported claims dominate the answer.
- `coverage_gap`: one or more sub-questions were left unresolved.
- `retrieval_failure`: the retrieval stage failed to supply usable evidence.

Other known limitations:
- Verifier rejection of supported paraphrases is currently unmeasured, which limits how strongly verified/unverified claim totals should be interpreted.
- Verifier can produce false positives on paraphrased claims (claim uses different wording than the source).
- The checked-in repo still lacks a frozen replay benchmark artifact generated from a live run.
- Async latency improvements can now be measured from frozen timing fields, but the repo does not yet publish those numbers.
- Cost fields originate from the live capture, while replay mode is zero-cost and deterministic.

## Future Work

- Run the full GAIA Level-1 validation set and report the official score separately from the 12-task local subset.
- Commit a frozen replay benchmark artifact generated from `core_live_28` and publish the reproducible numbers from that artifact.
- Measure and publish a failure taxonomy from the frozen replay benchmark.
- Measure verifier rejection on supported paraphrases empirically using the reference-set workflow in `eval.measure_verifier_fpr`.
- Publish async-vs-serial search timing from `eval.measure_replay_latency`.
- Expand adversarial and cross-domain task coverage.
- Add CI for smoke evals and regression checks against frozen fixtures.
- Test whether verifier verdicts can act as a useful training or selection signal, which would turn the PRM analogy into an empirical result (see [Extension 3: PRM vs ORM](https://github.com/kartikmunjal/rlhf-and-reward-modelling-alt#extension-3-process-reward-model-prm-vs-outcome-reward-model-orm)).
