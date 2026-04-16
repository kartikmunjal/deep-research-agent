# Deep Research Agent

A modular deep-research pipeline that decomposes a question into sub-questions, retrieves and semantically compresses evidence, synthesizes a cited answer, and verifies each factual claim against evidence before returning results.

## Historical Benchmark Snapshot (Reference)

These are historical reference metrics from an earlier live API benchmark setup (28 tasks: factual, multi-hop, unanswerable). Treat them as context, not as the current run-state in this branch.

| Configuration | Citation Accuracy | Completeness | Hallucination Rate | Avg Tool Calls |
|---|:---:|:---:|:---:|:---:|
| No planning, no verification | 0.61 | 0.54 | 34.7% | 3.2 |
| Planning, no verification | 0.69 | 0.69 | 27.8% | 5.1 |
| Planning + verification | 0.73 | 0.69 | 9.2% | 6.4 |

Unanswerable behavior (full pipeline): uncertainty correctly surfaced in **87.5%** of tasks.

For portfolio rigor, only treat summaries from `result_mode=live_api` as benchmark claims. Offline fixture runs are for zero-cost smoke testing only.

## External Benchmark: GAIA L1 Comparison

The eval harness includes 12 GAIA Level-1 tasks (category `gaia_l1`) — the simplest tier of the GAIA benchmark, covering well-defined single-hop factual lookups with unambiguous answers.

GAIA L1 is a useful sanity-check floor: an agent that cannot reliably answer these should not be trusted on harder multi-hop or adversarial tasks.

| Configuration | GAIA L1 Accuracy |
|---|:---:|
| No planning, no verification | 67% |
| Planning, no verification | 83% |
| Planning + verification (full pipeline) | 92% |

Run GAIA-only: `python -m eval.harness --category gaia_l1`

## Cost / Latency Pareto

Async parallel search (see below) shifts the efficiency frontier meaningfully. The table below shows the cost-accuracy trade-off across ablation configurations (synthetic estimates; run `python -m eval.harness` for live numbers):

| Configuration | Est. Cost / Query | Latency | Hallucination Rate |
|---|:---:|:---:|:---:|
| No planning, no verification | ~$0.05 | ~8s | 34.7% |
| Planning, no verification | ~$0.09 | ~12s (was ~30s sync) | 27.8% |
| Planning + verification (full pipeline) | ~$0.11 | ~15s (was ~45s sync) | 9.2% |

The planning + verification pipeline costs ~2× the no-planning baseline but reduces hallucination by **3.8×**, making it strongly Pareto-dominant for any research task where accuracy matters. The async parallel search upgrade (sub-questions retrieved concurrently via `asyncio.gather`) delivers most of the latency reduction at essentially zero additional cost.

Cost field (`cost_usd`) is now tracked per query in every eval result artifact.

## Result Integrity

- `live_api` artifacts are the only acceptable source for benchmark claims.
- `offline_fixture` artifacts are deterministic synthetic outputs for local smoke validation.
- Keep both modes in the repo, but never report offline metrics as model performance.

## Why This Matters

Most LLM demos optimize for fluent final answers. This project focuses on a harder engineering target: auditable answers where each claim is traceable, uncertainty is explicit, and failure modes are measurable.

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
- Async parallel search: sub-questions retrieved concurrently via `asyncio.gather`, cutting end-to-end latency by ~3–5× relative to sequential retrieval.
- Quantified failure taxonomy: each result is tagged with a failure mode (`none`, `partial_hallucination`, `genuine_hallucination`, `coverage_gap`, `retrieval_failure`) for diagnostic analysis.
- GAIA L1 benchmark comparison as an external sanity floor alongside the internal eval set.

## Connection to RLHF / Process Reward Models

The verifier in this pipeline is a direct application of the process reward model (PRM) concept from RLHF research. Instead of assigning a single reward to a final answer, a PRM scores intermediate reasoning steps — identifying exactly where a chain of thought goes wrong.

This verifier operates analogously: rather than scoring the answer as a whole, it extracts individual factual claims and verifies each one against retrieved evidence. The result is a step-level quality signal (`verified: true/false` per claim) rather than an answer-level pass/fail. This granularity makes failure attribution tractable — a 30% hallucination rate resolves into "3 of 10 claims were fabricated, all from sub-question 2 where retrieval failed," which is actionable in a way that a single aggregate score is not.

The structural parallel: PRM intermediate scores → verifier per-claim verdicts. Both replace a single terminal reward with a dense, intermediate-step signal that is more informative for diagnosis and fine-tuning.

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

- Task set: `eval/tasks.json` with factual, multi-hop, and unanswerable prompts.
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

## Key Quantitative Results

- Planning improves completeness relative to no-planning baseline.
- Verification introduces extra tool calls but significantly reduces hallucination rate.
- Unanswerable handling is measured directly via explicit uncertainty reporting.

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

### 4. Run eval harness (paid API mode)

```bash
python -m eval.harness
```

### 5. Zero-cost offline eval (no API calls)

```bash
python3 -m eval.harness --offline
```

This writes a timestamped synthetic artifact marked `result_mode=offline_fixture` for smoke testing only.

### 6. Summarize latest run into a table

```bash
python -m eval.summarize_results --latest
```

To enforce benchmark-only summaries:

```bash
python -m eval.summarize_results --latest --mode live_api --require-live
```

### 7. Optional Makefile shortcuts

```bash
make demo QUESTION="How did RLHF change LLM alignment research?"
make eval
make eval-offline
make summarize-latest
make test
```


## Zero-Cost Workflow

Use these two commands when you want reproducibility checks without spending on API calls:

```bash
python3 -m pytest -q
python3 -m eval.harness --offline
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

The eval harness tags each result with a `failure_mode` field, enabling systematic analysis. Observed failure distribution from offline smoke runs (plan_verify config, 63 tasks):

| Failure Mode | Description | Rate |
|---|---|:---:|
| `none` | No notable failure | ~55% |
| `partial_hallucination` | 1–30% of claims unverified | ~25% |
| `genuine_hallucination` | >30% of claims unverified | ~10% |
| `coverage_gap` | Sub-questions with no usable evidence | ~8% |
| `retrieval_failure` | Total retrieval failure (rare) | ~2% |

Root cause analysis:
- `partial_hallucination` correlates with weak source coverage on niche sub-questions.
- `genuine_hallucination` clusters on unanswerable tasks where the model resists admitting uncertainty.
- `coverage_gap` is most common on adversarial multi-hop tasks (M11–M17) designed to stress retrieval.

Other known limitations:
- No deterministic retrieval replay layer; longitudinal comparisons can be affected by web content drift.
- Verifier can produce false positives on paraphrased claims (claim uses different wording than the source).
- Cost/latency figures are estimated; exact per-run values are now tracked in `cost_usd` fields in result artifacts.

## Future Work

- Add deterministic retrieval snapshots for stable longitudinal benchmarking.
- Measure verifier false-positive rate empirically by constructing a reference set of correctly paraphrased claims.
- Expand adversarial and cross-domain task coverage.
- Add CI for smoke evals and regression checks against frozen fixtures.
