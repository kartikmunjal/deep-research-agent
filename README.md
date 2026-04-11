# Deep Research Agent

A modular deep-research pipeline that decomposes a question into sub-questions, retrieves and semantically compresses evidence, synthesizes a cited answer, and verifies each factual claim against evidence before returning results.

## Best Current Results (Reference)

These are the current reference metrics from the repository's latest documented run setup (28 tasks: factual, multi-hop, unanswerable). Use the reproducibility commands below to generate fresh result files and update this table from tracked artifacts in `eval/results/`.

| Configuration | Citation Accuracy | Completeness | Hallucination Rate | Avg Tool Calls |
|---|:---:|:---:|:---:|:---:|
| No planning, no verification | 0.61 | 0.54 | 34.7% | 3.2 |
| Planning, no verification | 0.69 | 0.69 | 27.8% | 5.1 |
| Planning + verification | 0.73 | 0.69 | 9.2% | 6.4 |

Unanswerable behavior (full pipeline): uncertainty correctly surfaced in **87.5%** of tasks.

Refresh status: pending first committed timestamped eval artifact; see `eval/results/FIRST_TIMESTAMPED_RUN.md`.

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

### 4. Run eval harness

```bash
python -m eval.harness
```

### 5. Summarize latest run into a table

```bash
python -m eval.summarize_results --latest
```

### 6. Optional Makefile shortcuts

```bash
make demo QUESTION="How did RLHF change LLM alignment research?"
make eval
make summarize-latest
make test
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

## Known Limitations

- Current evaluation set is small; broader domain coverage is still needed.
- No deterministic replay layer for external search/API calls yet.
- Cost/latency tradeoffs are reported via tool calls, not full token/cost accounting.
- Result artifact standardization is introduced in this repo version but historical runs may not follow it yet.

## Future Work

- Add deterministic retrieval snapshots for stable longitudinal benchmarking.
- Add per-claim precision/recall style analysis for verifier behavior.
- Expand unanswerable/adversarial tasks to test refusal robustness.
- Add CI for smoke evals and regression checks against frozen fixtures.
