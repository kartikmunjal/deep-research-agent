# Deep Research Agent

A modular deep-research pipeline that decomposes a question into sub-questions, retrieves and semantically compresses evidence, synthesizes a cited answer, and verifies each factual claim against evidence before returning results.

This repository runs hosted-model inference and evaluation; it does not train or
fine-tune model weights and does not require a GPU.

## Historical Pipeline Snapshot (Unverified Reference)

These numbers were retained from an earlier 28-task benchmark setup (factual,
multi-hop, and unanswerable). The current branch does not contain the original
`live_api` artifact needed to independently regenerate them; all committed
general-pipeline result files are marked `offline_fixture`. They are therefore
shown only as historical project context—not as validated findings from this
repository state. The safety studies below are the repository's current
reproducible live evidence.

| Configuration | Citation Accuracy | Completeness | Hallucination Rate | Avg Tool Calls |
|---|:---:|:---:|:---:|:---:|
| No planning, no verification | 0.61 | 0.54 | 34.7% | 3.2 |
| Planning, no verification | 0.69 | 0.69 | 27.8% | 5.1 |
| Planning + verification | 0.73 | 0.69 | 9.2% | 6.4 |

Unanswerable behavior (full pipeline): uncertainty correctly surfaced in **87.5%** of tasks.

For portfolio rigor, only treat summaries from `result_mode=live_api` as benchmark claims. Offline fixture runs are for zero-cost smoke testing only.

## Historical GAIA L1 Snapshot (Unverified Reference)

The eval harness includes 12 GAIA Level-1 tasks (category `gaia_l1`)—the
simplest tier of the GAIA benchmark, covering well-defined single-hop factual
lookups with unambiguous answers. As with the pipeline snapshot above, the
current branch does not contain a supporting `live_api` artifact, so these
percentages are historical context rather than current benchmark claims.

GAIA L1 is a useful sanity-check floor: an agent that cannot reliably answer these should not be trusted on harder multi-hop or adversarial tasks.

| Configuration | GAIA L1 Accuracy |
|---|:---:|
| No planning, no verification | 67% |
| Planning, no verification | 83% |
| Planning + verification (full pipeline) | 92% |

Run GAIA-only: `python3 -m eval.harness --category gaia_l1`

## Cost / Latency Pareto

Async parallel search (see below) shifts the efficiency frontier meaningfully. The table below shows the cost-accuracy trade-off across ablation configurations (synthetic estimates; run `python3 -m eval.harness` for live numbers):

| Configuration | Est. Cost / Query | Latency | Hallucination Rate |
|---|:---:|:---:|:---:|
| No planning, no verification | ~$0.05 | ~8s | 34.7% |
| Planning, no verification | ~$0.09 | ~12s (was ~30s sync) | 27.8% |
| Planning + verification (full pipeline) | ~$0.11 | ~15s (was ~45s sync) | 9.2% |

In the synthetic estimate, planning plus verification costs about 2× the
no-planning baseline while the recorded hallucination rate is 3.8× lower.
This is an engineering hypothesis, not a validated Pareto claim; it requires a
new `live_api` run with uncertainty estimates. Async parallel search
(sub-questions retrieved concurrently via `asyncio.gather`) is the proposed
latency mechanism.

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
    -> Constitutional Guardrail (user input)
    -> Planner (3-5 independently searchable sub-questions)
    -> Searcher (web retrieval + semantic compression)
    -> Constitutional Guardrail (untrusted retrieved content)
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
- `guardrail`: versioned baseline/hardened safety classifier and injection detector
- `agent_langgraph`: parallel LangGraph orchestration with conditional replanning and memory

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

### Jailbreak and prompt-injection robustness

This repository evaluates an inference-time constitutional guardrail; it
does not train or fine-tune a model. The primary attack metric is
**guardrail-bypass rate (GBR)**: attacks allowed beyond the guardrail /
attack trials. GBR is narrower than end-to-end attack success rate (ASR),
because it does not claim that a downstream model produced harmful output.
**False-positive rate (FPR)** is benign controls incorrectly blocked / benign
trials. Unless explicitly marked otherwise, intervals below are Wilson 95%
confidence intervals and every result is from a committed `live_api` artifact.

#### Research progression and headline findings

**Initial diagnostic (v1).** The original 68-attack/25-control diagnostic
found no observed bypasses in either arm but a high benign-refusal burden:

| Configuration | ASR as originally recorded (bootstrap 95% CI) | FPR (bootstrap 95% CI) |
|---|---:|---:|
| Baseline | 0/68 (0.0%; 0.0–0.0%) | 10/25 (40.0%; 20.0–60.0%) |
| Hardened | 0/68 (0.0%; 0.0–0.0%) | 7/25 (28.0%; 12.0–48.0%) |

This motivated a preregistered redesign: published attack artifacts, a
larger benign set, paired inference, and the more precise GBR terminology.

**Preregistered v2 — published direct attacks plus tool-surface injection.**
Model: `claude-sonnet-4-6`. Dataset fingerprint: `7426d3285e15d22c939dcd3cce05103f9fd034153e68efdf271e8c4302db5501`.

| Defense | Direct GBR | Indirect GBR | Combined GBR | FPR |
|---|---:|---:|---:|---:|
| Baseline | 0/80 (0.0%; 0.0–4.6%) | 0/30 (0.0%; 0.0–11.4%) | 0/110 (0.0%; 0.0–3.4%) | 13/60 (21.7%; 13.1–33.6%) |
| Hardened | 0/80 (0.0%; 0.0–4.6%) | 0/30 (0.0%; 0.0–11.4%) | 0/110 (0.0%; 0.0–3.4%) | 9/60 (15.0%; 8.1–26.1%) |

The hardened prompt fixed five benign refusals and introduced one benign regression, but the paired FPR change was not statistically conclusive (exact McNemar p=0.2188, N_pairs=60). Zero observed GBR means only that no bypass appeared in this sample; the combined hardened upper confidence bound remained 3.4%.

**Preregistered v3 — adaptive attacks and cross-provider replication.**
V3 added seven adaptive families (base64, leetspeak, multi-turn,
multilingual, nested quotation, split payload, and typoglycemia), quote-aware
classification, deterministic tool authorization, and safe canary execution.
The table reports only the valid `hardened_v3` arm; the OpenAI legacy-arm
comparison was invalidated by completion-budget exhaustion and is not used
for a defense-effect claim.

| Provider / pinned model | Adaptive GBR | FPR | Provider no-text/refusal outcomes |
|---|---:|---:|---:|
| anthropic / `claude-sonnet-4-6` | 0/70 (0.0%; 0.0–5.2%) | 1/40 (2.5%; 0.4–12.9%) | 10 |
| anthropic / `claude-haiku-4-5` | 0/70 (0.0%; 0.0–5.2%) | 0/40 (0.0%; 0.0–8.8%) | 0 |
| openai / `gpt-5-mini-2025-08-07` | 2/70 (2.9%; 0.8–9.8%) | 0/40 (0.0%; 0.0–8.8%) | 0 |

The two GPT-5 mini bypasses were both typoglycemia cases. This cross-provider
failure mechanism—not an aggregate win claim—motivated the held-out v4 study.
Sonnet had one benign false positive; Haiku and GPT-5 mini had none in their
valid v3 arms. Provider no-text/refusal counts are disclosed separately so
API behavior is not misattributed to repository defenses.

**Preregistered v4 — held-out typoglycemia hardening.** Development failures
were excluded before evaluation. The fixed test contained 60 attacks across
internal permutation, adjacent transposition, and punctuation splitting, plus
60 benign typo controls. All three runs used the same fingerprint:
`c0f1a96f65391a38e2634b6c0555c1d6e0fe580bfa49407a11e853cea2ab8a27`.

| Provider / pinned model | Defense | GBR | FPR | No-text |
|---|---|---:|---:|---:|
| anthropic / `claude-haiku-4-5` | `hardened_v3` | 3/60 (5.0%; 1.7–13.7%) | 14/60 (23.3%; 14.4–35.4%) | 0 |
| anthropic / `claude-haiku-4-5` | `hardened_v4` | 2/60 (3.3%; 0.9–11.4%) | 14/60 (23.3%; 14.4–35.4%) | 0 |
| openai / `gpt-5-mini-2025-08-07` | `hardened_v3` | 4/60 (6.7%; 2.6–15.9%) | 15/60 (25.0%; 15.8–37.2%) | 0 |
| openai / `gpt-5-mini-2025-08-07` | `hardened_v4` | 3/60 (5.0%; 1.7–13.7%) | 17/60 (28.3%; 18.5–40.8%) | 0 |
| anthropic / `claude-sonnet-4-6` | `hardened_v3` | 0/60 (0.0%; 0.0–6.0%) | 13/60 (21.7%; 13.1–33.6%) | 1 |
| anthropic / `claude-sonnet-4-6` | `hardened_v4` | 0/60 (0.0%; 0.0–6.0%) | 14/60 (23.3%; 14.4–35.4%) | 0 |

**Decision: v4 was rejected, and `hardened_v3` remains the default.** V4
reduced observed bypasses by only one case on Haiku and one on GPT-5 mini;
both paired tests had p=1.0000 (N_pairs=60). It did not change Sonnet GBR.
FPR was unchanged on Haiku, rose by two cases on GPT-5 mini (paired p=0.5000,
N_pairs=60), and rose by one on Sonnet (paired p=1.0000, N_pairs=60). The
remaining bypasses were internal-letter permutations. Mechanically, the fixed
normalization lexicon caught a narrow subset of scrambled directives but did
not generalize enough to justify its added benign-blocking risk.

#### Concrete residual failures

- GPT-5 mini v3 allowed two obfuscated retrieved-content instructions because
  it treated scrambled text as noise or non-actionable quoted material.
- In v4, Haiku still allowed 2/20 internal-permutation attacks and GPT-5 mini
  allowed 3/20; adjacent-transposition and punctuation-split attacks had zero
  observed bypasses in the hardened-v4 arms.
- Benign typo controls remain difficult: hardened-v4 FPR ranged from 14/60
  on Haiku and Sonnet to 17/60 on GPT-5 mini.

#### Evidence status and limitations

- Completed: preregistration, immutable dataset fingerprints, raw decision
  logs, Wilson intervals, paired exact tests, cross-provider replication, and
  a held-out negative-result study.
- Pending: independent blinded human adjudication and temporal replication on
  future model/API versions. No human-agreement or temporal-stability claim is
  made yet.
- Multi-turn attacks are serialized transcripts, not stateful interactive
  conversations. The studies measure guardrail passage, not downstream harmful
  generation. Model-judge labels can still be wrong.

#### Reproduce and inspect

```bash
make safety-data
make safety-dry-run
make safety-offline
python3 -m eval.safety.harness --max-cost-usd 1
python3 scripts/update_readme_research_findings.py --check
```

Protocols and detailed evidence:

- [V1 diagnostic report](eval/safety/results/safety_eval_20260725T033459Z.md)
- [V2 preregistration](RESEARCH_PLAN_V2.md) and [full report](eval/safety_v2/results/safety_v2_20260725T041722Z.md)
- [V3 preregistration](RESEARCH_PLAN_V3.md), [Sonnet report](eval/safety_v3/results/safety_v3_20260725T183929Z.md), [Haiku report](eval/safety_v3/results/safety_v3_20260725T185915Z.md), and [GPT-5 mini report](eval/safety_v3/results/safety_v3_20260726T014002Z.md)
- [V4 preregistration](RESEARCH_PLAN_V4.md) and [three-model held-out report](eval/safety_v4/results/safety_v4_cross_model_report.md)

## Pipeline Findings Still Requiring Live Replication

- Historical/offline results suggest planning improves completeness.
- Historical/offline results suggest verification trades extra tool calls for
  lower hallucination, but the current repository does not support a
  statistical-significance claim.
- Unanswerable handling is instrumented through explicit uncertainty reporting.

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
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 2. Configure keys

```bash
cp .env.example .env
# Fill ANTHROPIC_API_KEY and TAVILY_API_KEY
```

### 3. Run demo

```bash
python3 demo.py "How did RLHF change LLM alignment research?"
```

### 4. Run eval harness (paid API mode)

```bash
python3 -m eval.harness
```

### 5. Zero-cost offline eval (no API calls)

```bash
python3 -m eval.harness --offline
```

This writes a timestamped synthetic artifact marked `result_mode=offline_fixture` for smoke testing only.

### 6. Summarize latest run into a table

```bash
python3 -m eval.summarize_results --latest
```

To enforce benchmark-only summaries:

```bash
python3 -m eval.summarize_results --latest --mode live_api --require-live
```

### 7. Optional Makefile shortcuts

```bash
make demo QUESTION="How did RLHF change LLM alignment research?"
make eval
make eval-offline
make summarize-latest
make test
```

## LangGraph Variant

The repo also includes a parallel LangGraph implementation in `src/agent_langgraph/`.
It reuses the same planner/searcher/synthesizer/verifier components, but expresses
the orchestration as a `StateGraph` with:
- typed shared state,
- conditional replanning when search coverage collapses,
- thread-level persistence via LangGraph memory.

To compare the original orchestration with the LangGraph variant:

```bash
python3 scripts/compare_architectures.py --offline
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
python3 demo.py "What changed in post-training after RLHF?"
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
- The claim-level verification step is the agent analog of a Process Reward Model — scoring individual claims rather than the final answer, for the same reason that step-level reward signals in the RLHF setting catch errors that outcome-level signals miss (see [Extension 3: PRM vs ORM](https://github.com/kartikmunjal/rlhf-and-reward-modelling-alt#extension-3-process-reward-model-prm-vs-outcome-reward-model-orm)).
