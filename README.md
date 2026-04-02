# Deep Research Agent

A research agent that takes a question, decomposes it into sub-questions, retrieves and compresses evidence, synthesizes a cited answer, and verifies every factual claim against the retrieved sources.

The project is structured around three engineering problems that arise in any non-trivial agent: **context management under pressure**, **grounding and hallucination detection**, and **failure recovery**. Each has a dedicated module and a measurable outcome in the eval harness.

---

## Pipeline

```
Question
    │
    ▼
┌─────────────────────────────────────────┐
│  PLANNER                                │
│  Decomposes question into 3–5 focused   │
│  sub-questions, each independently      │
│  searchable                             │
└────────────────────┬────────────────────┘
                     │ sub-questions
                     ▼
┌─────────────────────────────────────────┐
│  SEARCHER                               │
│  Tavily web search per sub-question     │
│  → Claude semantic compression:         │
│    extracts only the sentences from     │
│    each article relevant to the sub-q   │
└────────────────────┬────────────────────┘
                     │ evidence (2–5 sentences + URL per source)
                     ▼
┌─────────────────────────────────────────┐
│  SYNTHESIZER                            │
│  Claude synthesizes a structured answer │
│  with inline [N] citations from the     │
│  compressed evidence                    │
└────────────────────┬────────────────────┘
                     │ draft answer
                     ▼
┌─────────────────────────────────────────┐
│  VERIFIER                               │
│  Extracts individual factual claims     │
│  Checks each claim against evidence     │
│  Flags unverified claims explicitly     │
└────────────────────┬────────────────────┘
                     │
                     ▼
              Verified Answer
          + Hallucination Report
          + Coverage Gap Summary
```

---

## Results

Evaluated on 28 research questions across three categories: factual (10), multi-hop (10), and unanswerable (8). Three ablation configurations are compared to isolate the contribution of each component.

| Configuration | Citation Acc. | Completeness | Hallucination % | Avg Tool Calls |
|---|:---:|:---:|:---:|:---:|
| No planning, no verification | 0.61 | 0.54 | 34.7% | 3.2 |
| Planning, no verification | 0.69 | 0.69 | 27.8% | 5.1 |
| Planning + verification | 0.73 | 0.69 | 9.2% | 6.4 |

The verification pass is the most expensive step (+1.3 tool calls) but produces the sharpest drop in hallucination rate: 27.8% → 9.2%. Planning independently improves completeness by 15 points by ensuring the synthesizer has evidence for each structural component of the answer before synthesis begins.

On the 8 unanswerable questions, the full pipeline correctly flags uncertainty in **87.5%** of cases rather than fabricating an answer.

Reproduce with: `python -m eval.harness`

Full breakdown and failure mode analysis: `notebooks/analysis.ipynb`

---

## What Makes This Hard

### 1. Context management via semantic compression

The naive approach passes full article text to the synthesizer. On a question with 4 sub-questions and 4 results each, that is ~16 articles × up to 10,000 characters = 160,000+ characters before the synthesis prompt is written. This exceeds practical context budgets and, more importantly, buries the relevant signal in noise.

The fix is a two-stage retrieval: Tavily fetches the article, then a Claude call extracts the 2–5 sentences from that article that directly answer the sub-question. Only these extracted sentences, with their source URLs, are passed forward. The synthesizer works from semantically compressed evidence, not raw documents. This is the `EXTRACT_PROMPT` in `src/agent/searcher.py`.

The tradeoff: one extra Claude call per source. On 4 results per sub-question and 4 sub-questions, that is 16 extraction calls. This is worthwhile because the synthesizer produces better answers from 16 targeted sentences than from 16 full articles, and the smaller context also makes the verification pass cheaper and more precise.

### 2. Claim-level grounding, not answer-level scoring

A standard agent pipeline scores the final answer: does it match the reference? This misses two important failure modes. First, an answer can be mostly correct but contain one fabricated claim — answer-level scoring masks this. Second, an answer can cite sources that don't actually support the claims cited — citation drift is a subtle hallucination that answer-level scoring never catches.

The verifier operates claim-by-claim. It first calls Claude to extract distinct factual claims from the answer (not summaries or hedges, just specific assertions). It then checks each claim against the evidence in a single pass, returning a verified/unverified label and the supporting excerpt when found.

This produces a per-claim hallucination rate rather than a binary pass/fail, which is a more actionable signal. It also catches citation drift: a claim can be plausible and even reference the right source, but if the cited source's extract doesn't contain supporting text, the claim is flagged.

### 3. Failure recovery and uncertainty reporting

When Tavily returns nothing useful for a sub-question, a naive agent either skips the sub-question silently or hallucinates an answer. Both are wrong.

The searcher handles this explicitly: if the initial search returns no usable extracts, it calls Claude to reformulate the query (stripping filler words, trying synonyms) and retries once. If the retry also fails, the sub-question is marked `search_successful=False`. The synthesizer sees which sub-questions have no evidence and is explicitly instructed not to fabricate — it lists them as coverage gaps in the output.

The unanswerable task category in the eval set tests this at a harder level: questions where web search returns confident-sounding but unreliable sources (e.g., "what are OpenAI's unpublished training hyperparameters?"). The verifier catches the hallucinations in these cases, and the synthesizer is instructed to flag them explicitly rather than present them as established facts.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/kartikmunjal/deep-research-agent
cd deep-research-agent
pip install -r requirements.txt

# 2. Set API keys
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and TAVILY_API_KEY

# 3. Run a demo question
python demo.py "How did RLHF change the trajectory of LLM alignment?"

# 4. Run the full eval (all configs, all tasks)
python -m eval.harness

# 5. Explore results
jupyter notebook notebooks/analysis.ipynb
```

API keys required:
- **Anthropic**: [console.anthropic.com](https://console.anthropic.com)
- **Tavily**: [app.tavily.com](https://app.tavily.com) (free tier available)

---

## Repository Structure

```
src/agent/
├── planner.py      # Question decomposition
├── searcher.py     # Tavily search + semantic compression
├── synthesizer.py  # Evidence → cited answer
├── verifier.py     # Claim-level grounding
├── pipeline.py     # Orchestration + failure recovery
└── models.py       # Shared dataclasses

eval/
├── tasks.json      # 28 research questions (factual / multi-hop / unanswerable)
├── harness.py      # Runs all configs, writes results to eval/results/
└── scoring.py      # Metrics: citation accuracy, completeness, hallucination rate

notebooks/
└── analysis.ipynb  # Ablation table + failure mode analysis

demo.py             # Single-question entrypoint
```
