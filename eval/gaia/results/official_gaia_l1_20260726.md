# Official GAIA Level-1 Validation — Live Tools vs Frozen Replay

Dataset: gated `gaia-benchmark/GAIA`, 2023 validation Level 1. No questions,
reference answers, attachments, predictions, task IDs, or tool outputs are
reproduced. Both runs use `claude-sonnet-4-6` at temperature 0.
Task-set fingerprint: `662339b1469b4a05bf30af059d141ab4dfae0d8c028a5c6afebec8715b43728e`.

| Mode | Accuracy (Wilson 95% CI) | Attachment tasks | No-attachment tasks |
|---|---:|---:|---:|
| `live` | 31/53 (58.5%; 45.1–70.7%) | 7/11 (63.6%; 35.4–84.8%) | 24/42 (57.1%; 42.2–70.9%) |
| `replay` | 19/53 (35.8%; 24.3–49.3%) | 5/11 (45.5%; 21.3–72.0%) | 14/42 (33.3%; 21.0–48.4%) |

## Paired comparison

Live minus replay accuracy: 22.6 percentage points (paired bootstrap 95% CI 9.4 to 35.8; N_pairs=53, 10,000 resamples).
Exact McNemar p=0.0042; live-only correct=14, replay-only correct=2, discordant pairs=16.

## Failure taxonomy

- `live`: `reasoning_error`=20, `tool_error`=2.
- `replay`: `reasoning_error`=14, `retrieval_error`=4, `tool_error`=16.

## Tool execution and estimated model cost

- `live`: calls `calculator`=24, `read_file`=15, `web_search`=138; tool errors `calculator_error`=1, `file_not_found`=4, `search_error`=1; 1,635,267 input tokens, 34,488 output tokens, estimated Anthropic cost $5.42.
- `replay`: calls `calculator`=26, `read_file`=23, `web_search`=249; tool errors `calculator_error`=1, `file_not_found`=10, `replay_miss`=225; 1,160,171 input tokens, 35,640 output tokens, estimated Anthropic cost $4.02.

Cost uses the standard Sonnet 4.6 list price of $3/million input tokens and
$15/million output tokens; Tavily and OpenAI audio-transcription charges are
not included.

## Interpretation

`live` executes Tavily search, restricted arithmetic, local attachment readers,
and approved OpenAI audio transcription. `replay` reuses exact recorded
search/calculator observations when the model repeats the same call and re-reads
attachments. The lower replay result is driven partly by replay misses when the
model changes tool arguments despite temperature 0. Therefore replay is not an
oracle reasoning ceiling. The experiment shows that exact-call fixture mocks can
underestimate a live agent when tool selection itself is nondeterministic.

The defensible real-world score is the live result. Published GPT-4 figures are
context only unless split, level, model, tools, and scoring protocol match.
