# Preregistered official GAIA validation study

Status: locked before official task outcomes are observed.

## Question

How much performance does this scaffold lose to live retrieval/tool execution,
and which failures arise from reasoning, retrieval, file reading, generic tool
execution, or the model provider?

## Dataset

- Official gated `gaia-benchmark/GAIA`, 2023 validation, Level 1.
- Primary run: the complete Level-1 validation split in deterministic task-id
  order. A `--limit` run is a plumbing pilot and cannot be reported as the
  primary benchmark.
- Questions, answers, attachments, predictions, and tool outputs remain private
  under the dataset terms. A SHA-256 task-set fingerprint establishes identity.
- The 12 self-authored `eval/tasks.json` items are excluded from benchmark claims.

## Fixed system

- Model: `claude-sonnet-4-6`, temperature 0.
- Maximum 12 model turns per task.
- Live tools: Tavily Advanced search (maximum 8 results/call), restricted-AST
  calculator, and read-only attachment reader.
- File access is confined to the downloaded GAIA snapshot.
- Audio attachments are transcribed with `gpt-4o-mini-transcribe` through the
  OpenAI API. This provider boundary was added before the first valid full run
  after an integrity pilot revealed that official Level 1 contains MP3 files;
  the user explicitly approved sending those two gated audio attachments solely
  for transcription. The path-defective pilot is marked invalid.

## Comparison

1. `live`: real Tavily, calculator, and attachment reads.
2. `replay`: same official tasks; search/calculator results replayed from the
   completed live trace, attachments deterministically re-read.

Replay controls network/tool-output variance. It is not an oracle and is not
interpreted as a pure reasoning ceiling.

## Metrics

- Primary: exact-match accuracy, events/N_trials, Wilson 95% CI.
- Failure counts: `reasoning_error`, `tool_error`, `retrieval_error`,
  `file_read_error`, `provider_error`.
- Report both modes. Interpret their difference descriptively unless paired
  discordances are sufficient for a prespecified exact McNemar test.
- Provider errors remain intention-to-treat failures and are reported separately.

## Reporting constraints

- No comparison to published GPT-4 performance unless the cited result uses the
  same official split/level and its differing model, tools, and scaffold are
  explicitly disclosed.
- No official result is reported from a partial pilot.
- No gated task content or private artifact is committed.
