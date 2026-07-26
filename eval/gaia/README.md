# Official GAIA evaluation

This track uses the gated `gaia-benchmark/GAIA` 2023 validation split. It is
separate from the 12 self-authored smoke-test questions formerly labeled
`gaia_l1` in `eval/tasks.json`.

Accept the dataset terms on Hugging Face, then set `HF_TOKEN`,
`ANTHROPIC_API_KEY`, and `TAVILY_API_KEY` in `.env`. Private dataset caches and
per-task artifacts are ignored by Git.

```bash
python3 -m eval.gaia.harness --mode live --level 1 --limit 5 --max-cost-usd 1
python3 -m eval.gaia.harness --mode live --level 1 --max-cost-usd 10 --resume <artifact>
python3 -m eval.gaia.harness --mode replay --level 1 --max-cost-usd 10 \
  --replay-artifact <completed-live-artifact>
python3 -m eval.gaia.report <live-artifact> <replay-artifact> \
  --output eval/gaia/results/official_gaia_report.md
```

Tools are genuine:

- `web_search` executes Tavily Advanced search.
- `calculator` evaluates a restricted arithmetic AST; it never returns canned
  values and never calls Python `eval`.
- `read_file` reads PDFs, XLSX/XLSM workbooks, DOCX, CSV/TSV, structured text,
  PPTX, and sends image attachments to the model as image blocks. Paths are
  confined to the private GAIA snapshot. Audio files are sent only to
  OpenAI `gpt-4o-mini-transcribe`, following explicit user approval.

Failure labels are mutually exclusive: `reasoning_error`, `tool_error`,
`retrieval_error`, `file_read_error`, and `provider_error`.

The replay mode is the valid same-task comparison available without answer
leakage: it freezes live search/calculator observations while retaining
deterministic attachment reads. It is not an oracle “reasoning ceiling.”
