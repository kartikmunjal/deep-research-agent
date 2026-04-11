from __future__ import annotations

from pathlib import Path

from eval import summarize_results


def test_summarize_results_markdown_table(tmp_path: Path) -> None:
    run_file = tmp_path / "20260411T120000Z.json"
    run_file.write_text(
        """
{
  "run_id": "20260411T120000Z",
  "results": [
    {
      "config": "plan_verify",
      "category": "factual",
      "tool_calls": 6,
      "scores": {
        "citation_accuracy": 0.8,
        "completeness": 0.7,
        "hallucination_rate": 0.1
      }
    },
    {
      "config": "plan_verify",
      "category": "multi_hop",
      "tool_calls": 8,
      "scores": {
        "citation_accuracy": 0.6,
        "completeness": 0.9,
        "hallucination_rate": 0.2
      }
    }
  ]
}
""".strip()
    )

    run = summarize_results._load(run_file)
    rows = summarize_results._summarize(run)
    markdown = summarize_results._to_markdown(rows)

    assert "| Config | Citation Accuracy | Completeness | Hallucination Rate | Avg Tool Calls | N |" in markdown
    assert "| plan_verify | 0.700 | 0.800 | 0.150 | 7.00 | 2 |" in markdown
