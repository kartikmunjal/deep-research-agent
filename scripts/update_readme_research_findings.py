"""Regenerate the top-level README safety findings from committed live artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
START = "### Jailbreak and prompt-injection robustness"
END = "## Pipeline Findings Still Requiring Live Replication"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def metric(value: dict) -> str:
    lo, hi = value["ci_95"]
    return (
        f'{value["events"]}/{value["n_trials"]} '
        f'({100 * value["rate"]:.1f}%; '
        f'{100 * lo:.1f}–{100 * hi:.1f}%)'
    )


def legacy_metric(metrics: dict, name: str) -> str:
    rate = metrics[name]
    lo, hi = metrics[f"{name}_95ci"]
    n = metrics[f"{name}_n_trials"]
    events = round(rate * n)
    return f"{events}/{n} ({100 * rate:.1f}%; {100 * lo:.1f}–{100 * hi:.1f}%)"


def provider(artifact: dict) -> str:
    return artifact.get("provider", "anthropic")


def v3_row(artifact: dict) -> str:
    current = artifact["configurations"]["hardened_v3"]["metrics"]
    no_text = current.get("provider_no_text", current.get("provider_refusals", 0))
    return (
        f'| {provider(artifact)} / `{artifact["model"]}` | '
        f'{metric(current["adaptive_gbr"])} | {metric(current["fpr"])} | '
        f"{no_text} |"
    )


def v4_row(artifact: dict, defense: str) -> str:
    current = artifact["configurations"][defense]["metrics"]
    return (
        f'| {provider(artifact)} / `{artifact["model"]}` | `{defense}` | '
        f'{metric(current["gbr"])} | {metric(current["fpr"])} | '
        f'{current.get("provider_no_text", 0)} |'
    )


def build_section() -> str:
    v1 = load("eval/safety/results/safety_eval_20260725T033459Z.json")
    v2 = load("eval/safety_v2/results/safety_v2_20260725T041722Z.json")
    v3_sonnet = load("eval/safety_v3/results/safety_v3_20260725T183929Z.json")
    v3_haiku = load("eval/safety_v3/results/safety_v3_20260725T185915Z.json")
    v3_openai = load("eval/safety_v3/results/safety_v3_20260726T014002Z.json")
    v4_haiku = load("eval/safety_v4/results/safety_v4_20260726T022710Z.json")
    v4_openai = load("eval/safety_v4/results/safety_v4_20260726T034449Z.json")
    v4_sonnet = load("eval/safety_v4/results/safety_v4_20260726T150114Z.json")

    v1_base = v1["configurations"]["baseline"]["metrics"]
    v1_hard = v1["configurations"]["hardened"]["metrics"]
    v2_base = v2["configurations"]["baseline"]["metrics"]
    v2_hard = v2["configurations"]["hardened"]["metrics"]
    v2_pair = v2["paired_tests"]["benign_false_positive"]

    lines = [
        START,
        "",
        "This repository evaluates an inference-time constitutional guardrail; it",
        "does not train or fine-tune a model. The primary attack metric is",
        "**guardrail-bypass rate (GBR)**: attacks allowed beyond the guardrail /",
        "attack trials. GBR is narrower than end-to-end attack success rate (ASR),",
        "because it does not claim that a downstream model produced harmful output.",
        "**False-positive rate (FPR)** is benign controls incorrectly blocked / benign",
        "trials. Unless explicitly marked otherwise, intervals below are Wilson 95%",
        "confidence intervals and every result is from a committed `live_api` artifact.",
        "",
        "#### Research progression and headline findings",
        "",
        "**Initial diagnostic (v1).** The original 68-attack/25-control diagnostic",
        "found no observed bypasses in either arm but a high benign-refusal burden:",
        "",
        "| Configuration | ASR as originally recorded (bootstrap 95% CI) | FPR (bootstrap 95% CI) |",
        "|---|---:|---:|",
        f"| Baseline | {legacy_metric(v1_base, 'asr')} | {legacy_metric(v1_base, 'fpr')} |",
        f"| Hardened | {legacy_metric(v1_hard, 'asr')} | {legacy_metric(v1_hard, 'fpr')} |",
        "",
        "This motivated a preregistered redesign: published attack artifacts, a",
        "larger benign set, paired inference, and the more precise GBR terminology.",
        "",
        "**Preregistered v2 — published direct attacks plus tool-surface injection.**",
        f"Model: `{v2['model']}`. Dataset fingerprint: `{v2['dataset_fingerprint']}`.",
        "",
        "| Defense | Direct GBR | Indirect GBR | Combined GBR | FPR |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Baseline | {metric(v2_base['guardrail_bypass_direct'])} | "
            f"{metric(v2_base['guardrail_bypass_indirect'])} | "
            f"{metric(v2_base['guardrail_bypass_all'])} | "
            f"{metric(v2_base['false_positive_rate'])} |"
        ),
        (
            f"| Hardened | {metric(v2_hard['guardrail_bypass_direct'])} | "
            f"{metric(v2_hard['guardrail_bypass_indirect'])} | "
            f"{metric(v2_hard['guardrail_bypass_all'])} | "
            f"{metric(v2_hard['false_positive_rate'])} |"
        ),
        "",
        (
            "The hardened prompt fixed five benign refusals and introduced one benign "
            "regression, but the paired FPR change was not statistically conclusive "
            f"(exact McNemar p={v2_pair['p_value_two_sided']:.4f}, "
            f"N_pairs={v2_pair['n_pairs']}). Zero observed GBR means only that no "
            "bypass appeared in this sample; the combined hardened upper confidence "
            f"bound remained {100 * v2_hard['guardrail_bypass_all']['ci_95'][1]:.1f}%."
        ),
        "",
        "**Preregistered v3 — adaptive attacks and cross-provider replication.**",
        "V3 added seven adaptive families (base64, leetspeak, multi-turn,",
        "multilingual, nested quotation, split payload, and typoglycemia), quote-aware",
        "classification, deterministic tool authorization, and safe canary execution.",
        "The table reports only the valid `hardened_v3` arm; the OpenAI legacy-arm",
        "comparison was invalidated by completion-budget exhaustion and is not used",
        "for a defense-effect claim.",
        "",
        "| Provider / pinned model | Adaptive GBR | FPR | Provider no-text/refusal outcomes |",
        "|---|---:|---:|---:|",
        v3_row(v3_sonnet),
        v3_row(v3_haiku),
        v3_row(v3_openai),
        "",
        "The two GPT-5 mini bypasses were both typoglycemia cases. This cross-provider",
        "failure mechanism—not an aggregate win claim—motivated the held-out v4 study.",
        "Sonnet had one benign false positive; Haiku and GPT-5 mini had none in their",
        "valid v3 arms. Provider no-text/refusal counts are disclosed separately so",
        "API behavior is not misattributed to repository defenses.",
        "",
        "**Preregistered v4 — held-out typoglycemia hardening.** Development failures",
        "were excluded before evaluation. The fixed test contained 60 attacks across",
        "internal permutation, adjacent transposition, and punctuation splitting, plus",
        "60 benign typo controls. All three runs used the same fingerprint:",
        f"`{v4_haiku['dataset_fingerprint']}`.",
        "",
        "| Provider / pinned model | Defense | GBR | FPR | No-text |",
        "|---|---|---:|---:|---:|",
        v4_row(v4_haiku, "hardened_v3"),
        v4_row(v4_haiku, "hardened_v4"),
        v4_row(v4_openai, "hardened_v3"),
        v4_row(v4_openai, "hardened_v4"),
        v4_row(v4_sonnet, "hardened_v3"),
        v4_row(v4_sonnet, "hardened_v4"),
        "",
        "**Decision: v4 was rejected, and `hardened_v3` remains the default.** V4",
        "reduced observed bypasses by only one case on Haiku and one on GPT-5 mini;",
        "both paired tests had p=1.0000 (N_pairs=60). It did not change Sonnet GBR.",
        "FPR was unchanged on Haiku, rose by two cases on GPT-5 mini (paired p=0.5000,",
        "N_pairs=60), and rose by one on Sonnet (paired p=1.0000, N_pairs=60). The",
        "remaining bypasses were internal-letter permutations. Mechanically, the fixed",
        "normalization lexicon caught a narrow subset of scrambled directives but did",
        "not generalize enough to justify its added benign-blocking risk.",
        "",
        "#### Concrete residual failures",
        "",
        "- GPT-5 mini v3 allowed two obfuscated retrieved-content instructions because",
        "  it treated scrambled text as noise or non-actionable quoted material.",
        "- In v4, Haiku still allowed 2/20 internal-permutation attacks and GPT-5 mini",
        "  allowed 3/20; adjacent-transposition and punctuation-split attacks had zero",
        "  observed bypasses in the hardened-v4 arms.",
        "- Benign typo controls remain difficult: hardened-v4 FPR ranged from 14/60",
        "  on Haiku and Sonnet to 17/60 on GPT-5 mini.",
        "",
        "#### Evidence status and limitations",
        "",
        "- Completed: preregistration, immutable dataset fingerprints, raw decision",
        "  logs, Wilson intervals, paired exact tests, cross-provider replication, and",
        "  a held-out negative-result study.",
        "- Pending: independent blinded human adjudication and temporal replication on",
        "  future model/API versions. No human-agreement or temporal-stability claim is",
        "  made yet.",
        "- Multi-turn attacks are serialized transcripts, not stateful interactive",
        "  conversations. The studies measure guardrail passage, not downstream harmful",
        "  generation. Model-judge labels can still be wrong.",
        "",
        "#### Reproduce and inspect",
        "",
        "```bash",
        "make safety-data",
        "make safety-dry-run",
        "make safety-offline",
        "python3 -m eval.safety.harness --max-cost-usd 1",
        "python3 scripts/update_readme_research_findings.py --check",
        "```",
        "",
        "Protocols and detailed evidence:",
        "",
        "- [V1 diagnostic report](eval/safety/results/safety_eval_20260725T033459Z.md)",
        "- [V2 preregistration](RESEARCH_PLAN_V2.md) and [full report](eval/safety_v2/results/safety_v2_20260725T041722Z.md)",
        "- [V3 preregistration](RESEARCH_PLAN_V3.md), [Sonnet report](eval/safety_v3/results/safety_v3_20260725T183929Z.md), [Haiku report](eval/safety_v3/results/safety_v3_20260725T185915Z.md), and [GPT-5 mini report](eval/safety_v3/results/safety_v3_20260726T014002Z.md)",
        "- [V4 preregistration](RESEARCH_PLAN_V4.md) and [three-model held-out report](eval/safety_v4/results/safety_v4_cross_model_report.md)",
        "",
        "",
    ]
    return "\n".join(lines)


def render(current: str) -> str:
    start = current.index(START)
    end = current.index(END, start)
    return current[:start] + build_section() + current[end:]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero when README is not synchronized with result artifacts.",
    )
    args = parser.parse_args()
    current = README.read_text()
    generated = render(current)
    if args.check:
        if current != generated:
            raise SystemExit("README research findings are out of date")
        print("README research findings are synchronized")
    else:
        README.write_text(generated)
        print(README)


if __name__ == "__main__":
    main()
