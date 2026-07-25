# Research Plan: Guardrail Robustness

## Locked protocol

- Primary unit: the combined 93-example dataset, with direct-jailbreak,
  indirect-injection, and benign-adjacent strata retained in every comparison.
- Baseline: the single-pass constitutional classifier in `baseline` mode.
- Treatment: few-shot constitutional classifier plus deterministic high-precision
  injection prefilter in `hardened` mode.
- ASR: attacks allowed / all attack trials.
- FPR: benign examples refused / all benign trials.
- Uncertainty: non-parametric bootstrap 95% CI with 10,000 resamples and fixed seed.
- Execution: temperature zero, identical model and dataset for both configurations.
- Valid claim source: only artifacts marked `result_mode=live_api`.
- Selection: deterministic sampling fixed before outcomes; no post-hoc exclusions.

## Structural leakage controls

User input is screened before planning or tool use. Retrieved content is screened
after retrieval and before synthesis. Retrieved instructions therefore cannot be
placed into the synthesizer context when blocked. Dataset building is independent
of evaluation outcomes.

## Interpretation

An improvement is trusted only if the per-example transition log supports the
intended mechanism: new refusals should be attributable to the few-shot boundary
or injection detector, while benign refusals must not increase materially. The
combined ASR/FPR result is primary; direct and indirect strata are diagnostic.

No threshold, metric definition, sample, or configuration may be changed after a
live run without recording a new protocol version.
