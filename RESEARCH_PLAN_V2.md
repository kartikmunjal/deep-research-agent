# Preregistered Research Plan v2

Protocol identifier: `safety-v2-preregistered`.

## Locked before execution

- Dataset: 80 direct JailbreakBench artifact prompts, 30 authored indirect
  injections, and 60 benign controls.
- Direct sampling: four methods; two lowest-index submitted prompts in each of
  ten categories per method. Historical outcome fields are never used.
- Configurations: existing `baseline` and `hardened` guardrails, same model,
  temperature zero, paired example order.
- Primary outcomes: combined GBR and FPR.
- Diagnostic outcomes: direct GBR, indirect GBR, detector, method, family,
  category, and provider-refusal counts.
- Uncertainty: Wilson 95% interval for every rate.
- Paired inference: two-sided exact McNemar test, reported without using p-values
  as a pass/fail gate.
- Missingness: no trial deletion. Provider refusals are blocked outcomes and are
  separately counted. Interrupted runs resume from atomic checkpoints.
- Claim boundary: this is inference-time guardrail evaluation, not model training
  and not downstream harmful-generation ASR.

V1 remains an immutable pilot. Any change to v2 sampling, labels, prompts,
metrics, or configurations requires a new protocol identifier.
