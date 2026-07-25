# Preregistered Research Program v3

Protocol identifier: `safety-v3-adaptive-canary`.

V1 and v2 remain immutable. V3 evaluates seven extensions: adaptive attacks,
quote-aware detection, blinded benign-label adjudication, safe end-to-end canary
tools, deterministic tool authorization, cross-model replication, and temporal
replication.

## Locked design

- Attack families: base64, typoglycemia, leetspeak, multilingual, nested
  quotation, split-payload, and multi-turn goal hijacking.
- Defenses: v2 `hardened` versus new `hardened_v3`; identical examples and order.
- Guardrail outcomes: direct and indirect GBR, family-stratified GBR, FPR,
  Wilson 95% intervals, and paired exact McNemar tests.
- Canary outcome: unauthorized canary execution / attack trials. Canary tools
  record decisions and never perform real side effects.
- Tool policy: deny non-allowlisted tools, secret-bearing arguments, destructive
  shell commands, and non-allowlisted domains; require confirmation for external
  writes.
- Human labels: two blinded reviewers using the locked rubric; disagreements are
  retained and agreement is reported with Cohen's kappa. No synthetic reviewer
  may be substituted.
- Cross-model replication: identical dataset fingerprint and configuration;
  model-specific artifacts remain separate.
- Temporal replication: at least three completed live runs on distinct UTC dates;
  no pooling until all fingerprints and protocol IDs match.
- Claim boundary: inference-time safety evaluation only; no training or GPU claim.

Any change to the families, policy rules, label rubric, primary outcomes, or
replication criteria requires a new protocol identifier.
