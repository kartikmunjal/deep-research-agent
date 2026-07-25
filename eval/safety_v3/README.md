# Adaptive Guardrail and Canary Evaluation v3

V3 extends the immutable v2 study with adaptive/obfuscated attacks, a quote-aware
defense, deterministic tool authorization, safe canary execution, blinded human
annotation, and cross-model/temporal replication validators.

## Build and validate without paid calls

```bash
source .venv/bin/activate
python3 scripts/build_safety_eval_v3_dataset.py
python3 -m eval.safety_v3.harness --dry-run
python3 -m eval.safety_v3.canary_harness
python3 -m eval.safety_v3.annotation export reviewer_a.csv
python3 -m pytest -q
```

The canary executor never performs real external effects. It records hypothetical
tool requests and fails if policy would permit an unauthorized execution.

## Live paired run

```bash
python3 -m eval.safety_v3.harness --max-cost-usd 3
python3 -m eval.safety_v3.report eval/safety_v3/results/safety_v3_<timestamp>.json
```

Interrupted runs resume with `--resume <artifact.json>`. Do not make v3 claims
until the live artifact is complete.

## Human labels

Give separately exported copies to two independent reviewers along with
`ANNOTATION_RUBRIC.md`, then run:

```bash
python3 -m eval.safety_v3.annotation analyze reviewer_a.csv reviewer_b.csv agreement.json
```

The script rejects missing labels and reports agreement and Cohen's kappa.

## Replication

Run the same fingerprint with another model ID, or on later UTC dates, then:

```bash
python3 -m eval.safety_v3.replication cross_model run_a.json run_b.json --output model_matrix.json
python3 -m eval.safety_v3.replication temporal run_1.json run_2.json run_3.json --output temporal.json
```

The validator refuses mismatched protocols/fingerprints, fewer than two model
IDs, or fewer than three distinct temporal dates. A local open-weight model needs
a separate provider adapter or an OpenAI-compatible endpoint; no such run is
claimed by this repository yet.
