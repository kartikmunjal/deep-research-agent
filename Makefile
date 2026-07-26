.PHONY: install install-dev demo eval eval-offline safety-data safety-dry-run safety-offline safety-v2-data safety-v2-dry-run safety-v3-data safety-v3-dry-run safety-v3-canary safety-v4-data safety-v4-dry-run compare-architectures summarize-latest summarize-live test

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

demo:
	$(PYTHON) demo.py "$(QUESTION)"

eval:
	$(PYTHON) -m eval.harness

eval-offline:
	$(PYTHON) -m eval.harness --offline

safety-data:
	$(PYTHON) scripts/build_safety_eval_dataset.py

safety-dry-run:
	$(PYTHON) -m eval.safety.harness --dry-run

safety-offline:
	$(PYTHON) -m eval.safety.harness --offline

safety-v2-data:
	$(PYTHON) scripts/build_safety_eval_v2_dataset.py

safety-v2-dry-run:
	$(PYTHON) -m eval.safety_v2.harness --dry-run

safety-v3-data:
	$(PYTHON) scripts/build_safety_eval_v3_dataset.py

safety-v3-dry-run:
	$(PYTHON) -m eval.safety_v3.harness --dry-run

safety-v3-canary:
	$(PYTHON) -m eval.safety_v3.canary_harness

safety-v4-data:
	$(PYTHON) scripts/build_safety_eval_v4_dataset.py

safety-v4-dry-run:
	$(PYTHON) -m eval.safety_v4.harness --dry-run

compare-architectures:
	$(PYTHON) scripts/compare_architectures.py --offline

summarize-latest:
	$(PYTHON) -m eval.summarize_results --latest

test:
	$(PYTHON) -m pytest -q


summarize-live:
	$(PYTHON) -m eval.summarize_results --latest --mode live_api --require-live
