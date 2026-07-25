.PHONY: install install-dev demo eval eval-offline safety-data safety-dry-run safety-offline compare-architectures summarize-latest summarize-live test

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

compare-architectures:
	$(PYTHON) scripts/compare_architectures.py --offline

summarize-latest:
	$(PYTHON) -m eval.summarize_results --latest

test:
	$(PYTHON) -m pytest -q


summarize-live:
	$(PYTHON) -m eval.summarize_results --latest --mode live_api --require-live
