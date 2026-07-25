.PHONY: install install-dev demo eval eval-offline safety-data safety-dry-run safety-offline compare-architectures summarize-latest summarize-live test

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

demo:
	python demo.py "$(QUESTION)"

eval:
	python -m eval.harness

eval-offline:
	python3 -m eval.harness --offline

safety-data:
	python3 scripts/build_safety_eval_dataset.py

safety-dry-run:
	python3 -m eval.safety.harness --dry-run

safety-offline:
	python3 -m eval.safety.harness --offline

compare-architectures:
	python scripts/compare_architectures.py --offline

summarize-latest:
	python -m eval.summarize_results --latest

test:
	python -m pytest -q


summarize-live:
	python -m eval.summarize_results --latest --mode live_api --require-live
