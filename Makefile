.PHONY: install install-dev demo eval eval-offline summarize-latest summarize-live test

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

summarize-latest:
	python -m eval.summarize_results --latest

test:
	python -m pytest -q


summarize-live:
	python -m eval.summarize_results --latest --mode live_api --require-live
