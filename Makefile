.PHONY: install install-dev demo eval summarize-latest test

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

demo:
	python demo.py "$(QUESTION)"

eval:
	python -m eval.harness

summarize-latest:
	python -m eval.summarize_results --latest

test:
	python -m pytest -q
