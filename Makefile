.PHONY: install install-dev demo eval eval-offline eval-smoke eval-live-core freeze-benchmark summarize-latest summarize-live summarize-benchmark verifier-fpr-template verifier-fpr-measure test

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

demo:
	python demo.py "$(QUESTION)"

eval:
	python -m eval.harness

eval-live-core:
	python -m eval.harness --profile core_live_28

eval-offline:
	python3 -m eval.harness --offline

eval-smoke:
	python3 -m eval.harness --synthetic-smoke

freeze-benchmark:
	python -m eval.freeze_benchmark --run-file "$(RUN_FILE)" --output "$(OUTPUT)" --benchmark-name "$(BENCHMARK_NAME)"

summarize-latest:
	python -m eval.summarize_results --latest

test:
	python -m pytest -q


summarize-live:
	python -m eval.summarize_results --latest --mode live_api --require-live

summarize-benchmark:
	python -m eval.summarize_results --latest --mode replay_benchmark --require-benchmarkable

verifier-fpr-template:
	python -m eval.measure_verifier_fpr --run-file "$(RUN_FILE)" --export-template "$(OUTPUT)"

verifier-fpr-measure:
	python -m eval.measure_verifier_fpr --reference-file "$(REFERENCE_FILE)" --output "$(OUTPUT)"
