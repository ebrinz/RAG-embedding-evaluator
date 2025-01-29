
# Run complete evaluation pipeline
evals: setup download-data prepare-evals run-evals

# Setup evals environment for model analysis
setup:
	uv venv && source .venv/bin/activate && \
	uv pip install -r requirements.txt && \
	uv pip install ipykernel jupyter && \
	python -m ipykernel install --user --name=evals-kernel

# Download dataset
download-data:
	python -m embedding.cli download --config config/embedding_eval.yaml

# Prepare evaluation data
prepare-evals:
	python -m embedding.cli prepare --config config/embedding_eval.yaml

# Run evaluation with prepared data
run-evals:
	python -m embedding.cli evaluate \
		--config config/embedding_eval.yaml \
		--output results/embedding_benchmarks/results.csv \
		--verbose

# Watch evaluation tests
test-evals-watch:
	pytest-watch embedding/tests/ -- -v --cov=embedding

# Clean up evals environment after use
clean:
	rm -rf evals/.venv
	which jupyter > /dev/null && jupyter kernelspec uninstall evals-kernel -y || true