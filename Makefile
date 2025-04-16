default: help

.PHONY: all
all: install lint format test build # Run all linting, formatting, testing, and building steps

.PHONY: setup-pre-commit
setup-pre-commit: # Setup pre-commit hooks for ensuring code quality.
	pipx install pre-commit
	pre-commit install

.PHONY: setup-uv
setup-uv: # Setup uv for dependency management.
	pipx install uv==0.5.21 --force

.PHONY: setup-java
setup-java: # Setup Java needed for PySpark using sdkman
	curl -s "https://get.sdkman.io" | bash && \
	source "$(HOME)/.sdkman/bin/sdkman-init.sh" && \
	sdk env install

.PHONY: setup
setup: setup-java setup-pre-commit setup-uv install # Setup the project for development.

.PHONY: clean
clean: # Clean the project of build artifacts.
	rm -rf dist
	rm -rf build
	rm -rf .tox
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .coverage
	rm -rf .eggs
	rm -rf .hypothesis
	rm -rf .pytest


.PHONY: docs
docs: # Generate documentation using Sphinx.
	mkdocs build && \
	mkdocs serve

.PHONY: install
install: # Install project dependencies.
	uv sync

.PHONY: lint
lint: # Check code quality using flake8 & pylint.
	uv run flake8 src/kamae --max-complexity 10 --max-line-length 88 --extend-ignore E203 -v && \
    uv run pylint src/kamae --fail-under 5 --disable=E0401,E0611,R0903

.PHONY: format
format: # Check code formatting using black & isort
	uv run black --check --diff src/kamae && \
	uv run isort --check src/kamae --profile black

.PHONY: test
test: # Run tests using pytest.
	uv run python -m pytest -n auto .

.PHONY: test-cov
test-cov: # Run tests using pytest and generate coverage report.
	uv run python -m pytest --cov-report term-missing --cov-fail-under 80 --cov-branch --cov src/kamae -n auto tests/

.PHONY: build
build: # Build the project into a wheel.
	uv build

.PHONY: run-example
run-example: # Run the example pipeline
	uv run python examples/spark/example_pipeline.py

.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

.PHONY: test-tf-serving
test-tf-serving: # Test the TensorFlow serving container
	read -p "Enter location of model bundle to serve [./output/test_keras_model/1/]: " source_loc; \
	read -p "Enter python script location to test inference [./examples/inference/tf_serving_request.py]: " script; \
	source_loc=$${source_loc:-./output/test_keras_model/1/}; \
	script=$${script:-./examples/inference/tf_serving_request.py}; \
	docker run -p 8501:8501 -d --platform linux/amd64 --rm --name=test_tf_serving --mount type=bind,source=$$source_loc,target=/models/test_keras_model/1/ -e MODEL_NAME=test_keras_model -t tensorflow/serving:2.6.2; \
	sleep 5; \
	uv run python $$script; \
	docker stop test_tf_serving

.PHONY: test-end-to-end
test-end-to-end: run-example test-tf-serving # Run end-to-end tests for the project
	@echo "End-to-end tests passed successfully"
