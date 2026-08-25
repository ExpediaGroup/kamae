---
name: verify
description: Run the standard quality pipeline — formatting, linting, tests, and coverage checks.
---

# Verify

Run this before declaring any task complete.

## Quick check

```bash
make format && make lint && make test
```

This runs:
- `black --check --diff` and `isort --check` (formatting)
- `flake8` with annotation checks and `pylint` with fail-under 5 (linting)
- `pytest -n auto` (all tests in parallel)

## Full CI-equivalent check

```bash
make test-cov
```

Runs tests with 80% coverage threshold and branch coverage enabled.

## Fix formatting and license headers

```bash
pre-commit run --all-files
```

This auto-fixes:
- Black formatting
- isort import ordering
- Apache 2.0 license headers on all `.py` files

## Common failures

| Failure | Fix |
|---|---|
| Missing license header | Run `pre-commit run --all-files` |
| flake8 ANN error | Add type annotations to function signatures |
| pylint score < 5 | Address pylint-reported issues; common disables for transformers: `unused-argument`, `invalid-name`, `too-many-ancestors`, `no-member` |
| Coverage < 80% | Add tests for uncovered branches — check `--cov-report term-missing` output |
| isort error | Run `pre-commit run --all-files` or `uv run isort src/kamae --profile black` |

## CI matrix

CI tests across a matrix — local `make test` runs one combination. The full matrix:
- Python: 3.10, 3.11, 3.12
- PySpark: 3.4.1, 3.5.0
- Keras: 3.3.0, 3.7.0, 3.10.0, 3.12.0
