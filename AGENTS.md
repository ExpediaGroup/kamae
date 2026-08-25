# Kamae

Kamae is a Python library that bridges Apache Spark preprocessing and Keras 3 model serving, eliminating training/serving skew by providing paired Spark transformers and Keras layers.

## Architecture

```
src/kamae/
  keras/
    core/
      base.py              # BaseLayer — all layers extend this
      layers/              # Multi-backend layers (Keras ops only, all backends)
      utils/
    tensorflow/
      layers/              # TF-only layers (string, datetime ops — TensorFlow backend only)
  spark/
    transformers/
      base.py              # BaseTransformer — all transformers extend this
    estimators/
      base.py              # BaseEstimator — for transforms needing fit()
    pipeline/              # KamaeSparkPipeline — chains transformers for export
    params/
      base.py              # I/O param mixins (SingleInput*, MultiInput*)
      shared.py            # Reusable param classes (MathFloatConstantParams, etc.)
    utils/                 # Transform helpers, UDFs, array utils
  graph/                   # Pipeline DAG construction (NetworkX)
  discovery.py             # get_compatible_layers(), get_compatible_transformers()
tests/kamae/               # Mirrors src/ structure
examples/spark/            # Example pipelines
docs/                      # Contributor guides
```

## Naming conventions

| Concept | Class name | File |
|---|---|---|
| Keras layer | `<X>Layer` | `keras/core/layers/<x>.py` or `keras/tensorflow/layers/<x>.py` |
| Spark transformer | `<X>Transformer` | `spark/transformers/<x>.py` |
| Spark estimator | `<X>Estimator` | `spark/estimators/<x>.py` |
| Params class | `<X>Params` | `spark/params/shared.py` or inline |

Use the verb stem: `StringIndex` not `StringIndexer`. File names are `snake_case`.

## Backend placement

- Uses only Keras 3 ops (`keras.ops`) -> `src/kamae/keras/core/layers/`
- Requires TensorFlow ops (strings, datetimes) -> `src/kamae/keras/tensorflow/layers/`

## Param mixins

From `src/kamae/spark/params/base.py`:
- `SingleInputSingleOutputParams` — one input col, one output col
- `SingleInputMultiOutputParams` — one input col, multiple output cols
- `MultiInputSingleOutputParams` — multiple input cols, one output col
- `MultiInputMultiOutputParams` — multiple input cols, multiple output cols

Shared param classes in `src/kamae/spark/params/shared.py` (e.g., `MathFloatConstantParams`, `StandardScaleParams`).

## Registration

When adding new classes, update these `__init__.py` files (alphabetical order):

| What | File |
|---|---|
| Core layer | `src/kamae/keras/core/layers/__init__.py` + `__all__` list |
| TF-only layer | `src/kamae/keras/tensorflow/layers/__init__.py` + `__all__` list |
| Transformer | `src/kamae/spark/transformers/__init__.py` |
| Estimator | `src/kamae/spark/estimators/__init__.py` |

Also add a serialisation test entry in `tests/kamae/keras/test_layer_serialisation.py`.

## Commands

| Command | Purpose |
|---|---|
| `make install` | Install dependencies (uses uv) |
| `make test` | Run tests in parallel (`pytest -n auto`) |
| `make test-cov` | Run tests with 80% coverage threshold (branch coverage) |
| `make lint` | flake8 (annotation checks) + pylint (fail-under 5) |
| `make format` | Check formatting (Black + isort, check-only) |
| `make build` | Build wheel |
| `make run-example` | Run example Spark pipeline |
| `make docs` | Build and serve docs locally |
| `pre-commit run --all-files` | Fix formatting, imports, license headers |

## Code style

- **Formatter:** Black (88 char line length)
- **Imports:** isort with black profile
- **Linting:** flake8 with ANN checks, pylint >= 5.0
- **License:** Apache 2.0 headers on all `.py` files (enforced by pre-commit)
- **Commits:** Conventional commits — `fix:` (patch), `feat:` (minor), `BREAKING CHANGE:` (major)
- **Coverage:** 80% minimum with branch coverage
- **Python:** 3.10+ for development
- **Deps:** managed with uv (`uv sync`)

Common pylint disables at top of transformer/estimator files:
```python
# pylint: disable=unused-argument
# pylint: disable=invalid-name
# pylint: disable=too-many-ancestors
# pylint: disable=no-member
```

## Reference implementations

| Complexity | Example | Files |
|---|---|---|
| Simple (no fit) | Multiply | `spark/transformers/multiply.py`, `keras/core/layers/multiply.py`, `tests/.../test_multiply.py` |
| With estimator | StandardScale | `spark/estimators/standard_scale.py`, `spark/transformers/standard_scale.py`, `keras/core/layers/standard_scale.py` |
| TF-only | StringIndex | `spark/transformers/string_index.py`, `keras/tensorflow/layers/string_index.py` |

## Key documentation

- `docs/adding_transformer.md` — full guide with code examples and checklists
- `docs/achieving_type_parity.md` — ensuring consistent dtypes between Spark and Keras
- `docs/achieving_shape_parity.md` — ensuring consistent shapes between Spark and Keras
- `docs/testing_inference.md` — validating outputs with TensorFlow Serving
- `.github/PULL_REQUEST_TEMPLATE.md` — PR checklist

## Agent skills

This repository uses agent-agnostic skills in `.agents/skills/`.

Before starting a task, inspect the available skills by reading the `name` and `description`
frontmatter in each `.agents/skills/*/SKILL.md`.

If a skill matches the task, read the full `SKILL.md` before acting. Use any referenced
files under that skill only when needed.

Project skills take precedence over user/global skills.
