# Kamae v3.0.0 Migration Guide

This guide covers everything you need to upgrade from Kamae 2.x to 3.0.0.

## What's in v3.0.0

v3.0.0 is a single coordinated breaking-change release. The headline change is **Keras 3 with multi-backend support**, but several other breaking changes ride along:

1. **Keras 3 multi-backend layers** — numeric layers run on TensorFlow, JAX, or PyTorch; string and datetime layers remain TF-only.
2. **Package layout reorganised** — `kamae.tensorflow.*` is gone; layers are split between `kamae.keras.core.layers` (multi-backend) and `kamae.keras.tensorflow.layers` (TF-only).
3. **`kamae.sklearn` removed** — the experimental scikit-learn integration is no longer shipped.
4. **Deprecated layers removed** — `StringEqualsIfStatementLayer`/`Transformer` and the `OneHotLayer` alias are gone.
5. **API renames** — every TF-prefixed method/parameter now has a backend-agnostic name (`get_tf_layer` → `get_keras_layer`, etc.).
6. **`DType.tf_dtype` → `DType.keras_dtype`** — returns a string (`"float32"`) rather than `tf.dtypes.DType`.
7. **Hash indexer null behaviour changed** — nulls now route to bin 0; non-null indices shift up by one. Models trained on a 2.x hash indexer must be retrained.
8. **Layer name may equal output name** — the previously-injected `IdentityLayer` for matching names is no longer added. Re-export any model that included it.
9. **Discovery API added** — `kamae.discovery` exposes helpers for finding backend-compatible layers and transformers.

If you only do one thing, do this: re-export every model you trained on 2.x. The wire format and several index-mapping behaviours have changed.

## Quick before / after

```python
# 2.x
import tensorflow as tf
from kamae.tensorflow.layers import AbsoluteValueLayer
from kamae.spark.pipeline import KamaeSparkPipeline

pipeline_model = pipeline.fit(df).build_keras_model(tf_input_schema=schema)
pipeline_model.save("model")  # SavedModel directory
loaded = tf.keras.models.load_model("model")
```

```python
# 3.0.0
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # or "jax" or "torch"

import keras
from kamae.keras.core.layers import AbsoluteValueLayer
from kamae.spark.pipeline import KamaeSparkPipeline

pipeline_model = pipeline.fit(df).build_keras_model(input_schema=schema)
pipeline_model.save("model.keras")  # single .keras file
loaded = keras.models.load_model("model.keras")
```

## Breaking changes

### 1. Package layout

`kamae.tensorflow` no longer exists. Layers are split based on whether they need TensorFlow at runtime:

```
kamae/
├── keras/
│   ├── core/                  # backend-agnostic
│   │   ├── base.py            # BaseLayer (unified for both packages)
│   │   ├── backend.py         # ALL_BACKENDS, TENSORFLOW_ONLY, require_tensorflow()
│   │   ├── layers/            # 33 multi-backend layers (numeric, array, logical, ifs, scaling)
│   │   └── utils/             # input_utils, ops_utils, shape_utils, tensor_utils
│   └── tensorflow/
│       ├── layers/            # 35 TF-only layers (strings, datetime, list ops, encoders)
│       └── utils/             # date_utils, list_utils, transform_utils, typing
├── spark/                     # Spark transformers and estimators
├── graph/                     # PipelineGraph (now backend-agnostic)
└── discovery.py               # new in v3.0.0
```

**Import migration table:**

| 2.x import | 3.0.0 import |
| --- | --- |
| `from kamae.tensorflow.layers import AbsoluteValueLayer` | `from kamae.keras.core.layers import AbsoluteValueLayer` |
| `from kamae.tensorflow.layers import StringIndexLayer` | `from kamae.keras.tensorflow.layers import StringIndexLayer` |
| `from kamae.tensorflow.layers.base import BaseLayer` | `from kamae.keras.core.base import BaseLayer` |
| `from kamae.tensorflow.utils import enforce_single_tensor_input` | `from kamae.keras.core.utils.input_utils import enforce_single_tensor_input` |
| `from kamae.tensorflow.typing import Tensor` | `from kamae.keras.tensorflow.utils.typing import Tensor` (only in TF-only code; in multi-backend code use `keras.KerasTensor`) |

Multi-backend layers (in `kamae.keras.core.layers`): `AbsoluteValue`, `ArrayConcatenate`, `ArrayCrop`, `ArrayReduceMax` *(new)*, `ArraySplit`, `ArraySubtractMinimum`, `BearingAngle`, `Bin`, `ConditionalStandardScale`, `CosineSimilarity`, `Divide`, `Exp`, `Exponent`, `HaversineDistance`, `Identity`, `Impute`, `Log`, `LogicalAnd`, `LogicalNot`, `LogicalOr`, `Max`, `Mean`, `Min`, `MinMaxScale`, `Modulo`, `Multiply`, `NumericalIfStatement`, `PairwiseCosineSimilarity` *(new)*, `Round`, `RoundToDecimal`, `StandardScale`, `Subtract`, `Sum`.

TF-only layers (in `kamae.keras.tensorflow.layers`): `BloomEncode`, `Bucketize`, `CurrentDate`, `CurrentDateTime`, `CurrentUnixTimestamp`, `DateAdd`, `DateDiff`, `DateParse`, `DateTimeToUnixTimestamp`, `HashIndex`, `IfStatement`, `LambdaFunction`, `ListMax`, `ListMean`, `ListMedian`, `ListMin`, `ListRank`, `ListStdDev`, `MinHashIndex`, `OneHotEncode`, `OrdinalArrayEncode`, `SharedOneHotEncode`, `SharedStringIndex`, `StringAffix`, `StringArrayConstant`, `StringCase`, `StringConcatenate`, `StringContains`, `StringContainsList`, `StringIndex`, `StringIsInList`, `StringListToString`, `StringMap`, `StringReplace`, `StringToStringList`, `SubStringDelimAtIndex`, `UnixTimestampToDateTime`.

### 2. `kamae.sklearn` removed

The experimental `kamae.sklearn` package (transformers, estimators, pipeline, params) and its examples are deleted in v3.0.0. There is no in-place replacement — sklearn pipelines were never on a Spark→Keras export path, and the module had been unmaintained.

If you depended on `kamae.sklearn`, your options are:
- Pin to `kamae<3.0.0` and continue using 2.x.
- Replicate the small handful of transformers (`ArrayConcatenate`, `ArraySplit`, `Identity`, `Log`, `StandardScale`) directly in your own code — they are straightforward sklearn `BaseEstimator`/`TransformerMixin` wrappers.

### 3. Deprecated public API removed

PR #50 removed three deprecated symbols. They were emitting `DeprecationWarning` in 2.x; in 3.0.0 they raise `ImportError`.

| Removed | Replacement |
| --- | --- |
| `kamae.tensorflow.layers.StringEqualsIfStatementLayer` | `kamae.keras.tensorflow.layers.IfStatementLayer` (handles both string and numeric comparisons via `condition_operator`) |
| `kamae.spark.transformers.StringEqualsIfStatementTransformer` | `kamae.spark.transformers.IfStatementTransformer` |
| `kamae.tensorflow.layers.OneHotLayer` (alias) | `kamae.keras.tensorflow.layers.OneHotEncodeLayer` (the underlying class — same constructor signature) |

Example `IfStatementTransformer` swap:

```python
# 2.x
StringEqualsIfStatementTransformer() \
    .setValueToCompare("a") \
    .setResultIfTrue("TRUE") \
    .setResultIfFalse("FALSE") \
    .setInputCol("col4") \
    .setOutputCol("col4_if")

# 3.0.0
IfStatementTransformer() \
    .setConditionOperator("eq") \
    .setValueToCompare("a") \
    .setResultIfTrue("TRUE") \
    .setResultIfFalse("FALSE") \
    .setInputCol("col4") \
    .setOutputCol("col4_if")
```

### 4. API renames (no aliases kept)

Every TF-prefixed method/parameter has been renamed for backend-agnostic naming. There are no compatibility aliases — calls to the old names raise `AttributeError`.

| 2.x | 3.0.0 | Where |
| --- | --- | --- |
| `Transformer.get_tf_layer()` | `Transformer.get_keras_layer()` | every Spark transformer subclass |
| `getInputTFDtype()` | `getInputKerasDtype()` | `HasInputDtype` mixin |
| `getOutputTFDtype()` | `getOutputKerasDtype()` | `HasOutputDtype` mixin |
| `KamaeSparkPipelineModel.get_all_tf_layers()` | `KamaeSparkPipelineModel.get_all_keras_layers()` | pipeline model |
| `pipeline.build_keras_model(tf_input_schema=...)` | `pipeline.build_keras_model(input_schema=...)` | `PipelineGraph.build_keras_model` and `build_keras_inputs` |

`build_keras_model` also drops support for `tf.TypeSpec` input schemas — only the list-of-dicts form is supported now (the dicts are forwarded as `**kwargs` to `keras.Input`):

```python
input_schema = [
    {"name": "col1", "shape": (None,), "dtype": "float32"},
    {"name": "col4", "shape": (None,), "dtype": "string"},
]
model = pipeline.fit(df).build_keras_model(input_schema=input_schema)
```

If you have custom transformers, rename your `get_tf_layer` override to `get_keras_layer` and switch the body from `getInputTFDtype()`/`getOutputTFDtype()` to `getInputKerasDtype()`/`getOutputKerasDtype()`.

### 5. `DType` enum: `tf_dtype` → `keras_dtype`

`kamae.utils.DType` no longer exposes `tf_dtype` (which returned a `tf.dtypes.DType`). It now exposes `keras_dtype`, which returns a string suitable for `keras.Input(dtype=...)`, `ops.cast(..., dtype=...)`, etc.

```python
# 2.x
from kamae.utils import DType
DType.FLOAT.tf_dtype      # tf.float32

# 3.0.0
from kamae.utils import DType
DType.FLOAT.keras_dtype   # "float32"
```

The Spark-side attributes (`spark_dtype`, `dtype_name`, `bytes`, `is_floating`, `is_integer`) are unchanged.

### 6. Hash indexer null behaviour (PR #41)

In 2.x, `hash_udf` returned `None` for `null` input rows and otherwise returned `hash_val % num_bins`. When a `mask_value` was provided, the mask reserved index 0 and other values were placed in `(hash_val % (num_bins - 1)) + 1`.

In 3.0.0, `hash_udf` always returns an `int`:
- `null` → `0`
- `mask_value` → `0`
- everything else → `(hash_val % (num_bins - 1)) + 1`

Index 0 is always reserved for the null/mask bucket, regardless of whether `mask_value` was set. `numBins` must be `> 1`. The `Hashing` layer used inside `BloomEncodeLayer` is created with `num_bins - 1` and the result has `+1` added so the layer-side encoding matches.

**Impact:** every non-null token's index changes (or its index becomes the same as before but the model's lookup tables shift by one). Models trained against a 2.x hash indexer **must be retrained** against a 3.0.0 indexer; loading a 2.x model and running 3.0.0 preprocessing will silently produce wrong predictions because index 5 in 2.x is index 6 in 3.0.0.

Action:
- Re-fit any pipeline containing `HashIndexEstimator`/`HashIndexTransformer` or `BloomEncodeLayer`.
- Update any feature-store or downstream code that hard-codes the meaning of index 0 — it is now always "null/mask".
- If you were relying on null passing through as `null`, switch to filling nulls explicitly upstream (e.g. with an `ImputeTransformer`) before the hash indexer.

### 7. Layer name may equal output name (PR #42)

In 2.x, if a stage's `layerName` matched its `outputCol` the pipeline graph would reject the configuration, and as a workaround it injected an `IdentityLayer` named `<outputCol>` for every model output.

In 3.0.0, the graph allows `layerName == outputCol` and no longer adds the trailing identity layer.

**Impact:** `keras.Model.get_layer(...)` results change. A model that previously had an `IdentityLayer` named after its output column no longer does — `model.get_layer("my_output")` will return the layer that produced the output instead of the identity. Saved 2.x models that contain that identity layer will fail to load in 3.0.0 because the layer is no longer registered.

Action: re-export every model. Update any introspection code that walks the model layer-by-layer and assumed a final identity for each output.

### 8. Backend selection

Set `KERAS_BACKEND` *before* importing `keras`:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow", "torch"

import keras
from kamae.keras.core.layers import SumLayer
```

Multi-backend layers raise a clear `RuntimeError` if you accidentally use them on an unsupported backend (today there are none — all multi-backend layers support all three backends). TF-only layers call `require_tensorflow()` in their constructor and raise:

```
RuntimeError: This layer requires TensorFlow backend. Current backend: jax. Set KERAS_BACKEND=tensorflow before importing keras.
```

If you only use string and datetime layers, leave the backend on `tensorflow` (the default) — switching to JAX or PyTorch buys you nothing in that case.

## New capabilities (non-breaking, but worth flagging)

### Discovery API

```python
from kamae.discovery import (
    get_compatible_layers,
    get_compatible_transformers,
    get_jit_compatible_layers,
    get_jit_compatible_transformers,
)

get_compatible_layers("jax")            # {name: class} layers usable on JAX
get_jit_compatible_transformers()       # transformers whose layers are jit-safe
```

Use this for tooling that filters supported layers per backend or that wants to gate JIT compilation.

### Per-class capability metadata

Every `BaseLayer` and `BaseTransformer` subclass exposes:

- `supported_backends: FrozenSet[str]` — `ALL_BACKENDS` or `TENSORFLOW_ONLY` (importable from `kamae.keras.core.backend`).
- `jit_compatible: bool` — whether the layer can run inside `keras.ops` JIT compilation.

These are read by the discovery functions but you can read them directly too.

### New layers

- `ArrayReduceMaxLayer` / `ArrayReduceMaxTransformer` — reduces an array column to its maximum element.
- `PairwiseCosineSimilarityLayer` / `PairwiseCosineSimilarityTransformer` — cosine similarity between a query embedding and N candidates packed in a flat array.

Both ship as multi-backend.

## Migration checklist

For users:

- [ ] Set `KERAS_BACKEND` before importing `keras` (only required if you want non-TF backends).
- [ ] Replace `tf.keras` imports with `keras`.
- [ ] Replace `kamae.tensorflow.layers` imports — multi-backend numeric layers move to `kamae.keras.core.layers`, string/datetime layers move to `kamae.keras.tensorflow.layers`.
- [ ] Save with `.keras` extension; load with `keras.models.load_model`.
- [ ] Rename the `tf_input_schema` keyword to `input_schema` and switch any `tf.TypeSpec` schemas to dict form.
- [ ] Re-export every model trained on 2.x — the wire format changed and the layer-name-equals-output-name fix means saved graphs differ.
- [ ] Re-fit and re-export any pipeline using `HashIndex*` or `BloomEncodeLayer` — index 0 is now reserved for null/mask.
- [ ] Drop `kamae.sklearn` usage.
- [ ] Replace `StringEqualsIfStatement*` with `IfStatement*` (`condition_operator="eq"`) and `OneHotLayer` with `OneHotEncodeLayer`.
- [ ] If you read the `DType` enum, switch `.tf_dtype` to `.keras_dtype` (returns string).

For contributors:

- [ ] New numeric layers go in `src/kamae/keras/core/layers/` and inherit from `kamae.keras.core.base.BaseLayer`.
- [ ] New string or datetime layers go in `src/kamae/keras/tensorflow/layers/` (set `supported_backends = TENSORFLOW_ONLY`).
- [ ] Use `@keras.saving.register_keras_serializable(package=kamae.__name__)`, not `@tf.keras.utils.register_keras_serializable`.
- [ ] Use `keras.ops` for math, not `tf.math` (multi-backend layers only).
- [ ] Return `List[str]` from `compatible_dtypes`, not `List[tf.dtypes.DType]`.
- [ ] Set `supported_backends` and `jit_compatible` class attributes on the layer and on the transformer.
- [ ] Override `get_keras_layer` (not `get_tf_layer`) on Spark transformers; use `getInputKerasDtype()`/`getOutputKerasDtype()` inside it.
- [ ] Tests live alongside source: `tests/kamae/keras/core/layers/` for multi-backend, `tests/kamae/keras/tensorflow/layers/` for TF-only.

## Verification

After upgrading, run:

```bash
KERAS_BACKEND=tensorflow uv run pytest tests/
make lint
```

Smoke-check the runtime contract:

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from kamae.discovery import get_compatible_layers, get_compatible_transformers

assert "ArrayReduceMaxLayer" in get_compatible_layers("tensorflow")
assert "ArrayReduceMaxLayer" in get_compatible_layers("jax")

# Ensure null routes to bin 0 in the new hash indexer:
from kamae.spark.utils.user_defined_functions import hash_udf
assert hash_udf(None, num_bins=8) == 0
assert hash_udf("mask", num_bins=8, mask_value="mask") == 0
assert 1 <= hash_udf("foo", num_bins=8) <= 7

# Ensure save/load roundtrips on the new format:
import tempfile, pathlib
m = keras.Sequential([keras.Input(shape=(3,)), keras.layers.Dense(1)])
with tempfile.TemporaryDirectory() as d:
    p = pathlib.Path(d) / "m.keras"
    m.save(p)
    keras.models.load_model(p)
```

## Resources

- [Keras 3 documentation](https://keras.io/)
- [Keras 3 migration guide (upstream)](https://keras.io/keras_3/)
- Adding new layers: [docs/adding_transformer.md](adding_transformer.md)
- Chaining preprocessing with trained models: [docs/chaining_models.md](chaining_models.md)
