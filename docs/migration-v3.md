# Migration Guide: v2 to v3

This guide covers all breaking changes in v3.0.0. Each section shows the old and new patterns.

## Keras 2 to Keras 3

Kamae has been migrated from Keras 2 (tf.keras) to Keras 3, enabling multi-backend support.

### Multi-Backend Architecture

Kamae now supports three backends: **TensorFlow**, **JAX**, and **PyTorch**.

```python
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'  # or 'jax' or 'torch'

import keras
from kamae.keras.core.layers import AbsoluteValueLayer  # Works on all backends
```

### Package Structure

```
kamae/
├── keras/
│   ├── core/                    # Backend-agnostic layers (numeric ops)
│   │   ├── base.py              # Unified BaseLayer
│   │   ├── layers/              # 31 multi-backend layers
│   │   └── utils/               # Backend-agnostic utilities
│   └── tensorflow/              # TensorFlow-specific layers
│       ├── layers/              # 36 TF-only layers (strings, datetime)
│       └── utils/               # TF-specific utilities
├── spark/                       # Spark transformers (unchanged)
├── graph/                       # Pipeline graph (now backend-agnostic)
└── utils/                       # General utilities
```

**Removed:**
- `kamae.tensorflow.layers/` — moved to `kamae.keras.core.layers/` or `kamae.keras.tensorflow.layers/`
- `kamae.sklearn/` — removed (was experimental, not maintained)

### Model Serialization

Keras 3 uses `.keras` format (replaces `.h5`):

```python
# OLD (Keras 2)
model.save("path/to/model")
model = tf.keras.models.load_model("path/to/model")

# NEW (Keras 3)
model.save("model.keras")
model = keras.models.load_model("model.keras")
```

### Import Changes

```python
# OLD (Keras 2)
import tensorflow as tf
from kamae.tensorflow.layers import AbsoluteValueLayer

layer = AbsoluteValueLayer()
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# NEW (Keras 3)
import keras
from kamae.keras.core.layers import AbsoluteValueLayer

layer = AbsoluteValueLayer()
model = keras.Model(inputs=inputs, outputs=outputs)
```

### DType Changes

```python
# OLD (Keras 2)
from kamae.utils import DType
dtype = DType.INT
tf_dtype = dtype.tf_dtype  # Returns tf.int32

# NEW (Keras 3)
from kamae.utils import DType
dtype = DType.INT
keras_dtype = dtype.keras_dtype  # Returns "int32" (string)
```

### API Method Renames

| Old Name (Keras 2) | New Name (Keras 3) | Location |
|-------------------|-------------------|----------|
| `get_tf_layer()` | `get_keras_layer()` | All transformers |
| `getInputTFDtype()` | `getInputKerasDtype()` | Transformer parameters |
| `getOutputTFDtype()` | `getOutputKerasDtype()` | Transformer parameters |
| `get_all_tf_layers()` | `get_all_keras_layers()` | PipelineModel |
| `tf_input_schema` parameter | `input_schema` parameter | build_keras_model() |

---

## Spark Param Renames

### `mathFloatConstant` replaced with domain-specific names

The generic `mathFloatConstant` param has been replaced with meaningful names per transformer:

| Old Param | New Param | Transformer |
|-----------|-----------|-------------|
| `mathFloatConstant` | `divisor` | `DivideTransformer` |
| `mathFloatConstant` | `multiplier` | `MultiplyTransformer` |
| `mathFloatConstant` | `addend` | `SumTransformer` |
| `mathFloatConstant` | `subtrahend` | `SubtractTransformer` |
| `mathFloatConstant` | `meanConstant` | `MeanTransformer` |
| `mathFloatConstant` | `minConstant` | `MinTransformer` |
| `mathFloatConstant` | `maxConstant` | `MaxTransformer` |

Before:
```python
MultiplyTransformer(inputCol="x", outputCol="y", mathFloatConstant=5.0)
```

After:
```python
MultiplyTransformer(inputCol="x", outputCol="y", multiplier=5.0)
```

### `stddev` renamed to `variance`

Affects: `StandardScaleTransformer`, `ConditionalStandardScaleTransformer`, and their estimators.

The `stddev` param has been replaced with `variance`. The transformer now stores variance directly and computes the scale factor as `1/sqrt(variance)`. Estimators now use Spark's `F.var_pop()` to compute variance directly rather than computing stddev and squaring it.

Before:
```python
StandardScaleTransformer(inputCol="x", outputCol="y", mean=[0.0], stddev=[2.0])
transformer.getStddev()  # [2.0]
```

After:
```python
StandardScaleTransformer(inputCol="x", outputCol="y", mean=[0.0], variance=[4.0])
transformer.getVariance()  # [4.0]
```

### `labelsArray` renamed to `vocabulary`

Affects: `StringIndexTransformer`, `SharedStringIndexTransformer`, `OneHotEncodeTransformer`, `SharedOneHotEncodeTransformer`, and their estimators.

Before:
```python
StringIndexTransformer(inputCol="x", outputCol="y", labelsArray=["a", "b"])
```

After:
```python
StringIndexTransformer(inputCol="x", outputCol="y", vocabulary=["a", "b"])
```

---

## Mixin Classes Removed

All Spark param mixin classes have been removed. If you subclassed or directly used any of these, switch to `ParamSpec` dicts or shared param dicts:

**Removed mixins:** `UnixTimestampParams`, `DefaultIntValueParams`, `MaskValueParams`, `DropUnseenParams`, `StringIndexParams`, `MathFloatConstantParams`, `ListwiseParams`, `NanFillValueParams`, `StandardScaleSkipZerosParams`, `AutoBroadcastParams`, `StringConstantParams`, `NegationParams`, `StringRegexParams`, `ConstantStringArrayParams`, `HashIndexParams`, `PadValueParams`, `ImputeMethodParams`, `DateTimeParams`, `LatLonConstantParams`, `MaskStringValueParams`, `ListwiseStatisticsParams`, `SampleFractionParams`

**Replacement:** Use `ParamSpec` in your `_params` dict. For shared params, spread the shared dicts:

```python
from kamae.params import ParamSpec, _UNSET
from kamae.params.shared_specs import MASK_VALUE_PARAMS

class MyTransformer(BaseTransformer, SingleInputSingleOutputParams):
    _params = {
        **MASK_VALUE_PARAMS,
        "myParam": ParamSpec(spark_typeconverter=TypeConverters.toFloat, default=_UNSET, doc="My custom param"),
    }
```

Available Spark shared dicts: `UNIX_TIMESTAMP_PARAMS`, `DEFAULT_INT_VALUE_PARAMS`, `MASK_VALUE_PARAMS`, `MASK_STRING_VALUE_PARAMS`, `DROP_UNSEEN_PARAMS`, `STRING_INDEX_PARAMS`, `STANDARD_SCALE_PARAMS`, `LAT_LON_CONSTANT_PARAMS`, `SAMPLE_FRACTION_PARAMS`, `LISTWISE_PARAMS`.

Available Keras shared dicts: `MASK_VALUE_PARAMS`, `UNIX_TIMESTAMP_PARAMS`, `STRING_INDEX_PARAMS`, `DROP_UNSEEN_PARAMS`, `LISTWISE_PARAMS`, `LISTWISE_SEGMENT_PARAMS`.

---

## Keras Layer Changes

### `_layer_params` renamed to `_params`

The Keras layer attribute `_layer_params` has been renamed to `_params` for symmetry with the Spark side.

Before:
```python
class MyLayer(BaseLayer):
    _layer_params = {
        "alpha": KerasParamSpec(default=0.0),
    }
```

After:
```python
class MyLayer(BaseLayer):
    _params = {
        "alpha": ParamSpec(default=0.0),
    }
```

### `_init_validator` renamed to `_post_init`

The `_init_validator` staticmethod has been renamed to `_post_init` and is now a regular method (not a staticmethod).

Before:
```python
@staticmethod
def _init_validator(self):
    if self.lo >= self.hi:
        raise ValueError("lo must be less than hi")
```

After:
```python
def _post_init(self):
    if self.lo >= self.hi:
        raise ValueError("lo must be less than hi")
```

### Positional arguments removed

Layers that previously accepted custom params as positional arguments now require keyword arguments. The codegen `__init__` only accepts `name`, `input_dtype`, `output_dtype` positionally.

Before:
```python
DateParseLayer("MonthOfYear")
```

After:
```python
DateParseLayer(date_part="MonthOfYear")
```

---

## Spec Class Renames

| Old | New | Module Path |
|-----|-----|-------------|
| `kamae.spark.params.param_spec.ParamSpec` | `kamae.params.ParamSpec` | Unified location |
| `kamae.keras.core.layer_spec.LayerParamSpec` | `kamae.params.ParamSpec` | Unified location |
| `kamae.keras.core.layer_spec` | `kamae.keras.core.param_spec` | (module renamed) |

Both Spark and Keras now use the same `ParamSpec` from `kamae.params`.

Before:
```python
from kamae.keras.core.layer_spec import LayerParamSpec
from kamae.spark.params.param_spec import ParamSpec
```

After:
```python
from kamae.params import ParamSpec
```

---

## Migration Checklist

### For Users

- [ ] Update model save/load to use `.keras` extension
- [ ] Change `tf.keras` imports to `keras`
- [ ] Update `tf.keras.models.load_model()` to `keras.models.load_model()`
- [ ] Set `KERAS_BACKEND` environment variable if not using TensorFlow
- [ ] Update `tf_input_schema` parameter to `input_schema` in `build_keras_model()` calls
- [ ] Rename `mathFloatConstant` to operation-specific param names
- [ ] Rename `stddev` to `variance` (value = old_stddev ** 2)
- [ ] Rename `labelsArray` to `vocabulary`
- [ ] Update `getStddev()` calls to `getVariance()`

### For Contributors

- [ ] Use `kamae.keras.core.layers` for new numeric operations (multi-backend)
- [ ] Use `kamae.keras.tensorflow.layers` for string/datetime operations (TF-only)
- [ ] Import from `kamae.keras.core.base.BaseLayer` (not `kamae.tensorflow.layers.base`)
- [ ] Use `keras.ops` for numeric operations (not `tf.math`)
- [ ] Return string dtypes from `compatible_dtypes` property (not tf.DType objects)
- [ ] Use `get_keras_layer()` instead of `get_tf_layer()` in transformer implementations
- [ ] Use `_params` dict (not mixin classes) for new transformers
- [ ] See [codegen.md](codegen.md) for declarative patterns
