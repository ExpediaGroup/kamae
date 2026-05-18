# Declarative Codegen Reference

Kamae uses declarative class attributes to auto-generate boilerplate (`__init__`, `setParams`, getters/setters, `get_config`, `get_keras_layer`, `compatible_dtypes`). This document describes what the system generates and when to override.

## ParamSpec

Both Spark and Keras use the same `ParamSpec` dataclass from `kamae.params`:

```python
from kamae.params import ParamSpec, _REQUIRED
```

Both Spark and Keras use the same `ParamSpec` directly.

**`ParamSpec` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `default` | `Any` | Default value. Use `_REQUIRED` for mandatory params, `_UNSET` for optional with no default. |
| `doc` | `str` | Documentation string. |
| `type` | `Type` | Python type for PySpark TypeConverter inference. Ignored on Keras side. |
| `type_converter` | `Callable` | Explicit TypeConverter override. Ignored on Keras side. |
| `validator` | `Callable` | Single-field validation. Keras: called during `__init__`. Spark: called in the setter. |

## Keras Layers

### `_params`

Dict mapping snake_case param names to `ParamSpec`. Codegen generates `__init__` and `get_config`.

```python
from kamae.keras.core.base import BaseLayer
from kamae.params import ParamSpec, _REQUIRED

class MyLayer(BaseLayer):
    _params = {
        "alpha": ParamSpec(default=0.0, doc="Smoothing constant"),
        "vocabulary": ParamSpec(default=_REQUIRED, doc="Required vocab list"),
        "mask_value": ParamSpec(default=None, doc="Optional mask"),
    }

    def _call(self, inputs, **kwargs):
        # self.alpha, self.vocabulary, self.mask_value are set by codegen
        ...
```

Layers with no custom params don't need to define `_params` at all — the base class defaults to `{}`.

**What codegen generates:**
- `__init__(self, name=None, input_dtype=None, output_dtype=None, **kwargs)` — pops each param from kwargs, sets as `self.attr`, calls `super().__init__()`, then calls `_post_init` if defined
- `get_config(self)` — calls `super().get_config()`, adds each param
- Keras serialization registration via `@register_keras_serializable`

**Auto camelCase conversion:** If `_params` keys are in camelCase (e.g. from shared specs), `BaseLayer.__init_subclass__` automatically converts them to snake_case. You can define params in either convention and the layer will use snake_case attributes.

**Param inheritance:** `cls._params` contains the merged params from the full class hierarchy. A child class inherits all parent params. `build_keras_layer_from_specs` uses this merged dict to determine which Spark params map to Keras params.

### `_compatible_dtypes`

List of dtype strings, or `None` for any-type. A concrete property on `BaseLayer` exposes it as `compatible_dtypes`.

```python
class MyLayer(BaseLayer):
    _compatible_dtypes = ["float32", "float64"]
```

For TensorFlow-only layers, use string literals (not `tf.string` etc.):
```python
_compatible_dtypes = ["string"]  # not [tf.string]
```

### `_post_init`

A method that runs after all params are set. Use for:
- **Validation**: cross-field checks that raise on invalid combinations
- **Derived attributes**: computing values from multiple params
- **Internal object construction**: creating TF layers like `Hashing`, `StringLookup`
- **Storing originals**: preserving raw values before `build()` overwrites them

```python
class MyLayer(BaseLayer):
    _params = {
        "num_bins": ParamSpec(default=_REQUIRED),
        "mask_value": ParamSpec(default=None),
    }

    def _post_init(self):
        if self.mask_value is not None:
            self.hash_indexer = Hashing(
                num_bins=self.num_bins, mask_value=self.mask_value
            )
        else:
            self.hash_indexer = Hashing(num_bins=self.num_bins - 1)
```

`_post_init` is looked up from `cls.__dict__` — only the class's own definition is called. Parent `_post_init` methods are invoked via the normal `__init__` chain (parent codegen-generated `__init__` calls its own `_post_init`).

### Layers with `build()`

Layers that reshape params into tensors in `build()` must store the original values for serialization. Pattern:

```python
class NormalizeLayer(BaseLayer):
    _params = {
        "mean": ParamSpec(default=_REQUIRED, doc="Mean of the feature values."),
        "variance": ParamSpec(default=_REQUIRED, doc="Variance of the feature values."),
        "axis": ParamSpec(default=-1, doc="Axis for normalization"),
    }

    def _post_init(self):
        # Standardize axis
        if self.axis is None:
            self.axis = ()
        elif isinstance(self.axis, int):
            self.axis = (self.axis,)
        else:
            self.axis = tuple(self.axis)
        # Store originals before build() overwrites self.mean/self.variance
        self.input_mean = self.mean
        self.input_variance = self.variance

    def build(self, input_shape):
        # Reshape to broadcast shape — overwrites self.mean, self.variance
        self.mean = ops.reshape(...)
        self.variance = ops.reshape(...)

    def get_config(self):
        config = super().get_config()
        # Serialize the originals, not the reshaped tensors
        config["mean"] = listify_tensors(self.input_mean)
        config["variance"] = listify_tensors(self.input_variance)
        config["axis"] = list(self.axis) if self.axis else None
        return config
```

Children of such layers inherit the parent's params and only declare their own additional params:

```python
class StandardScaleLayer(NormalizeLayer):
    _params = {**MASK_VALUE_PARAMS}  # Only declares own params

    def _call(self, inputs, **kwargs):
        # self.mean, self.variance available from parent
        # self.mask_value available from own params
        ...
```

### Shared ParamSpec Dicts (Keras)

For params reused across layers, spread shared dicts into `_params`:

```python
from kamae.params.shared_specs import MASK_VALUE_PARAMS

class MyLayer(BaseLayer):
    _params = {
        **MASK_VALUE_PARAMS,
        "my_param": ParamSpec(default=0.0),
    }
```

Available shared dicts in `kamae.params.shared_specs`:

| Dict | Params | Used by |
|------|--------|---------|
| `MASK_VALUE_PARAMS` | `mask_value` | HashIndex, MinHashIndex, BloomEncode, MinMaxScale, StandardScale |
| `UNIX_TIMESTAMP_PARAMS` | `unit` | CurrentUnixTimestamp, DateTimeToUnixTimestamp, UnixTimestampToDateTime |
| `STRING_INDEX_PARAMS` | `vocabulary`, `num_oov_indices`, `mask_token`, `encoding` | StringIndex, OneHotEncode |
| `DROP_UNSEEN_PARAMS` | `drop_unseen` | OneHotEncode |
| `LISTWISE_PARAMS` | `top_n`, `sort_order`, `min_filter_value`, `nan_fill_value`, `axis` | All list layers |
| `LISTWISE_SEGMENT_PARAMS` | `with_segment` | ListMax, ListMean, ListMin |

These are derived from the canonical camelCase definitions in `kamae.params.shared_specs` — `BaseLayer.__init_subclass__` auto-converts keys to snake_case.

### TensorFlow-only layers

TF-only layers live in `src/kamae/keras/tensorflow/layers/` and must set:

```python
from kamae.keras.core.backend import TENSORFLOW_ONLY

class MyTFLayer(BaseLayer):
    supported_backends = TENSORFLOW_ONLY
    _compatible_dtypes = ["string"]
    ...
```

## Spark Transformers

### `_params`

Dict mapping camelCase param names to `ParamSpec`. Codegen generates `Param` objects, getters, setters, and `__init__`/`setParams`.

```python
from kamae.spark.transformers.base import BaseTransformer
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.params import ParamSpec, _UNSET

class MyTransformer(BaseTransformer, SingleInputSingleOutputParams):
    _params = {
        "alpha": ParamSpec(type=float, default=0.0, doc="Smoothing constant"),
    }
    _compatible_dtypes = [FloatType(), DoubleType()]
    _keras_layer_class = MyLayer

    def _transform(self, dataset):
        alpha = self.getAlpha()  # generated getter
        ...
```

Transformers with no custom params don't need to define `_params` at all — the base class defaults to `{}`.

**What codegen generates:**
- `Param` object on the class (e.g. `cls.alpha = Param(...)`)
- `getAlpha(self)` — calls `self.getOrDefault("alpha")`
- `setAlpha(self, value)` — runs validator if present, then `self._set(alpha=value)`
- `__init__(self, **kwargs)` — calls `super().__init__()`, `_setDefault(...)` for params with defaults, `self.setParams(**kwargs)`
- `setParams(self, **kwargs)` — calls each setter

**Sentinels:**
- `_UNSET` — optional param with no sensible default; `_setDefault` sets it to `None`, and `setParams` skips `None` values so the type converter is never invoked on it

### `_compatible_dtypes`

List of PySpark `DataType` instances, or `None` for any-type. A concrete property on `BaseTransformer` exposes it as `compatible_dtypes`.

```python
_compatible_dtypes = [FloatType(), DoubleType()]
```

### `_keras_layer_class`

When set, codegen generates `get_keras_layer()` using `build_keras_layer_from_specs`. It passes `name`, `input_dtype`, `output_dtype`, plus each Spark param (converted from camelCase to snake_case). Params that don't exist on the Keras layer's `_params` (including inherited params) are automatically skipped.

```python
_keras_layer_class = DivideLayer
```

### When to override `get_keras_layer`

Set `_keras_layer_class = None` and implement `get_keras_layer` manually when:

1. **Param names don't align** — Spark param name (camelCase→snake_case) doesn't match the Keras param name.
   ```python
   # Spark: constantStringArray → constant_string_array
   # Keras layer expects: string_constant_list
   def get_keras_layer(self):
       return StringContainsListLayer(
           name=self.getLayerName(),
           string_constant_list=self.getConstantStringArray(),
           ...
       )
   ```

2. **Hardcoded values** — Keras layer has params with no Spark equivalent.
   ```python
   # axis/keepdims are fixed for this transformer
   def get_keras_layer(self):
       return CosineSimilarityLayer(axis=-1, keepdims=True, ...)
   ```

3. **Multiple layers returned** — transformer produces a list of layers (e.g. `SharedStringIndexTransformer`).

### Shared ParamSpec Dicts (Spark)

For params reused across transformers, spread shared dicts into `_params`:

```python
from kamae.params.shared_specs import MASK_VALUE_PARAMS

class MyTransformer(BaseTransformer, SingleInputSingleOutputParams):
    _params = {
        **MASK_VALUE_PARAMS,
        "myParam": ParamSpec(type=float, default=0.0, doc="Custom param"),
    }
```

Available shared dicts in `kamae.params.shared_specs`:

| Dict | Params | Used by |
|------|--------|---------|
| `UNIX_TIMESTAMP_PARAMS` | `unit` | CurrentUnixTimestamp, DateTimeToUnixTimestamp, UnixTimestampToDateTime |
| `DEFAULT_INT_VALUE_PARAMS` | `defaultValue` | DateDiff, DateAdd |
| `MASK_VALUE_PARAMS` | `maskValue` (float) | MinMaxScale, StandardScale estimators |
| `MASK_STRING_VALUE_PARAMS` | `maskValue` (str) | HashIndex, BloomEncode, MinHashIndex |
| `DROP_UNSEEN_PARAMS` | `dropUnseen` | OneHotEncode transformers/estimators |
| `STRING_INDEX_PARAMS` | `vocabulary`, `stringOrderType`, `maskToken`, `numOOVIndices`, `maxNumLabels` | StringIndex, OneHotEncode |
| `STANDARD_SCALE_PARAMS` | `mean`, `variance` | StandardScale, ConditionalStandardScale |
| `LAT_LON_CONSTANT_PARAMS` | `latLonConstant` | BearingAngle, HaversineDistance |
| `SAMPLE_FRACTION_PARAMS` | `sampleFraction` | StandardScale, MinMaxScale, Impute estimators |
| `LISTWISE_PARAMS` | `queryIdCol`, `topN`, `sortOrder` | All list transformers |

### Validators

Spark has two validation patterns:

**Single-field validators** (static, in ParamSpec): Run in the setter. Good for checking one param's value in isolation.

```python
def _validate_splits(value):
    if value != sorted(value):
        raise ValueError("splits must be sorted")
    return value
```

**Cross-field validators** (instance method, in setter): Run in a custom setter. Good for checking param combinations. Spark params are mutable, so cross-field checks must happen in setters to catch both construction-time and runtime mutations.

```python
def _validate_num_days(self, value):
    if self.isDefined("inputCols"):
        raise ValueError("Cannot set numDays with multiple inputCols")
    return value
```

## Spark Estimators

Estimators use the same `_params`, `_compatible_dtypes` system. Define `_params` on the estimator class and implement `_fit`.

## Adding a New Transformer (End-to-End)

1. **Keras layer** — define `_params`, `_compatible_dtypes`, implement `_call`. Add `_post_init` if needed.
2. **Spark transformer** — define `_params`, `_compatible_dtypes`, `_keras_layer_class`, implement `_transform`.
3. **Tests** — unit tests for both, plus Spark/Keras parity tests.

Minimal example:

```python
# src/kamae/keras/core/layers/negate.py
from kamae.keras.core.base import BaseLayer

class NegateLayer(BaseLayer):
    _compatible_dtypes = ["float32", "float64"]

    def _call(self, inputs, **kwargs):
        from keras import ops
        return ops.negative(inputs[0])
```

```python
# src/kamae/spark/transformers/negate.py
from kamae.keras.core.layers import NegateLayer
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.transformers.base import BaseTransformer
from pyspark.sql.types import FloatType, DoubleType

class NegateTransformer(BaseTransformer, SingleInputSingleOutputParams):
    _compatible_dtypes = [FloatType(), DoubleType()]
    _keras_layer_class = NegateLayer

    def _transform(self, dataset):
        import pyspark.sql.functions as F
        return dataset.withColumn(self.getOutputCol(), -F.col(self.getInputCol()))
```

No `__init__`, no `setParams`, no `getConfig`, no `compatible_dtypes` property, no `get_keras_layer` — all generated.
