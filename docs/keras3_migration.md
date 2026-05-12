# Keras 3 Migration Guide

This document summarizes the migration of Kamae to Keras 3.

## Overview

Kamae has been migrated from Keras 2 (tf.keras) to Keras 3, enabling multi-backend support while maintaining full backward compatibility for existing TensorFlow-based workflows.

## Key Changes

### 1. Multi-Backend Architecture

Kamae now supports three backends: **TensorFlow**, **JAX**, and **PyTorch**.

```python
# Set backend before importing keras
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'  # or 'jax' or 'torch'

import keras
from kamae.keras.core.layers import AbsoluteValueLayer  # Works on all backends
```

### 2. Package Structure

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
- `kamae.tensorflow.layers/` - moved to `kamae.keras.core.layers/` or `kamae.keras.tensorflow.layers/`
- `kamae.sklearn/` - removed (was experimental, not maintained)

### 3. Layer Categories

#### Multi-Backend Layers (31 layers)
Located in `kamae.keras.core.layers/`, work on TensorFlow, JAX, and PyTorch:

- **Numeric operations**: AbsoluteValue, Divide, Exp, Exponent, Log, Max, Mean, Min, Modulo, Multiply, Subtract, Sum
- **Array operations**: ArrayConcatenate, ArrayCrop, ArraySplit, ArraySubtractMinimum
- **Statistical operations**: StandardScale, MinMaxScale, ConditionalStandardScale, Impute
- **Mathematical operations**: BearingAngle, CosineSimilarity, HaversineDistance
- **Logical operations**: LogicalAnd, LogicalNot, LogicalOr
- **Binning/Rounding**: Bin, Round, RoundToDecimal
- **Control flow**: NumericalIfStatement
- **Utility**: Identity

#### TensorFlow-Only Layers (36 layers)
Located in `kamae.keras.tensorflow.layers/`, require TensorFlow backend:

- **String operations**: StringAffix, StringArrayConstant, StringCase, StringConcatenate, StringContains, StringContainsList, StringEqualsIfStatement, StringIndex, StringIsInList, StringListToString, StringMap, StringReplace, StringToStringList, SubStringDelimAtIndex
- **DateTime operations**: CurrentDate, CurrentDateTime, CurrentUnixTimestamp, DateAdd, DateDiff, DateParse, DateTimeToUnixTimestamp, UnixTimestampToDateTime
- **List operations**: ListMax, ListMean, ListMedian, ListMin, ListRank, ListStdDev
- **Encoding**: BloomEncode, HashIndex, MinHashIndex, OneHotEncode, OrdinalArrayEncode, SharedOneHotEncode, SharedStringIndex
- **Other**: Bucketize, IfStatement, LambdaFunction, SingleFeatureArrayStandardScale

### 4. Model Serialization

**Keras 3 uses `.keras` format** (replaces `.h5`):

```python
# OLD (Keras 2)
model.save("path/to/model")
model = tf.keras.models.load_model("path/to/model")

# NEW (Keras 3)
model.save("model.keras")
model = keras.models.load_model("model.keras")
```

### 5. Import Changes

```python
# OLD (Keras 2)
import tensorflow as tf
from kamae.tensorflow.layers import AbsoluteValueLayer

layer = AbsoluteValueLayer()
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.save("path/to/model")

# NEW (Keras 3)
import keras
from kamae.keras.core.layers import AbsoluteValueLayer

layer = AbsoluteValueLayer()
model = keras.Model(inputs=inputs, outputs=outputs)
model.save("model.keras")
```

### 6. DType Changes

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

### 7. Type Annotations

```python
# OLD (Keras 2)
from typing import Optional, List
import tensorflow as tf

def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
    return [tf.float32, tf.float64]

# NEW (Keras 3 - Multi-backend)
from typing import Optional, List

def compatible_dtypes(self) -> Optional[List[str]]:
    return ["float32", "float64"]
```

### 8. API Method Renames

**Methods renamed for backend-agnostic naming:**

| Old Name (Keras 2) | New Name (Keras 3) | Location |
|-------------------|-------------------|----------|
| `get_tf_layer()` | `get_keras_layer()` | All transformers |
| `getInputTFDtype()` | `getInputKerasDtype()` | Transformer parameters |
| `getOutputTFDtype()` | `getOutputKerasDtype()` | Transformer parameters |
| `get_all_tf_layers()` | `get_all_keras_layers()` | PipelineModel |
| `tf_input_schema` parameter | `input_schema` parameter | build_keras_model() |

## Migration Checklist

### For Users

- [ ] Update model save/load to use `.keras` extension
- [ ] Change `tf.keras` imports to `keras`
- [ ] Update `tf.keras.models.load_model()` to `keras.models.load_model()`
- [ ] Remove Keras 2 vs 3 version checking code
- [ ] Set `KERAS_BACKEND` environment variable if not using TensorFlow
- [ ] Update `tf_input_schema` parameter to `input_schema` in `build_keras_model()` calls

### For Contributors

- [ ] Use `kamae.keras.core.layers` for new numeric operations (multi-backend)
- [ ] Use `kamae.keras.tensorflow.layers` for string/datetime operations (TF-only)
- [ ] Import from `kamae.keras.core.base.BaseLayer` (not `kamae.tensorflow.layers.base`)
- [ ] Use `@keras.saving.register_keras_serializable` decorator (not `tf.keras.utils`)
- [ ] Return string dtypes from `compatible_dtypes` property (not tf.DType objects)
- [ ] Use `keras.ops` for numeric operations (not `tf.math`)
- [ ] Add tests to the corresponding test directory (`tests/kamae/keras/core/layers/` for multi-backend layers, `tests/kamae/keras/tensorflow/layers/` for TF-only layers)
- [ ] Use `get_keras_layer()` instead of `get_tf_layer()` in transformer implementations
- [ ] Use `getInputKerasDtype()` and `getOutputKerasDtype()` instead of TF-prefixed versions

## Backend-Specific String Operations

The `BaseLayer` class supports string operations, but they **only work on TensorFlow backend**:

```python
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from kamae.keras.core.layers import BinLayer

# String output types work on TensorFlow backend
layer = BinLayer(
    condition_operators=["lt", "gt"],
    bin_values=[5, 10],
    bin_labels=["small", "large"],
    default_label="medium"
)
```

If you try to use string dtypes on JAX or PyTorch backends, you'll get a clear error message.

## Testing

All existing tests pass. Test organization now mirrors source structure:
- `tests/kamae/keras/core/layers/` - 32 test files for multi-backend layers
- `tests/kamae/keras/tensorflow/layers/` - 36 test files for TF-only layers

## Backward Compatibility

Spark pipelines continue to work exactly as before:
- All Spark transformers unchanged
- `build_keras_model()` works identically
- Generated Keras models are backward compatible with TensorFlow Serving

## Performance

No performance regressions. Multi-backend layers use `keras.ops` which compiles efficiently on all backends.

## Documentation

All documentation updated:
- README.md - Updated to Keras 3, removed sklearn references
- docs/adding_transformer.md - Updated for Keras 3 layer development
- docs/chaining_models.md - Updated code examples to use `keras` imports
- examples/spark/*.py - All examples updated to Keras 3

## Breaking Changes

1. **Removed sklearn support** - `kamae.sklearn` package removed (was experimental)
2. **Module paths changed**:
   - `kamae.tensorflow.layers` → `kamae.keras.core.layers` or `kamae.keras.tensorflow.layers`
   - `kamae.tensorflow.utils` → `kamae.keras.core.utils` or `kamae.keras.tensorflow.utils`
   - `kamae.tensorflow.typing` → `kamae.keras.tensorflow.utils.typing`
3. **DType enum** - `tf_dtype` attribute renamed to `keras_dtype` (returns string, not tf.DType)
4. **Model format** - Should use `.keras` extension (`.h5` still works but deprecated)
5. **API method names** - All TensorFlow-prefixed methods renamed for backend-agnostic naming:
   - `get_tf_layer()` → `get_keras_layer()`
   - `getInputTFDtype()` → `getInputKerasDtype()`
   - `getOutputTFDtype()` → `getOutputKerasDtype()`
   - `get_all_tf_layers()` → `get_all_keras_layers()`
   - `tf_input_schema` parameter → `input_schema`

## Benefits

1. **Multi-backend support** - Run on TensorFlow, JAX, or PyTorch
2. **Cleaner architecture** - Clear separation between multi-backend and TF-only code
3. **Better maintainability** - Unified BaseLayer, no code duplication
4. **Future-proof** - Built on Keras 3, the future of Keras
5. **Smaller package** - Removed unmaintained sklearn code

## Resources

- [Keras 3 Documentation](https://keras.io/)
- [Keras 3 Migration Guide](https://keras.io/keras_3/)
- [Multi-backend Guide](https://keras.io/guides/distributed_training_with_jax/)
