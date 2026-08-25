---
name: add-transformer
description: Implement a new Spark transformer and paired Keras layer with tests. Includes optional estimator path for transforms needing fit().
---

# Add Transformer

Use this skill when adding a new preprocessing operation to Kamae. Every operation requires a paired Keras layer and Spark transformer. If the operation needs to learn from data (e.g., computing mean/stddev), it also needs an estimator.

## Before you start

Read the reference implementation that matches your case:
- **Simple numeric op:** `src/kamae/spark/transformers/multiply.py` + `src/kamae/keras/core/layers/multiply.py`
- **With estimator:** `src/kamae/spark/estimators/standard_scale.py` + `src/kamae/spark/transformers/standard_scale.py`
- **TF-only (string/datetime):** `src/kamae/spark/transformers/string_index.py` + `src/kamae/keras/tensorflow/layers/string_index.py`

For the full guide with code examples and checklists, read `docs/adding_transformer.md`.

## Step 1: Decide placement

Does your layer need TensorFlow-specific ops (string manipulation, datetime parsing)?
- **Yes** -> `src/kamae/keras/tensorflow/layers/<x>.py`, import `TENSORFLOW_ONLY` from `kamae.keras.core.backend`
- **No** -> `src/kamae/keras/core/layers/<x>.py`, use only `keras.ops`, import `ALL_BACKENDS` from `kamae.keras.core.backend`

## Step 2: Implement the Keras layer

Create `src/kamae/keras/{core or tensorflow}/layers/<x>.py`:

```python
from typing import Any, Dict, List, Optional

import keras
from keras import KerasTensor, ops

import kamae
from kamae.keras.core.backend import ALL_BACKENDS
from kamae.keras.core.base import BaseLayer


@keras.saving.register_keras_serializable(package=kamae.__name__)
class <X>Layer(BaseLayer):
    supported_backends = ALL_BACKENDS  # or TENSORFLOW_ONLY
    jit_compatible = True  # or False

    def __init__(
        self,
        name: Optional[str] = None,
        input_dtype: Optional[str] = None,
        output_dtype: Optional[str] = None,
        # add custom params here
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs)
        # store custom params as instance attributes

    @property
    def compatible_dtypes(self) -> Optional[List[str]]:
        return ["float32", "float64"]  # or None for all dtypes

    def _call(self, inputs: KerasTensor, **kwargs: Any) -> KerasTensor:
        # transform logic using keras.ops
        ...

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({...})  # add custom params
        return config
```

Key points:
- Constructor must accept `name`, `input_dtype`, `output_dtype` and pass them to `super().__init__`
- `compatible_dtypes` returns dtype *strings* (e.g., `"float32"`), not Spark types
- Use `@allow_single_or_multiple_tensor_input` decorator from `kamae.keras.core.utils.input_utils` if the layer accepts single or multiple tensor inputs

## Step 3: Implement the Spark transformer

Create `src/kamae/spark/transformers/<x>.py`:

```python
# pylint: disable=unused-argument
# pylint: disable=invalid-name
# pylint: disable=too-many-ancestors
# pylint: disable=no-member
from typing import List, Optional

import keras
from pyspark import keyword_only
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType, DoubleType, FloatType

from kamae.keras.core.backend import ALL_BACKENDS
from kamae.keras.{core or tensorflow}.layers import <X>Layer
from kamae.spark.params import SingleInputSingleOutputParams  # pick the right mixin
from .base import BaseTransformer


class <X>Transformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    # add shared param classes if needed
):
    supported_backends = ALL_BACKENDS
    jit_compatible = True

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        # add custom params here
    ) -> None:
        super().__init__()
        # self._setDefault(param=default) for custom params with defaults
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        return [FloatType(), DoubleType()]  # Spark types, not strings

    def _transform(self, dataset: DataFrame) -> DataFrame:
        # transform logic
        return dataset.withColumn(self.getOutputCol(), ...)

    def get_keras_layer(self) -> keras.layers.Layer:
        return <X>Layer(
            name=self.getLayerName(),
            input_dtype=self.getInputKerasDtype(),
            output_dtype=self.getOutputKerasDtype(),
            # pass custom params
        )
```

Key points:
- Always include the four pylint disable comments at the top
- `@keyword_only` on `__init__`, then `super().__init__()` then `self.setParams(**kwargs)`
- Custom params need setter methods named `set<ParamName>` — either via a shared params class or defined inline
- `compatible_dtypes` returns Spark `DataType` instances, not strings
- Use transform utilities from `kamae.spark.utils.transform_utils` for nested array support

## Step 4 (if fit needed): Implement the Spark estimator

Create `src/kamae/spark/estimators/<x>.py` (separate file from the transformer):

```python
from kamae.spark.estimators.base import BaseEstimator

class <X>Estimator(BaseEstimator, SingleInputSingleOutputParams, ...):
    # same constructor pattern as transformer

    def _fit(self, dataset: DataFrame) -> "<X>Transformer":
        # compute statistics from dataset
        return <X>Transformer(
            inputCol=self.getInputCol(),
            outputCol=self.getOutputCol(),
            layerName=self.getLayerName(),
            inputDtype=self.getInputDtype(),
            outputDtype=self.getOutputDtype(),
            # pass computed statistics
        )
```

Reference: `src/kamae/spark/estimators/standard_scale.py`

## Step 5: Register

Add imports (alphabetical order) to:

1. **Layer `__init__.py`** — `src/kamae/keras/{core or tensorflow}/layers/__init__.py`
   - Add the import line
   - Add to the `__all__` list
2. **Transformer `__init__.py`** — `src/kamae/spark/transformers/__init__.py`
3. **Estimator `__init__.py`** (if applicable) — `src/kamae/spark/estimators/__init__.py`

## Step 6: Write tests

### Layer tests
Create `tests/kamae/keras/{core or tensorflow}/layers/test_<x>.py`:
- Use `@pytest.mark.parametrize` with multiple input shapes, dtypes, and edge cases
- Assert on output dtype, shape, and values
- Use `tf.debugging.assert_near` for floating-point comparisons

### Transformer tests
Create `tests/kamae/spark/transformers/test_<x>.py`:
- Use the `spark_session` and `example_dataframe` fixtures from `conftest.py`
- Test transform output against expected DataFrames

### Parity tests
Verify Spark transformer and Keras layer produce identical results:
- Transform with Spark, convert output to numpy
- Run Keras layer on same input, convert to numpy
- `np.testing.assert_almost_equal(spark_result, keras_result)`

### Serialisation test
Add an entry to `tests/kamae/keras/test_layer_serialisation.py`:
- Import your layer
- Add a tuple to the `@pytest.mark.parametrize` list: `(YourLayer, [input_tensors], {kwargs}, skip_predict)`
- `skip_predict=True` only for non-deterministic layers (datetime/timestamp)

## Step 7: Verify

```bash
make format && make lint && make test
```

If parity fails, read `docs/achieving_type_parity.md` and `docs/achieving_shape_parity.md`.
