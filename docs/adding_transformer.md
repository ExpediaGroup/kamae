# Contributing a Keras layer and Spark transformer

Follow this guide to contribute a new transformer to the project.

## Overview
In order to contribute a new transformer, you will need to implement a Spark Transformer, a corresponding Keras layer, and a Spark Estimator if your transformer needs a fit method.
We also require unit tests for all new classes, in particular parity tests ensuring your Spark Transformer and Keras layer produce the same output.

For full details on the declarative codegen system, see [codegen.md](codegen.md).

## Naming
In order to avoid name clashes and to keep consistency, we have a naming convention for all new classes.

If an operation is called `<X>` then:

- `<X>Estimator` = Spark estimator (if applicable)
- `<X>Transformer` = Spark transformer
- `<X>Layer` = Keras layer

We just keep the verb stem. E.g string indexing is StringIndexTransformer, not StringIndexerTransformer.

The name of the file should then be `<X>.py`. E.g. `src/kamae/spark/transformers/string_index.py` and `src/kamae/keras/core/layers/string_index.py` (for multi-backend layers) or `src/kamae/keras/tensorflow/layers/string_index.py` (for TensorFlow-only layers).

Finally, if you need to create an estimator, then the estimator and its corresponding transformer should be in different files. E.g. `src/kamae/spark/transformers/string_index.py` and `src/kamae/spark/estimators/string_index.py`.

## Keras layer
Your Keras layer should extend [BaseLayer](../src/kamae/keras/core/base.py) and implement the `_call` method.

Use `_params` to declare parameters and `_compatible_dtypes` for dtype restrictions. The base class auto-generates `__init__`, `get_config`, `compatible_dtypes`, and Keras serialization registration.

If you need post-construction logic (validation, derived attributes, or creating internal TF layer objects), define a `_post_init` method instead of writing a manual `__init__`.

For params shared across multiple layers (e.g. `mask_value`, `unit`), spread shared dicts from `kamae.params.shared_specs` into `_params`. See [codegen.md](codegen.md) for details.

**Note:** Multi-backend layers should be placed in `src/kamae/keras/core/layers/` and use only Keras 3 operations. TensorFlow-only layers (those requiring TensorFlow-specific operations) should be placed in `src/kamae/keras/tensorflow/layers/` and can import TensorFlow for backend-specific functionality.

### Example

```python
from kamae.keras.core.base import BaseLayer
from kamae.params import ParamSpec, _REQUIRED

class MyLayer(BaseLayer):
    _compatible_dtypes = ["float32", "float64"]
    _params = {
        "my_param": ParamSpec(default=_REQUIRED, doc="A required parameter"),
        "threshold": ParamSpec(default=0.5, doc="Optional threshold"),
    }

    def _post_init(self):
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")

    def _call(self, inputs, **kwargs):
        # self.my_param and self.threshold are set by codegen
        return outputs
```

### When to write manual `__init__`

Almost never. The codegen system handles `__init__` generation for all standard cases including:
- Layers with `_post_init` for validation/derived attributes
- Layers inheriting from intermediate parents (e.g. `NormalizeLayer`)
- Layers with required params, optional params, shared params

The only case requiring manual `__init__` is inheriting from a non-codegen parent that has a non-standard `__init__` signature (e.g. `tf.keras.layers.Lambda`).

### Checklist

- [ ] I have implemented a Keras layer that extends [BaseLayer](../src/kamae/keras/core/base.py)
- [ ] I have defined `_params` with `ParamSpec` entries (from `kamae.params`) for each parameter
- [ ] I have defined `_compatible_dtypes` (list of dtype strings, or `None` for any type)
- [ ] I have implemented the `_call` method
- [ ] I have added `_post_init` if post-construction logic is needed
- [ ] I have unit tests of my implementation
- [ ] I have a specific test of layer serialisation added [here](../tests/kamae/keras/test_layer_serialisation.py)

## Spark Transformer/Estimator
Your Spark Transformer should extend [BaseTransformer](../src/kamae/spark/transformers/base.py) and implement the `_transform` method.

Use `_params` to declare custom parameters with `ParamSpec`. The base class auto-generates `__init__`, `setParams`, getters, setters, and `compatible_dtypes`.

Set `_keras_layer_class` to your Keras layer class to auto-generate `get_keras_layer`.

Your transformer should use one (or more) of the input/output mixin classes from [base.py](../src/kamae/spark/params/base.py):
- `SingleInputSingleOutputParams`
- `SingleInputMultiOutputParams`
- `MultiInputSingleOutputParams`
- `MultiInputMultiOutputParams`

Only use more than one if you want to support two usages of your transformer, e.g. `MyTransformer(inputCol="a", outputCol="b")` and `MyTransformer(inputCols=["a", "b"], outputCols=["c", "d"])`.

We have provided utils for transformers & estimators to natively transform nested Spark array columns.
You can use one of the following functions from [here](../src/kamae/spark/utils/transform_utils.py):

- `single_input_single_output_scalar_transform`
- `single_input_single_output_array_transform`
- `single_input_single_output_scalar_udf_transform`
- `single_input_single_output_array_udf_transform`
- `multi_input_single_output_scalar_transform`
- `multi_input_single_output_array_transform`


### Example
Note that the methods are named `_fit` and `_transform`. `fit` and `transform` wrap these internal methods and should not be overridden.

```python
from pyspark.sql import DataFrame
from pyspark.sql.types import FloatType, DoubleType

from kamae.spark.params import SingleInputSingleOutputParams
from kamae.params import ParamSpec, _UNSET
from kamae.spark.transformers.base import BaseTransformer
from kamae.keras.core.layers import MyLayer


class MyTransformer(BaseTransformer, SingleInputSingleOutputParams):
    _compatible_dtypes = [FloatType(), DoubleType()]
    _keras_layer_class = MyLayer
    _params = {
        "myParam": ParamSpec(
            spark_typeconverter=TypeConverters.toFloat,
            default=0.5,
            doc="A custom parameter",
        ),
    }

    def _transform(self, dataset: DataFrame) -> DataFrame:
        my_param = self.getMyParam()  # auto-generated getter
        # Do some transformation...
        return dataset.withColumn(self.getOutputCol(), output_of_transform)
```

For estimators:

```python
from kamae.spark.estimators.base import BaseEstimator
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.params import ParamSpec

class MyEstimator(BaseEstimator, SingleInputSingleOutputParams):
    _compatible_dtypes = [FloatType(), DoubleType()]
    _params = {
        "myParam": ParamSpec(spark_typeconverter=TypeConverters.toFloat, default=0.5, doc="A custom parameter"),
    }

    def _fit(self, dataset):
        fitted_value = compute_something(dataset)
        return MyTransformer(
            inputCol=self.getInputCol(),
            outputCol=self.getOutputCol(),
            layerName=self.getLayerName(),
            inputDtype=self.getInputDtype(),
            outputDtype=self.getOutputDtype(),
            myParam=fitted_value,
        )
```

### When to write manual `get_keras_layer`

Set `_keras_layer_class = None` and implement `get_keras_layer` when:
- **Param names don't align** — Spark camelCase→snake_case doesn't match the Keras param name (e.g. `constantStringArray` → `string_constant_list`)
- **Hardcoded values** — Keras layer has params with no Spark equivalent (e.g. `axis=-1, keepdims=True`)
- **Multiple layers returned** — transformer produces a list of layers

Otherwise, set `_keras_layer_class` and let codegen handle it. See [codegen.md](codegen.md) for details.

### Checklist
- [ ] I have implemented a Spark Transformer that extends [BaseTransformer](../src/kamae/spark/transformers/base.py)
- [ ] If my transformer needs a fit method, I have implemented a Spark Estimator that extends [BaseEstimator](../src/kamae/spark/estimators/base.py)
- [ ] I have defined `_params` with `ParamSpec` entries for each custom parameter
- [ ] I have defined `_compatible_dtypes` as a list of PySpark `DataType` instances
- [ ] I have set `_keras_layer_class` to my Keras layer class (or written manual `get_keras_layer` if needed)
- [ ] I have used one (or more) of the input/output mixin classes from [base.py](../src/kamae/spark/params/base.py)
- [ ] I have implemented the `_transform` method
- [ ] I have unit tests of my implementation, including Spark/Keras parity tests
