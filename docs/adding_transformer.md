# Contributing a Keras layer and Spark/Scikit-learn transformer

Follow this guide to contribute a new transformer to the project.

## Overview
In order to contribute a new transformer, you will need to implement a Spark Transformer, a corresponding Keras layer, and a Spark Estimator if your transformer needs a fit method.
We also require unit tests for all new classes, in particular parity tests ensuring your Spark Transformer and Keras layer produce the same output.

You may wish to also implement a Scikit-learn transformer, however we deem the scikit-learn usage pattern to be experimental for now and so this is not required.

## Naming
In order to avoid name clashes and to keep consistency, we have a naming convention for all new classes.

If an operation is called `<X>` then:

- `<X>Estimator` = Spark estimator (if applicable)
- `<X>Transformer` = Spark transformer
- `<X>Layer` = Tensorflow/Keras layer
- `<X>Params` = Spark params class

We just keep the verb stem. E.g string indexing is StringIndexTransformer, not StringIndexerTransformer.

The name of the file should then be `<X>.py`. E.g. `src/kame/spark/transformers/string_index.py` and `src/kame/tensorflow/layers/string_index.py`.

Finally, if you need to create an estimator, then the estimator and its corresponding transformer should be in different files. E.g. `src/kame/spark/transformers/string_index.py` and `src/kame/spark/estimators/string_index.py`.

## Keras layer
Your Keras layer should extend [BaseLayer](../src/kamae/tensorflow/layers/base.py) and implement the `_call` method. Furthermore, you will need to define the `compatible_dtypes` property which should return a list of compatible dtypes for the layer (or `None` if the layer is compatible with all dtypes).
You should ensure your layer is serializable by implementing the `get_config` method. 
You also need to add the decorator `@tf.keras.utils.register_keras_serializable(package=kamae.__name__)` to the class.

### Example

```python
from typing import List, Optional

import tensorflow as tf
import kamae

from .base import BaseLayer

@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class MyLayer(BaseLayer):
    def __init__(self, name, input_dtype, output_dtype, my_param, **kwargs):
        # Ensure that the name, input_dtype, and output_dtype are passed to the super constructor
        super().__init__(name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs)
        self.my_param = my_param
    
    @property
    def compatible_dtypes(self) -> Optional[List[tf.DType]]:
        return [tf.float32, tf.float64]

    def _call(self, inputs):
        # do something with inputs
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({'my_param': self.my_param})
        return config
```

### Checklist

- [ ] I have implemented a Keras layer that extends [BaseLayer](../src/kamae/tensorflow/layers/base.py)
- [ ] I have implemented the `_call` method of my Keras layer.
- [ ] I have defined the `compatible_dtypes` property of my Keras layer.
- [ ] I have added the decorator `@tf.keras.utils.register_keras_serializable(package=kamae.__name__)` to my Keras layer.
- [ ] I have ensured that my layer takes a `name`, `input_dtype`, and `output_dtype` as arguments to the constructor and that this is passed to the super constructor.
- [ ] My Keras layer is serializable. I have implemented the `get_config` method and added the decorator seen above to the class.
- [ ] I have unit tests of my implementation. 
- [ ] I have a specific test of layer serialisation added [here](../../tests/tensorflow/test_layer_serialisation.py).

## Spark Transformer/Estimator
Your Spark Transformer should extend [BaseTransformer](../src/kamae/spark/transformers/base.py). 
In this it should implement the `get_tf_layer` method, which should return an instance of your Keras layer.
If your transformer needs a fit method, you should also implement a Spark Estimator (which extends [BaseEstimator](../src/kamae/spark/estimators/base.py)) whose fit method returns an instance of your transformer.

Spark has a peculiar way of building constructors, in that the `__init__` calls a `setParams` method, which sets the parameters of the transformer.
See the example below for how this works. All estimators and transformers follow this boilerplate code.
The `setParams` method is implemented in the base transformer and estimator classes, so you do not need to implement it yourself.
However, you do need to call it from your `__init__` method, as shown below. You also need to ensure that all custom parameters have a setter method,
which is in the form `set<ParamName>`, as the `setParams` method will look for this method.

Your transformer should use one (or more) of the input/output mixin classes from [base.py](../src/kamae/spark/params/base.py)
- `SingleInputSingleOutputParams`
- `SingleInputMultiOutputParams`
- `MultiInputSingleOutputParams`
- `MultiInputMultiOutputParams`

Only use more than one if you want to support two usages of your transformer, e.g. `MyTransformer(inputCol="a", outputCol="b")` and `MyTransformer(inputCols=["a", "b"], outputCols=["c", "d"])`.
See for example the [SumTransformer](../src/kamae/spark/transformers/sum.py), which supports single input with a constant to add, or multiple inputs to sum.

These mixins provide the `inputCol(s)` and `outputCol(s)` parameters, which are used to specify the input and output columns of your transformer.

If your transformer requires more parameters that would need to be serialised to the Spark ML pipeline, you should add create a parameter class by extending the `Params` class [here](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.param.Params.html).

Lastly, we have provided utils for transformers & estimators to natively transform nested Spark array columns. 
You can use one of the following functions from [here](../src/kamae/spark/utils/transform_utils.py) according to your usecase if you need to add support for nested columns:

- `single_input_single_output_scalar_transform`
- `single_input_single_output_array_transform`
- `single_input_single_output_scalar_udf_transform`
- `single_input_single_output_array_udf_transform`
- `multi_input_single_output_scalar_transform`
- `multi_input_single_output_array_transform`


### Example
Note that the methods are named `_fit` and `_transform`. `fit` and `transform` wrap these internal methods and should not be overridden.

```python
from typing import List, Optional

from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType, StringType, BinaryType

from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.transformers import BaseTransformer
from kamae.spark.estimators import BaseEstimator


class MyCustomParams(Params):
    myParam = Param(
        Params._dummy(),
        "myParam",
        "Description of myParam",
        typeConverter=TypeConverters.toFloat,
    )

    # Setter method must be in the form set<ParamName> otherwise 
    # the setParams method will not find the set method. 
    def setMyParam(self, value: float) -> "MyCustomParams":
        return self._set(myParam=value)

    def getMyParam(self) -> float:
        return self.getOrDefault(self.myParam)


class MyEstimator(
    BaseEstimator,
    SingleInputSingleOutputParams,
    MyCustomParams
):

    @keyword_only
    def __init__(
            self,
            inputCol: Optional[str] = None,
            outputCol: Optional[str] = None,
            layerName: Optional[str] = None,
            inputDtype: Optional[str] = None,
            outputDtype: Optional[str] = None,
            myParam: Optional[float] = None,
    ) -> None:
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        return [StringType(), BinaryType()]

    def _fit(self, dataset: DataFrame) -> "MyTransformer":
        # Do some fitting...
        return MyTransformer(
            inputCol=self.getInputCol(),
            outputCol=self.getOutputCol(),
            layerName=self.getLayerName(),
            inputDtype=self.getInputDtype(),
            outputDtype=self.getOutputDtype(),
            myParam=self.getMyParam(),
        )


class MyTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    MyCustomParams
):

    @keyword_only
    def __init__(
            self,
            inputCol: Optional[str] = None,
            outputCol: Optional[str] = None,
            layerName: Optional[str] = None,
            inputDtype: Optional[str] = None,
            outputDtype: Optional[str] = None,
            myParam: Optional[float] = None,
    ) -> None:
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        return [StringType(), BinaryType()]

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        # Ensure that the layer has the layer name, input dtype, and output dtype
        # as arguments `name`, `input_dtype`, and `output_dtype` respectively.
        return MyLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            out_dtype=self.getOutputTFDtype(),
            my_param=self.getMyParam(),
        )

    def _transform(self, dataset: DataFrame) -> DataFrame:
        # Do some transformation...
        return dataset.withColumn(
            self.getOutputCol(),
            output_of_transform,
        )
```

### Checklist
- [ ] I have implemented a Spark Transformer that extends [BaseTransformer](../src/kamae/spark/transformers/base.py).
- [ ] If my transformer needs a fit method, I have implemented a Spark Estimator that extends [BaseEstimator](../src/kamae/spark/estimators/base.py).
- [ ] I have followed the instructions for the `__init__` and `setParams` methods.
- [ ] I have used one (or more) of the input/output mixin classes from [base.py](../src/kamae/spark/params/base.py).
- [ ] If my transformer requires more parameters that would need to be serialised to the Spark ML pipeline, I have added a parameter class by extending the `Params` class [here](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.param.Params.html).
- [ ] I have defined the `compatible_dtypes` property to specify the input/output data types that my transformer/estimator supports.
- [ ] I used a Keras subclassed layer for my `get_tf_layer` method.
- [ ] I have unit tests of my implementation. In particular, I have parity tests between the Spark and Keras implementations.

## Scikit-learn Transformer/Estimator

If your transformer is a wrapper around an existing Scikit-Learn transformer, you should
also extend the [BaseTransformerMixin](../src/kamae/sklearn/transformers/base.py) class. This will provide the required functionality
to be incorporated into the Kamae framework. 

If you are writing a custom transformer, you should extend the [BaseTransformer](../src/kamae/sklearn/transformers/base.py) class.
The only difference between these classes is that the `BaseTransformer` class also extends
the `BaseEstimator` and `TransformerMixin` classes from scikit-learn. If you are wrapping
an existing transformer, these are already extended by the transformer you are wrapping.
See the [StandardScaleEstimator](../src/kamae/sklearn/estimators/standard_scale.py) for an example of a wrapper around an existing transformer.
See the [IdentityTransformer](../src/kamae/sklearn/transformers/identity.py) for an example of a custom transformer.

Additionally, your transformer should use one (or more) of the input/output mixin classes from [base.py](../src/kamae/sklearn/params/base.py)
- SingleInputSingleOutputMixin
- SingleInputMultiOutputMixin
- MultiInputSingleOutputMixin
- MultiInputMultiOutputMixin

Only use more than one if you want to support two usages of your transformer.
We have no scikit-learn examples of this yet, only Spark. The behaviour is the same.
See above to the Spark section to understand why you may want to do this.

In scikit-learn, everything is an estimator. If your transformer does not require a fit method,
just return `self` from the `fit` method. If your transformer does require a fit method, you
should implement it within the `fit` method of your transformer.

### Example
```python
import pandas as pd
import tensorflow as tf
from kamae.sklearn.params import SingleInputSingleOutputMixin
from kamae.sklearn.transformers import BaseTransformer

class MyTransformer(
    BaseTransformer, SingleInputSingleOutputMixin
):
    def __init__(self, input_col: str, output_col: str, layer_name: str) -> None:
        super().__init__()
        self.input_col = input_col
        self.output_col = output_col
        self.layer_name = layer_name
        
    def fit(self, X: pd.DataFrame, y=None) -> "MyTransformer":
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X[self.output_col] = output_of_transform
        return X
    
    def get_tf_layer(self) -> tf.keras.layers.Layer:
        return MyLayer(
            name=self.layer_name,
        )
```

### Checklist
- [ ] I have implemented a Scikit-learn Transformer that extends [BaseTransformer](../src/kamae/sklearn/transformers/base.py) (if custom) or [BaseTransformerMixin](../src/kamae/sklearn/transformers/base.py) (if wrapping an existing transformer).
- [ ] If my transformer needs a fit method, I have implemented it within the `fit` method of my transformer.
- [ ] I have used one (or more) of the input/output mixin classes from [base.py](../src/kamae/sklearn/params/base.py).
- [ ] I used a Keras subclassed layer for my `get_tf_layer` method.
- [ ] I have unit tests of my implementation. In particular, I have parity tests between the scikit-learn and Keras implementations.
