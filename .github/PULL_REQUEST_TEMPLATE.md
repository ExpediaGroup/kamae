### Description
Provide a short description of the PR changes.


The below checklists come from the docs page on adding new transformers [here](../blob/main/docs/adding_transformer.md)
### Keras Layer Checklist
Verify that:
- [ ] The new Keras layer extends [BaseLayer](../blob/main/src/kamae/tensorflow/layers/base.py)
- [ ] The `_call` method has been implemented in the new layer.
- [ ] The `compatible_dtypes` property is defined in the new layer.
- [ ] The new layer is decorated with `@tf.keras.utils.register_keras_serializable(package=kamae.__name__)`.
- [ ] The new layer takes a `name`, `input_dtype`, and `output_dtype` as arguments to the constructor and that this is passed to the super constructor.
- [ ] The Keras layer is serializable. I have implemented the `get_config` method.
- [ ] There are unit tests of the new layer. 
- [ ] There is a specific test of layer serialisation added [here](../blob/main/tests/kamae/tensorflow/test_layer_serialisation.py).
- [ ] The new layer is imported in the [__init__.py](../blob/main/src/kamae/tensorflow/layers/__init__.py) file in the `layers` directory.

### Spark Transformer/Estimator Checklist
Verify that:
- [ ] The new Spark Transformer extends [BaseTransformer](../blob/main/src/kamae/spark/transformers/base.py).
- [ ] If the new transform needs a fit method, a Spark Estimator has been implemented that extends [BaseEstimator](../blob/main/src/kamae/spark/estimators/base.py).
- [ ] The instructions in the above docs page have been followed for the `__init__` and `setParams` methods.
- [ ] The transformer uses one of the input/output mixin classes from [base.py](../blob/main/src/kamae/spark/params/base.py).
- [ ] If the new transformer requires more parameters that would need to be serialised to the Spark ML pipeline, there is a implemented parameter class by extending the `Params` class [here](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.param.Params.html).
- [ ] The `compatible_dtypes` property has been implemented to specify the input/output data types that my transformer/estimator supports.
- [ ] A Keras subclassed layer is returned in the transformer's `get_tf_layer` method.
- [ ] There are unit tests of the new transform. In particular, there are parity tests between the Spark and Keras implementations.
- [ ] The new transformer/estimator is imported in the [__init__.py](../blob/main/src/kamae/spark/transformers/__init__.py) file in the `transformers`/`estimators` directory.

Finally, please verify that:
- [ ] There is a new entry (alphabetical order) in the README table describing the new layer/transformer
