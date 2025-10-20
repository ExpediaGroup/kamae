# Copyright [2024] Expedia, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import keras
import tensorflow as tf
from packaging.version import Version
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType

from kamae.spark.pipeline import KamaeSparkPipeline, KamaeSparkPipelineModel
from kamae.spark.transformers import LambdaFunctionTransformer

is_keras_3 = Version(keras.__version__) >= Version("3.0.0")

if __name__ == "__main__":
    print("Starting test of Spark pipeline and integration with Tensorflow")

    spark = SparkSession.builder.getOrCreate()

    # Create a spark dataframe some fake data
    fake_data = spark.createDataFrame(
        [
            (1, 2, 3, "a"),
            (4, 5, 6, "b"),
            (7, 8, 9, "c"),
        ],
        ["col1", "col2", "col3", "col4"],
    )
    print("Created fake data")
    fake_data.show()

    def my_tf_fn(x):
        return tf.square(x) - tf.math.log(x)

    def my_multi_tf_fn(x):
        x0 = x[0]
        x1 = x[1]
        return tf.concat([tf.square(x0), tf.math.log(x1)], axis=-1)

    def my_single_input_multi_output_fn(x):
        return [tf.square(x), tf.math.log(x)]

    def my_multi_input_multi_output_fn(x):
        x0 = x[0]
        x1 = x[1]
        return [tf.square(x0), tf.concat([tf.square(x0), tf.math.log(x1)], axis=-1)]

    tf_single_input_single_output_lambda_fn_transformer = LambdaFunctionTransformer(
        inputCol="col2",
        outputCol="col2_single_in_out_tf_fn",
        function=my_tf_fn,
        functionReturnTypes=[FloatType()],
        inputDtype="float",
    )
    tf_multi_input_single_output_fn_transformer = LambdaFunctionTransformer(
        inputCols=["col2", "col3"],
        outputCol="col2_col3_multi_in_single_out_tf_fn",
        function=my_multi_tf_fn,
        functionReturnTypes=[ArrayType(FloatType())],
        inputDtype="float",
    )
    tf_single_input_multi_output_fn_transformer = LambdaFunctionTransformer(
        inputCol="col2",
        outputCols=[
            "col2_single_in_multi_out_tf_fn1",
            "col2_single_in_multi_out_tf_fn2",
        ],
        function=my_single_input_multi_output_fn,
        functionReturnTypes=[FloatType(), FloatType()],
        inputDtype="float",
    )
    tf_multi_input_multi_output_fn_transformer = LambdaFunctionTransformer(
        inputCols=["col2", "col3"],
        outputCols=["col2_multi_in_multi_out_tf_fn", "col3_multi_in_multi_out_tf_fn"],
        function=my_multi_input_multi_output_fn,
        functionReturnTypes=[FloatType(), ArrayType(FloatType())],
        inputDtype="float",
    )
    print("Creating pipeline and writing to disk")
    test_pipeline = KamaeSparkPipeline(
        stages=[
            tf_single_input_single_output_lambda_fn_transformer,
            tf_multi_input_single_output_fn_transformer,
            tf_single_input_multi_output_fn_transformer,
            tf_multi_input_multi_output_fn_transformer,
        ]
    )
    test_pipeline.write().overwrite().save("./output/test_pipeline/")

    print("Loading pipeline from disk")
    loaded_pipeline = KamaeSparkPipeline.load("./output/test_pipeline/")

    print("Transforming data with loaded pipeline")
    fit_pipeline = loaded_pipeline.fit(fake_data)
    fit_pipeline.transform(fake_data).show()

    print("Writing fitted pipeline to disk")
    fit_pipeline.write().overwrite().save("./output/test_fitted_pipeline/")

    print("Loading fitted pipeline from disk")
    loaded_fitted_pipeline = KamaeSparkPipelineModel.load(
        "./output/test_fitted_pipeline/"
    )

    # Create input schema for keras model.
    tf_input_schema = [
        {
            "name": "col2",
            "dtype": "int32",
            "shape": (None, 1),
        },
        {
            "name": "col3",
            "dtype": "float32",
            "shape": (None, 1),
        },
    ]
    keras_model = loaded_fitted_pipeline.build_keras_model(
        tf_input_schema=tf_input_schema
    )
    # print(keras_model.summary())
    model_path = "./output/test_keras_model"
    if is_keras_3:
        model_path += ".keras"
    keras_model.save(model_path)

    print("Loading keras model from disk")
    loaded_keras_model = tf.keras.models.load_model(model_path)
    inputs = [
        tf.constant([[[2], [5], [8]]]),
        tf.constant([[[3], [6], [9]]]),
    ]
    print("Predicting with loaded keras model")
    print(loaded_keras_model.predict(inputs))
