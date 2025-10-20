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

import keras
import tensorflow as tf
from packaging.version import Version
from pyspark.sql import SparkSession

from kamae.spark.pipeline import KamaeSparkPipeline, KamaeSparkPipelineModel
from kamae.spark.transformers import (
    ModuloTransformer,
    RoundToDecimalTransformer,
    RoundTransformer,
)

is_keras_3 = Version(keras.__version__) >= Version("3.0.0")

if __name__ == "__main__":
    print("Starting test of Spark pipeline and integration with Tensorflow")

    spark = SparkSession.builder.getOrCreate()

    # Create a spark dataframe some fake data
    fake_data = spark.createDataFrame(
        [
            (1.4567, 2.23424, 3.687979),
            (4.2343, 5.46456, 6.9990),
            (7.1234435, 8.45657567, 9.3454545),
        ],
        ["col1", "col2", "col3"],
    )
    print("Created fake data")
    fake_data.show()

    # Setup transformers, can use set methods or just pass in the args to the constructor.
    round_transformer = (
        RoundTransformer()
        .setInputCol("col1")
        .setOutputCol("round_col1")
        .setRoundType("round")
        .setLayerName("round_col1_output")
    )
    round_decimals_transformer = (
        RoundToDecimalTransformer()
        .setInputCol("col1")
        .setOutputCol("round_decimals_col1")
        .setDecimals(3)
        .setLayerName("round_decimals_col1_output")
    )
    ceil_transformer = (
        RoundTransformer()
        .setInputCol("col2")
        .setOutputCol("ceil_col2")
        .setRoundType("ceil")
        .setLayerName("ceil_col2_output")
    )
    floor_transformer = (
        RoundTransformer()
        .setInputCol("col3")
        .setOutputCol("floor_col3")
        .setRoundType("floor")
        .setLayerName("floor_col3_output")
    )
    modulo_int_transformer = (
        ModuloTransformer()
        .setInputCols(["ceil_col2", "round_col1"])
        .setOutputCol("modulo_col2_col1")
        .setLayerName("modulo_col2_col1_output")
    )
    modulo_float_transformer = (
        ModuloTransformer()
        .setInputCols(["col3", "col1"])
        .setOutputCol("modulo_col3_col1")
        .setLayerName("modulo_col3_col1_output")
    )

    print("Creating pipeline and writing to disk")
    test_pipeline = KamaeSparkPipeline(
        stages=[
            round_transformer,
            ceil_transformer,
            floor_transformer,
            modulo_int_transformer,
            modulo_float_transformer,
        ]
    )
    test_pipeline.write().overwrite().save("./output/test_pipeline/")

    print("Loading pipeline from disk")
    loaded_pipeline = KamaeSparkPipeline.load("./output/test_pipeline/")

    print("Transforming data with loaded pipeline")
    fit_pipeline = loaded_pipeline.fit(fake_data)
    fit_pipeline.transform(fake_data).show(20, False)

    print("Writing fitted pipeline to disk")
    fit_pipeline.write().overwrite().save("./output/test_fitted_pipeline/")

    print("Loading fitted pipeline from disk")
    loaded_fitted_pipeline = KamaeSparkPipelineModel.load(
        "./output/test_fitted_pipeline/"
    )

    print("Building keras model from fit pipeline")
    # Create input schema for keras model.
    tf_input_schema = [
        {
            "name": "col1",
            "dtype": tf.float32,
            "shape": (None, 1),
        },
        {
            "name": "col2",
            "dtype": tf.float32,
            "shape": (None, 1),
        },
        {
            "name": "col3",
            "dtype": tf.float32,
            "shape": (None, 1),
        },
    ]
    keras_model = loaded_fitted_pipeline.build_keras_model(
        tf_input_schema=tf_input_schema
    )
    print(keras_model.summary())
    model_path = "./output/test_keras_model"
    if is_keras_3:
        model_path += ".keras"
    keras_model.save(model_path)

    print("Loading keras model from disk")
    loaded_keras_model = tf.keras.models.load_model(model_path)
    inputs = [
        tf.constant([[[1.4567], [4.2343], [7.1234435]]]),
        tf.constant([[[2.23424], [5.46456], [8.45657567]]]),
        tf.constant([[[3.687979], [6.9990], [9.3454545]]]),
    ]
    print("Predicting with loaded keras model")
    print(loaded_keras_model.predict(inputs))
    print(loaded_keras_model.output_names)
