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

import tensorflow as tf
from pyspark.sql import SparkSession

from kamae.spark.pipeline import KamaeSparkPipeline, KamaeSparkPipelineModel
from kamae.spark.transformers import (
    LogicalAndTransformer,
    LogicalNotTransformer,
    LogicalOrTransformer,
)

if __name__ == "__main__":
    print("Starting test of Spark pipeline and integration with Tensorflow")

    spark = SparkSession.builder.getOrCreate()

    # Create a spark dataframe some fake data
    fake_data = spark.createDataFrame(
        [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ],
        ["col1", "col2"],
    )
    print("Created fake data")
    fake_data.show()

    # Setup transformers, can use set methods or just pass in the args to the constructor.
    logical_and_transformer = LogicalAndTransformer(
        inputCols=["col1", "col2"], outputCol="logical_and_col1_col2"
    ).setLayerName("logical_and")

    logical_or_transformer = LogicalOrTransformer(
        inputCols=["col1", "col2"], outputCol="logical_or_col1_col2"
    ).setLayerName("logical_or")

    logical_not_transformer = LogicalNotTransformer(
        inputCol="col1", outputCol="logical_not_col1"
    ).setLayerName("logical_not")

    logical_nor_transformer = LogicalNotTransformer(
        inputCol="logical_or_col1_col2", outputCol="logical_nor_col1_col2"
    ).setLayerName("logical_nor")

    logical_nand_transformer = LogicalNotTransformer(
        inputCol="logical_and_col1_col2", outputCol="logical_nand_col1_col2"
    ).setLayerName("logical_nand")

    logical_xor_transformer = LogicalAndTransformer(
        inputCols=["logical_or_col1_col2", "logical_nand_col1_col2"],
        outputCol="logical_xor_col1_col2",
    ).setLayerName("logical_xor")

    print("Creating pipeline and writing to disk")
    test_pipeline = KamaeSparkPipeline(
        stages=[
            logical_and_transformer,
            logical_or_transformer,
            logical_not_transformer,
            logical_nor_transformer,
            logical_nand_transformer,
            logical_xor_transformer,
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

    print("Building keras model from fit pipeline")
    # Create input schema for keras model. A list of tf.TypeSpec objects.
    tf_input_schema = [
        tf.TensorSpec(name="col1", shape=(None, 1), dtype=tf.bool),
        tf.TensorSpec(name="col2", shape=(None, 1), dtype=tf.bool),
    ]
    keras_model = loaded_fitted_pipeline.build_keras_model(
        tf_input_schema=tf_input_schema
    )
    print(keras_model.summary())
    keras_model.save("./output/test_keras_model/")

    print("Loading keras model from disk")
    loaded_keras_model = tf.keras.models.load_model("./output/test_keras_model/")
    inputs = [
        tf.constant([[True], [True], [False], [False]]),
        tf.constant([[True], [False], [True], [False]]),
    ]
    print("Predicting with loaded keras model")
    print(loaded_keras_model.predict(inputs))
