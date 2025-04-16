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
    NumericalIfStatementTransformer,
    StringEqualsIfStatementTransformer,
)

if __name__ == "__main__":
    print("Starting test of Spark pipeline and integration with Tensorflow")

    spark = SparkSession.builder.getOrCreate()

    # Create a spark dataframe some fake data
    fake_data = spark.createDataFrame(
        [
            (1, 2, 3, "a"),
            (40, 5, 6, "b"),
            (7, 8, 9, "c"),
        ],
        ["col1", "col2", "col3", "col4"],
    )
    print("Created fake data")
    fake_data.show()

    # Setup transformers, can use set methods or just pass in the args to the constructor.
    numeric_if_statement = (
        NumericalIfStatementTransformer()
        .setConditionOperator("geq")
        .setResultIfTrue(1.0)
        .setInputCols(["col1", "col2", "col3"])
        .setOutputCol("col1_col2_col3_if_statement")
        .setInputDtype("float")
    )
    string_if_statement = (
        StringEqualsIfStatementTransformer()
        .setResultIfTrue("TRUE")
        .setInputCol("col4")
        .setOutputCol("col4_if_statement")
        .setResultIfFalse("FALSE")
        .setValueToCompare("a")
    )

    print("Creating pipeline and writing to disk")
    test_pipeline = KamaeSparkPipeline(
        stages=[numeric_if_statement, string_if_statement]
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
        tf.TensorSpec(name="col1", shape=(None, None, 1), dtype=tf.int32),
        tf.TensorSpec(name="col2", shape=(None, None, 1), dtype=tf.int32),
        tf.TensorSpec(name="col3", shape=(None, None, 1), dtype=tf.int32),
        tf.TensorSpec(name="col4", shape=(None, None, 1), dtype=tf.string),
    ]
    keras_model = loaded_fitted_pipeline.build_keras_model(
        tf_input_schema=tf_input_schema
    )
    print(keras_model.summary())
    keras_model.save("./output/test_keras_model/")

    print("Loading keras model from disk")
    loaded_keras_model = tf.keras.models.load_model("./output/test_keras_model/")
    inputs = [
        tf.constant([[[1], [4], [7]]]),
        tf.constant([[[2], [5], [8]]]),
        tf.constant([[[3], [6], [9]]]),
        tf.constant([[["a"], ["b"], ["c"]]]),
    ]
    print("Predicting with loaded keras model")
    print(loaded_keras_model.predict(inputs))
