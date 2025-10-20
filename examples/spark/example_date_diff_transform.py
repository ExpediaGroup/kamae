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

from kamae.spark.pipeline import KamaeSparkPipeline
from kamae.spark.transformers import DateDiffTransformer

is_keras_3 = Version(keras.__version__) >= Version("3.0.0")

if __name__ == "__main__":
    print("Starting test of Spark pipeline and integration with Tensorflow")

    spark = SparkSession.builder.getOrCreate()

    # Create a spark dataframe some fake data
    fake_data = spark.createDataFrame(
        [
            ("2019-01-01", "2019-01-02", "2019-01-01 00:00:00", "2019-01-05 00:00:00"),
            ("2019-01-01", "2019-01-03", "2019-01-01 00:00:00", "2019-01-03 00:00:00"),
        ],
        ["col1", "col2", "col3", "col4"],
    )
    print("Created fake data")
    fake_data.show()

    # Setup transformers, can use set methods or just pass in the args to the constructor.

    date_diff_layer_1 = DateDiffTransformer(
        inputCols=["col1", "col2"],
        outputCol="col2_minus_col1",
    )

    date_diff_layer_2 = DateDiffTransformer(
        inputCols=["col3", "col4"],
        outputCol="col4_minus_col3",
    )

    print("Creating pipeline and writing to disk")
    test_pipeline = KamaeSparkPipeline(stages=[date_diff_layer_1, date_diff_layer_2])

    print("Transforming data with loaded pipeline")
    fit_pipeline = test_pipeline.fit(fake_data)
    fit_pipeline.transform(fake_data).show(20, False)

    # Create input schema for keras model.
    # Or a list of dicts.
    tf_input_schema = [
        {
            "name": "col1",
            "dtype": tf.string,
            "shape": (None, 1),
        },
        {
            "name": "col2",
            "dtype": tf.string,
            "shape": (None, 1),
        },
        {
            "name": "col3",
            "dtype": tf.string,
            "shape": (None, 1),
        },
        {
            "name": "col4",
            "dtype": tf.string,
            "shape": (None, 1),
        },
    ]
    keras_model = fit_pipeline.build_keras_model(tf_input_schema=tf_input_schema)
    print(keras_model.summary())
    model_path = "./output/test_saved_model"
    if is_keras_3:
        model_path += ".keras"
    keras_model.save(model_path)

    print("Loading keras model from disk")
    loaded_keras_model = tf.keras.models.load_model(model_path)
    inputs = [
        tf.constant(
            [
                [
                    ["2019-01-01"],
                    ["2019-01-03"],
                    ["2019-01-01 00:00:00"],
                    ["2019-01-05 00:00:00"],
                ]
            ]
        ),  # col1
        tf.constant(
            [
                [
                    ["2019-01-02"],
                    ["2019-01-06"],
                    ["2019-01-01 00:00:00"],
                    ["2019-01-03 00:00:00"],
                ]
            ]
        ),  # col2
        tf.constant(
            [
                [
                    ["2019-01-01"],
                    ["2019-01-03"],
                    ["2019-01-01 00:00:00"],
                    ["2019-01-05 00:00:00"],
                ]
            ]
        ),
        tf.constant(
            [
                [
                    ["2019-01-02"],
                    ["2019-01-06"],
                    ["2019-01-01 00:00:00"],
                    ["2019-01-03 00:00:00"],
                ]
            ]
        ),
    ]

    print("Predicting with loaded keras model")
    print(keras_model.predict(inputs))

    print(keras_model.outputs)
    print(keras_model.inputs)
