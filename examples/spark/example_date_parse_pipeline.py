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
from kamae.spark.transformers import DateParseTransformer

is_keras_3 = Version(keras.__version__) >= Version("3.0.0")

if __name__ == "__main__":
    print("Starting test of Spark pipeline and integration with Tensorflow")

    spark = SparkSession.builder.getOrCreate()

    # Create a spark dataframe some fake data
    fake_data = spark.createDataFrame(
        [
            (1, 2, 3, "EXPEDIA", "2022-09-12", "2022-09-12 00:10:17"),
            (4, 5, 6, "EXPEDIA_UK", "2022-08-04", "2023-08-04 00:00:00"),
            (7, 8, 9, "EXPEDIA_UK_4EVA", "2023-09-12", "2023-09-12 00:00:00"),
            (7, 8, 9, None, None, None),
        ],
        ["col1", "col2", "col3", "col4", "col5", "col6"],
    )
    print("Created fake data")
    fake_data.show()

    # Setup transformers, can use set methods or just pass in the args to the constructor.

    date_parse_month = DateParseTransformer(
        inputCol="col5",
        outputCol="col4_month",
        datePart="MonthOfYear",
    )

    date_parse_day_of_week = DateParseTransformer(
        inputCol="col6",
        outputCol="col6_day_of_week",
        datePart="DayOfWeek",
    )

    date_parse_second = DateParseTransformer(
        inputCol="col6",
        outputCol="col6_second",
        datePart="Second",
    )

    date_parse_day_of_year = DateParseTransformer(
        inputCol="col6",
        outputCol="col6_day_of_year",
        datePart="DayOfYear",
    )

    date_parse_day_of_month = DateParseTransformer(
        inputCol="col6",
        outputCol="col6_day_of_month",
        datePart="DayOfMonth",
    )

    print("Creating pipeline and writing to disk")
    test_pipeline = KamaeSparkPipeline(
        stages=[
            date_parse_month,
            date_parse_day_of_week,
            date_parse_second,
            date_parse_day_of_month,
            date_parse_day_of_year,
        ]
    )

    print("Transforming data with loaded pipeline")
    fit_pipeline = test_pipeline.fit(fake_data)
    fit_pipeline.transform(fake_data).show(20, False)

    # Create input schema for keras model.
    tf_input_schema = [
        {
            "name": "col5",
            "dtype": tf.string,
            "shape": (None, 3, 1),
        },
        {
            "name": "col6",
            "dtype": tf.string,
            "shape": (None, 3, 1),
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
                [["2023-09-12"], ["2022-08-04"], ["2021-08-19"]],
                [["2023-09-12"], ["2022-08-04"], ["2021-08-19"]],
                [["2023-09-12"], ["2022-08-04"], ["2021-08-19"]],
            ]
        ),
        tf.constant(
            [
                [
                    ["2023-08-19 00:00:00"],
                    ["2021-07-05 23:10:20"],
                    ["2021-07-14 21:19:10"],
                ],
                [
                    ["2023-08-19 00:00:00"],
                    ["2021-07-05 23:10:20"],
                    ["2021-07-14 21:19:10"],
                ],
                [
                    ["2023-08-19 00:00:00"],
                    ["2021-07-05 23:10:20"],
                    ["2021-07-14 21:19:10"],
                ],
            ]
        ),
    ]
    print("Predicting with loaded keras model")
    print(keras_model.predict(inputs))

    print(keras_model.outputs)
    print(keras_model.inputs)
