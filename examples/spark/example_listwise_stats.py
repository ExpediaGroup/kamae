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

from kamae.spark.pipeline import KamaeSparkPipeline
from kamae.spark.transformers import (
    ListMaxTransformer,
    ListMeanTransformer,
    ListMinTransformer,
    SubtractTransformer,
)

if __name__ == "__main__":
    print("Starting test of Spark pipeline and integration with Tensorflow")

    spark = SparkSession.builder.getOrCreate()

    # Create a spark dataframe some fake data
    fake_data = spark.createDataFrame(
        [
            (
                1.0,
                2.0,
                3.0,
                "EXPEDIA",
            ),
            (
                1.0,
                5.0,
                6.0,
                "EXPEDIA_UK",
            ),
            (
                1.0,
                8.0,
                9.0,
                "HCOM",
            ),
            (
                1.0,
                8.0,
                9.0,
                "EXPEDIA_UK",
            ),
            (
                1.0,
                9.0,
                9.0,
                "HCOM",
            ),
            (
                1.0,
                1.0,
                3.0,
                "EXPEDIA",
            ),
        ],
        [
            "query_id",
            "col2",
            "col3",
            "col4",
        ],
    )
    print("Created fake data")
    fake_data.show()

    # Setup transformers, can use set methods or just pass in the args to the constructor.

    seg_min = ListMinTransformer(
        layerName="col2_min_by_col4",
        inputCols=["col2", "col4"],
        outputCol="seg_min",
        queryIdCol="query_id",
        withSegment=True,
    )

    overall_min = ListMinTransformer(
        layerName="col2_min",
        inputCol="col2",
        outputCol="overall_min",
        queryIdCol="query_id",
    )

    seg_max = ListMaxTransformer(
        layerName="col2_max_by_col3",
        inputCols=["col2", "col3"],
        outputCol="seg_max",
        queryIdCol="query_id",
        withSegment=True,
    )

    seg_mean = ListMeanTransformer(
        layerName="col2_mean_by_col4",
        inputCols=["col2", "col4"],
        outputCol="seg_mean",
        queryIdCol="query_id",
        withSegment=True,
    )

    sorted_overall_mean = ListMeanTransformer(
        layerName="col2_sorted_mean",
        inputCols=["col2", "col3"],
        outputCol="overall_sorted_mean",
        queryIdCol="query_id",
        # withSegment=False,
        topN=3,  # mean of col2 for top 3 values of col3
    )

    print("Creating pipeline and writing to disk")
    test_pipeline = KamaeSparkPipeline(
        stages=[seg_min, seg_mean, seg_max, overall_min, sorted_overall_mean]
    )

    print("Transforming data with loaded pipeline")
    fit_pipeline = test_pipeline.fit(fake_data)
    fit_pipeline.transform(fake_data).show(20, False)

    # Create input schema for keras model. A list of tf.TypeSpec objects.
    tf_input_schema = [
        tf.TensorSpec(name="col2", dtype=tf.float32, shape=(None, None, 1)),
        tf.TensorSpec(name="col3", dtype=tf.float32, shape=(None, None, 1)),
        tf.TensorSpec(name="col4", dtype=tf.string, shape=(None, None, 1)),
    ]
    keras_model = fit_pipeline.build_keras_model(tf_input_schema=tf_input_schema)
    print(keras_model.summary())
    keras_model.save("./output/test_keras_model/")

    print("Loading keras model from disk")
    loaded_keras_model = tf.keras.models.load_model("./output/test_keras_model/")
    inputs = {
        "col2": tf.constant(
            [
                [[2.0], [5.0], [8.0], [8.0], [9.0], [1.0]],
            ]
        ),
        "col3": tf.constant(
            [
                [[3.0], [6.0], [9.0], [9.0], [9.0], [3.0]],
            ]
        ),
        "col4": tf.constant(
            [
                [
                    ["EXPEDIA"],
                    ["EXPEDIA_UK"],
                    ["HCOM"],
                    ["EXPEDIA_UK"],
                    ["HCOM"],
                    ["EXPEDIA"],
                ],
            ]
        ),
    }
    print("Predicting with loaded keras model")
    print(keras_model.predict(inputs))
