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
    ListMinTransformer,
    SegmentMaxTransformer,
    SegmentMeanTransformer,
    SegmentMinTransformer,
    SubtractTransformer,
)

if __name__ == "__main__":
    print("Starting test of Spark pipeline and integration with Tensorflow")

    spark = SparkSession.builder.getOrCreate()

    # Create a spark dataframe some fake data
    fake_data = spark.createDataFrame(
        [
            (
                1,
                2,
                3,
                "EXPEDIA",
            ),
            (
                1,
                5,
                6,
                "EXPEDIA_UK",
            ),
            (
                1,
                8,
                9,
                "HCOM",
            ),
            (
                1,
                8,
                9,
                "EXPEDIA_UK",
            ),
            (
                1,
                9,
                9,
                "HCOM",
            ),
            (
                1,
                1,
                3,
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

    overall_min = ListMinTransformer(
        layerName="col2_min",
        inputCol="col2",
        outputCol="overall_min",
        queryIdCol="query_id",
    )

    segment_max = SegmentMaxTransformer(
        layerName="col2_max_segmented_by_col4",
        inputCols=["col2", "col4"],
        outputCol="segment_max",
        queryIdCol="query_id",
    )

    segment_min = SegmentMinTransformer(
        layerName="col2_min_segmented_by_col4",
        inputCols=["col2", "col4"],
        outputCol="segment_min",
        queryIdCol="query_id",
    )

    segment_mean = SegmentMeanTransformer(
        layerName="col2_mean_segmented_by_col3",
        inputCols=["col2", "col3"],
        outputCol="segment_mean",
        queryIdCol="query_id",
    )

    subtraction = SubtractTransformer(
        layerName="col2_range_layer",
        inputCols=[
            "segment_max",
            "segment_min",
        ],
        outputCol="col2_range",
    )

    print("Creating pipeline and writing to disk")
    test_pipeline = KamaeSparkPipeline(
        stages=[segment_max, segment_min, overall_min, segment_mean, subtraction]
    )

    print("Transforming data with loaded pipeline")
    fit_pipeline = test_pipeline.fit(fake_data)
    fit_pipeline.transform(fake_data).show(20, False)

    # Create input schema for keras model. A list of tf.TypeSpec objects.
    tf_input_schema = [
        tf.TensorSpec(name="col2", dtype=tf.int32, shape=(None, None, 1)),
        tf.TensorSpec(name="col3", dtype=tf.int32, shape=(None, None, 1)),
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
                [[2], [5], [8], [8], [9], [1]],
            ]
        ),
        "col3": tf.constant(
            [
                [[3], [6], [9], [9], [9], [3]],
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
