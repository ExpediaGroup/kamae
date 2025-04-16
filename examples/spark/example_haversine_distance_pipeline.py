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
from kamae.spark.transformers import HaversineDistanceTransformer

if __name__ == "__main__":
    print("Starting test of Spark pipeline and integration with Tensorflow")

    spark = SparkSession.builder.getOrCreate()

    # Create a spark dataframe some fake data
    fake_data = spark.createDataFrame(
        [
            (45.78, 23.09, 67.89, 12.34),
            (-45.90, -167.78, -0.12, 91.07),
            (-90.0, 180.0, 90.0, -180.0),
        ],
        ["lat1", "lon1", "lat2", "lon2"],
    )
    print("Created fake data")
    fake_data.show()

    # Setup transformers, can use set methods or just pass in the args to the constructor.

    hav_dist_layer_constant = HaversineDistanceTransformer(
        inputCols=["lat1", "lon1"],
        latLonConstant=[67.89, 12.34],
        outputCol="haversine_distance_constant",
    )

    hav_dist_layer_variable = HaversineDistanceTransformer(
        inputCols=["lat1", "lon1", "lat2", "lon2"],
        outputCol="haversine_distance_variable",
    )

    print("Creating pipeline and writing to disk")
    test_pipeline = KamaeSparkPipeline(
        stages=[hav_dist_layer_constant, hav_dist_layer_variable]
    )

    print("Transforming data with loaded pipeline")
    fit_pipeline = test_pipeline.fit(fake_data)
    fit_pipeline.transform(fake_data).show(20, False)

    # Create input schema for keras model. A list of tf.TypeSpec objects.
    # Or a list of dicts.
    tf_input_schema = [
        {
            "name": "lat1",
            "dtype": tf.float32,
            "shape": (None, 1),
        },
        {
            "name": "lon1",
            "dtype": tf.float32,
            "shape": (None, 1),
        },
        {
            "name": "lat2",
            "dtype": tf.float32,
            "shape": (None, 1),
        },
        {
            "name": "lon2",
            "dtype": tf.float32,
            "shape": (None, 1),
        },
    ]
    keras_model = fit_pipeline.build_keras_model(tf_input_schema=tf_input_schema)
    print(keras_model.summary())
    keras_model.save("./output/test_keras_model/")

    # print("Loading keras model from disk")
    # loaded_keras_model = tf.keras.models.load_model("./output/test_keras_model/")
    inputs = [
        tf.constant([[45.78], [23.09], [67.89], [12.34]]),
        tf.constant([[-45.90], [-167.78], [-0.12], [91.07]]),
        tf.constant([[-90.0], [180.0], [90.0], [-180.0]]),
        tf.constant([[180.0], [-180.0], [180.0], [-180.0]]),
    ]
    print("Predicting with loaded keras model")
    print(keras_model.predict(inputs))

    print(keras_model.outputs)
    print(keras_model.inputs)
