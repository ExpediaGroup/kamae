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

from kamae.spark.estimators import OneHotEncodeEstimator, StandardScaleEstimator
from kamae.spark.pipeline import KamaeSparkPipeline, KamaeSparkPipelineModel
from kamae.spark.transformers import (
    ArrayConcatenateTransformer,
    IdentityTransformer,
    LogTransformer,
)

if __name__ == "__main__":
    print("Starting test of Spark pipeline and integration with Tensorflow")

    spark = SparkSession.builder.getOrCreate()

    # Create a spark dataframe some fake data
    fake_data = spark.createDataFrame(
        [
            (1, 2, 3, "a"),
            (4, 5, 6, "b"),
            (7, 8, 9, "c"),
            (7, 8, 9, None),
        ],
        ["col1", "col2", "col3", "col4"],
    )
    print("Created fake data")
    fake_data.show()

    # Setup transformers, can use set methods or just pass in the args to the constructor.
    log_transformer = (
        LogTransformer()
        .setInputCol("col1")
        .setOutputCol("log_col1")
        .setAlpha(1)
        .setInputDtype("float")
        .setLayerName("log_one_plus_x")
    )
    log_transformer2 = (
        LogTransformer()
        .setInputCol("col2")
        .setOutputCol("log_col2")
        .setAlpha(1)
        .setInputDtype("float")
        .setLayerName("log2_one_plus_x")
    )
    identity_transformer = (
        IdentityTransformer()
        .setInputCol("col3")
        .setOutputCol("identity_col3")
        .setLayerName("identity_col3_output")
    )
    one_hot_encoder = OneHotEncodeEstimator(
        inputCol="col4",
        outputCol="col4_one_hot",
    ).setLayerName("one_hot_encoder")

    vector_assembler = ArrayConcatenateTransformer(
        inputCols=["log_col1", "log_col2", "identity_col3", "col4_one_hot"],
        outputCol="features",
        inputDtype="float",
    ).setLayerName("vec_assemble")

    standard_scalar_layer = StandardScaleEstimator(
        inputCol="features",
        outputCol="scaled_features",
    ).setLayerName("numericals_scaled")

    print("Creating pipeline and writing to disk")
    test_pipeline = KamaeSparkPipeline(
        stages=[
            log_transformer,
            log_transformer2,
            identity_transformer,
            one_hot_encoder,
            vector_assembler,
            standard_scalar_layer,
        ]
    )
    test_pipeline.write().overwrite().save("./output/test_pipeline/")

    print("Loading pipeline from disk")
    loaded_pipeline = KamaeSparkPipeline.load("./output/test_pipeline/")

    print("Transforming data with loaded pipeline")
    fit_pipeline = loaded_pipeline.fit(fake_data)
    fit_pipeline.transform(fake_data).select("scaled_features").show(20, False)

    print("Writing fitted pipeline to disk")
    fit_pipeline.write().overwrite().save("./output/test_fitted_pipeline/")

    print("Loading fitted pipeline from disk")
    loaded_fitted_pipeline = KamaeSparkPipelineModel.load(
        "./output/test_fitted_pipeline/"
    )

    print("Building keras model from fit pipeline")
    # Create input schema for keras model. A list of tf.TypeSpec objects.
    tf_input_schema = [
        tf.TensorSpec(name="col1", dtype=tf.int32, shape=(None, None, 1)),
        tf.TensorSpec(name="col2", dtype=tf.int32, shape=(None, None, 1)),
        tf.TensorSpec(name="col3", dtype=tf.int32, shape=(None, None, 1)),
        tf.TensorSpec(name="col4", dtype=tf.string, shape=(None, None, 1)),
    ]
    keras_model = loaded_fitted_pipeline.build_keras_model(
        tf_input_schema=tf_input_schema
    )
    print(keras_model.summary())
    keras_model.save("./output/test_keras_model/")

    print("Loading keras model from disk")
    loaded_keras_model = tf.keras.models.load_model("./output/test_keras_model/")
    inputs = [
        tf.constant([[[1], [4], [7], [7]]]),
        tf.constant([[[2], [5], [8], [8]]]),
        tf.constant([[[3], [6], [9], [9]]]),
        tf.constant([[["a"], ["b"], ["c"], ["d"]]]),
    ]
    print("Predicting with loaded keras model")
    print(loaded_keras_model.predict(inputs))
