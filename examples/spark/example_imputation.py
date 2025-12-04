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

from kamae.spark.estimators import ImputeEstimator
from kamae.spark.pipeline import KamaeSparkPipeline, KamaeSparkPipelineModel

is_keras_3 = Version(keras.__version__) >= Version("3.0.0")

if __name__ == "__main__":
    print("Starting test of Spark pipeline and integration with Tensorflow")

    spark = SparkSession.builder.getOrCreate()

    # Create a spark dataframe some fake data
    fake_data = spark.createDataFrame(
        [
            (1, 2, -999, "a"),
            (4, 5, 6, "b"),
            (7, 8, 9, "c"),
            (100, 100, 100, "z"),
            (100, 100, None, "x"),
        ],
        ["col1", "col2", "col3", "col4"],
    )
    print("Created fake data")
    fake_data.show()

    mean_impute_layer = (
        ImputeEstimator()
        .setInputCol("col3")
        .setOutputCol("mean_imputed_col3")
        .setMaskValue(-999)
        .setInputDtype("float")
        .setOutputDtype("float")
        .setLayerName("impute_mean_col3")
    )

    median_impute_layer = (
        ImputeEstimator()
        .setInputCol("col3")
        .setOutputCol("median_imputed_col3")
        .setMaskValue(-999)
        .setInputDtype("float")
        .setOutputDtype("float")
        .setImputeMethod("median")
        .setLayerName("impute_median_col3")
    )

    print("Creating pipeline and writing to disk")
    test_pipeline = KamaeSparkPipeline(
        stages=[
            mean_impute_layer,
            median_impute_layer,
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
    # Create input schema for keras model.
    tf_input_schema = [
        {
            "name": "col1",
            "dtype": "int32",
            "shape": (None, 1),
        },
        {
            "name": "col2",
            "dtype": "int32",
            "shape": (None, 1),
        },
        {
            "name": "col3",
            "dtype": "int32",
            "shape": (None, 1),
        },
        {
            "name": "col4",
            "dtype": "string",
            "shape": (None, 1),
        },
    ]
    keras_model = loaded_fitted_pipeline.build_keras_model(
        tf_input_schema=tf_input_schema
    )
    print(keras_model.summary())
    model_path = "./output/test_saved_model"
    if is_keras_3:
        model_path += ".keras"
    keras_model.save(model_path)

    print("Loading keras model from disk")
    loaded_keras_model = tf.keras.models.load_model(model_path)
    inputs = [
        tf.constant([[[1], [4], [7], [100]]]),
        tf.constant([[[2], [5], [8], [100]]]),
        tf.constant([[[-999], [6], [9], [100]]]),
        tf.constant([[["a"], ["b"], ["c"], ["z"]]]),
    ]
    print("Predicting with loaded keras model")
    print(loaded_keras_model.predict(inputs))
