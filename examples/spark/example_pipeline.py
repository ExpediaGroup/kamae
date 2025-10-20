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

from kamae.spark.estimators import StandardScaleEstimator, StringIndexEstimator
from kamae.spark.pipeline import KamaeSparkPipeline, KamaeSparkPipelineModel
from kamae.spark.transformers import (
    ArrayConcatenateTransformer,
    ArraySplitTransformer,
    BinTransformer,
    IdentityTransformer,
    LogTransformer,
)

is_keras_3 = Version(keras.__version__) >= Version("3.0.0")

if __name__ == "__main__":
    print("Starting test of Spark pipeline and integration with Tensorflow")

    spark = SparkSession.builder.getOrCreate()

    # Create a spark dataframe some fake data
    fake_data = spark.createDataFrame(
        [
            (1, 2, 3, "a"),
            (4, 5, 6, "b"),
            (7, 8, 9, "c"),
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
    bin_transformer = (
        BinTransformer()
        .setInputCol("col3")
        .setOutputCol("bin_col3")
        .setConditionOperators(["lt", "gt", "eq"])
        .setBinValues([2, 6, 3])
        .setBinLabels(["less_than_2", "greater_than_6", "equal_to_3"])
        .setDefaultLabel("default")
        .setLayerName("bin_col3_output")
        .setInputDtype("float")
    )
    identity_transformer = (
        IdentityTransformer()
        .setInputCol("col3")
        .setOutputCol("identity_col3")
        .setLayerName("identity_col3_output")
    )
    string_indexer_freq = StringIndexEstimator(
        inputCol="col4",
        outputCol="col4_indexed_freq",
        layerName="string_indexer_freq",
        stringOrderType="frequencyDesc",
    )
    string_indexer_alpha = StringIndexEstimator(
        inputCol="col4",
        outputCol="col4_indexed_alpha",
        layerName="string_indexer_alpha",
        stringOrderType="alphabeticalDesc",
    )
    vector_assembler = ArrayConcatenateTransformer(
        inputCols=["log_col1", "col2", "col3"],
        outputCol="features",
        inputDtype="float",
    ).setLayerName("vec_assemble_log_col1_col2_col3")

    standard_scalar_layer = StandardScaleEstimator(
        inputCol="features",
        outputCol="scaled_features",
    ).setLayerName("numericals_scaled")

    vector_slicer = ArraySplitTransformer(
        inputCol="scaled_features",
        outputCols=["scaled_col1", "scaled_col2", "scaled_col3"],
    ).setLayerName("numericals_scaled_sliced")

    log_slice_vector = LogTransformer(
        inputCol="scaled_col3",
        outputCol="log_scaled_col3",
        alpha=1,
    ).setLayerName("log_transform_scaled_col3")

    print("Creating pipeline and writing to disk")
    test_pipeline = KamaeSparkPipeline(
        stages=[
            log_transformer,
            log_transformer2,
            vector_assembler,
            standard_scalar_layer,
            identity_transformer,
            string_indexer_alpha,
            string_indexer_freq,
            vector_slicer,
            log_slice_vector,
            bin_transformer,
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
    model_path = "./output/test_keras_model"
    if is_keras_3:
        model_path += ".keras"
    keras_model.save(model_path)

    print("Loading keras model from disk")
    loaded_keras_model = tf.keras.models.load_model(model_path)
    inputs = [
        tf.constant([[[1], [4], [7]]]),
        tf.constant([[[2], [5], [8]]]),
        tf.constant([[[3], [6], [9]]]),
        tf.constant([[["a"], ["b"], ["c"]]]),
    ]
    print("Predicting with loaded keras model")
    print(loaded_keras_model.predict(inputs))
