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

from kamae.spark.estimators import StringIndexEstimator
from kamae.spark.pipeline import KamaeSparkPipeline, KamaeSparkPipelineModel
from kamae.spark.transformers import (
    ArrayCropTransformer,
    LogTransformer,
    OrdinalArrayEncodeTransformer,
)

if __name__ == "__main__":
    print("Starting test of Spark pipeline and integration with Tensorflow")

    spark = SparkSession.builder.getOrCreate()

    # Create a spark dataframe with some fake data in an array
    array_fake_data = spark.createDataFrame(
        [
            ([1, 2, 3], ["a", "b", "b"]),
            ([4, 5, 6], ["d", "d", "f"]),
            ([7, 8, 9], ["g", "-1", "-1"]),
            ([], []),
        ],
        ["col1", "col4"],
    )
    print("Created array fake data")
    array_fake_data.show()

    # Create a spark dataframe with some fake array data that we will transform.
    # Notice that "c" and "e" are not in the string indexer model, so they will be
    # indexed as 0.
    array_fake_data_to_transform = spark.createDataFrame(
        [
            ([1, 2, 3], ["a", "b", "c"]),
            ([4, 5, 6], ["d", "e", "f"]),
            ([7, 8, 9], ["g", "-1", "-1"]),
            ([], []),
        ],
        ["col1", "col4"],
    )
    print("Created array fake data to be transformed.")
    array_fake_data_to_transform.show()

    # Create two transformers. We will show we can transform arrays
    array_pad_string_transformer = ArrayCropTransformer(
        inputCol="col4",
        outputCol="padded_col4",
        layerName="pad_transform_string",
        padValue="-1",
        arrayLength=5,
    )
    session_encoder = OrdinalArrayEncodeTransformer(
        inputCol="padded_col4",
        outputCol="col4_encoded",
        layerName="ordinal_encoder",
        padValue="-1",
    )
    indexer = StringIndexEstimator(
        inputCol="padded_col4",
        outputCol="col4_indexed",
        layerName="indexer",
        stringOrderType="alphabeticalAsc",
    )
    array_pad_transformer = ArrayCropTransformer(
        inputCol="col1",
        outputCol="padded_col1",
        layerName="pad_transform",
        padValue=0,
        arrayLength=5,
    )
    log_transform = LogTransformer(
        inputCol="padded_col1",
        outputCol="log_col1",
        layerName="log_transform",
        inputDtype="float",
    )

    pipeline = KamaeSparkPipeline(
        stages=[
            array_pad_string_transformer,
            array_pad_transformer,
            session_encoder,
            log_transform,
            indexer,
        ]
    )

    pipeline.write().overwrite().save("./output/test_pipeline/")

    print("Loading pipeline from disk")
    loaded_pipeline = KamaeSparkPipeline.load("./output/test_pipeline/")

    print("Transforming data with loaded pipeline")
    fit_pipeline = loaded_pipeline.fit(array_fake_data)
    fit_pipeline.transform(array_fake_data).show(20, False)

    print("Writing fitted pipeline to disk")
    fit_pipeline.write().overwrite().save("./output/test_fitted_pipeline/")

    print("Loading fitted pipeline from disk")
    loaded_fitted_pipeline = KamaeSparkPipelineModel.load(
        "./output/test_fitted_pipeline/"
    )

    print("Transformed array fake data")
    loaded_fitted_pipeline.transform(array_fake_data_to_transform).show(20, False)

    tf_input_schema = [
        tf.TensorSpec(name="col4", dtype=tf.string, shape=(None, None, None)),
        tf.TensorSpec(name="col1", dtype=tf.int32, shape=(None, None, None)),
    ]
    keras_model = loaded_fitted_pipeline.build_keras_model(
        tf_input_schema=tf_input_schema
    )
    print(keras_model.summary())

    print("Start: Predicting with the model with reg_inputs")
    # Predict with the model with inputs
    reg_inputs = {
        "col1": tf.constant(
            [
                [
                    [1, 2, 3],
                ]
            ]
        ),
        "col4": tf.constant(
            [
                [
                    ["a", "b", "c"],
                ]
            ],
            dtype="string",
        ),
    }

    print("Predicting with the model with reg_inputs")
    print(keras_model.predict(reg_inputs))

    print("Start: Predicting with the model with pad_inputs")
    # Predict with the model with pad inputs
    pad_inputs = {
        "col1": tf.constant(
            [
                [
                    [-1, -1, -1],
                ]
            ]
        ),
        "col4": tf.constant(
            [
                [
                    ["-1", "-1", "-1"],
                ]
            ],
            dtype="string",
        ),
    }

    print("Predicting with the model with pad_inputs")
    print(keras_model.predict(pad_inputs))

    print("Start: Predicting with the model with empty inputs")
    # Predict with the model with empty inputs
    empty_inputs = {
        "col1": tf.constant(
            [
                [
                    [],
                ]
            ]
        ),
        "col4": tf.constant(
            [
                [
                    [],
                ]
            ],
            dtype="string",
        ),
    }

    print("Predicting with the model with empty_inputs")
    print(keras_model.predict(empty_inputs))

    # Saving model in pb format
    print("Saving model in pb format")
    keras_model.save("./output/test_saved_model/")
    print("Model saved in pb format")

    # Load model from SavedModel format
    print("Loading model from pb format")
    loaded_model = tf.keras.models.load_model("./output/test_saved_model/")
    print("Model loaded from pb format")

    # Predict with the loaded model
    print("Predicting with the loaded model")
    print(loaded_model.predict(reg_inputs))
    print(loaded_model.predict(pad_inputs))
    print(loaded_model.predict(empty_inputs))
