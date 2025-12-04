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

from kamae.spark.pipeline import KamaeSparkPipeline, KamaeSparkPipelineModel
from kamae.spark.transformers import CosineSimilarityTransformer

is_keras_3 = Version(keras.__version__) >= Version("3.0.0")

if __name__ == "__main__":
    print("Starting test of Spark pipeline and integration with Tensorflow")

    spark = SparkSession.builder.getOrCreate()

    # Create a spark dataframe some fake data
    fake_data = spark.createDataFrame(
        [
            ([67.9, 45.80, -1.45, 3.45], [-1.234, 5.67, -3.45, 1.067]),
            ([5.9, -4.80, 145.0, 3.45], [-5.678, 1.245, 7.890, -1.456]),
            ([7.9, 45.80, -1.45, 3.45], [-1.234, 5.67, -3.45, 1.067]),
        ],
        ["col1", "col2"],
    )
    print("Created fake data")
    fake_data.show()

    # Setup transformers, can use set methods or just pass in the args to the constructor.

    cosine_sim = CosineSimilarityTransformer(
        inputCols=["col1", "col2"],
        outputCol="col1_col2_cosine_sim",
    ).setLayerName("cosine_sim")

    print("Creating pipeline and writing to disk")
    test_pipeline = KamaeSparkPipeline(
        stages=[
            cosine_sim,
        ]
    )
    test_pipeline.write().overwrite().save("./output/test_pipeline/")

    print("Loading pipeline from disk")
    loaded_pipeline = KamaeSparkPipeline.load("./output/test_pipeline/")

    print("Transforming data with loaded pipeline")
    fit_pipeline = loaded_pipeline.fit(fake_data)
    fit_pipeline.transform(fake_data).show(20, False)

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
            "dtype": "float32",
            "shape": (None, 4),
        },
        {
            "name": "col2",
            "dtype": "float32",
            "shape": (None, 4),
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
        tf.constant(
            [
                [
                    [67.9, 45.80, -1.45, 3.45],
                    [5.9, -4.80, 145, 3.45],
                    [7.9, 45.80, -1.45, 3.45],
                ]
            ]
        ),
        tf.constant(
            [
                [
                    [-1.234, 5.67, -3.45, 1.067],
                    [-5.678, 1.245, 7.890, -1.456],
                    [-1.234, 5.67, -3.45, 1.067],
                ]
            ]
        ),
    ]
    print("Predicting with loaded keras model")
    print(loaded_keras_model.predict(inputs))
