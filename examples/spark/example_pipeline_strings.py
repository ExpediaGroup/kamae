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
from kamae.spark.transformers import (
    StringAffixTransformer,
    StringArrayConstantTransformer,
    StringConcatenateTransformer,
    StringListToStringTransformer,
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
    string_array_constant_transformer = StringArrayConstantTransformer(
        inputCol="col4",
        outputCol="string_array_constant",
        constantStringArray=["hello", "world"],
    ).setLayerName("string_array_constant_transformer")
    string_list_to_string_transformer = StringListToStringTransformer(
        inputCol="string_array_constant",
        outputCol="constant_string",
    ).setLayerName("string_list_to_string_transformer")
    string_concat_transformer = StringConcatenateTransformer(
        inputCols=["col4", "constant_string"],
        outputCol="col4_constant_join",
        separator="--",
    ).setLayerName("string_concat_transformer")
    string_affix_transformer = StringAffixTransformer(
        inputCol="col4_constant_join",
        outputCol="col4_affixed",
        prefix="<<",
        suffix=">>",
    ).setLayerName("string_affix_transformer")

    print("Creating pipeline and writing to disk")
    test_pipeline = KamaeSparkPipeline(
        stages=[
            string_array_constant_transformer,
            string_list_to_string_transformer,
            string_concat_transformer,
            string_affix_transformer,
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
            "name": "col4",
            "dtype": tf.string,
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
        tf.constant([[["a"], ["b"], ["c"]]]),
    ]
    print("Predicting with loaded keras model")
    print(loaded_keras_model.predict(inputs))
