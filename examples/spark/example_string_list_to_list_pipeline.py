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
    StringListToStringTransformer,
    StringToStringListTransformer,
)

is_keras_3 = Version(keras.__version__) >= Version("3.0.0")

if __name__ == "__main__":
    print("Starting test of Spark pipeline and integration with Tensorflow")

    spark = SparkSession.builder.getOrCreate()

    # Create a spark dataframe some fake data
    fake_data = spark.createDataFrame(
        [
            (["a", "b", "c"], ["d", "e", "f", "g", "h", "i"], "hello|world"),
            (["j", "k", "l"], ["m", "n", "o", "p", "q", "r"], "foo|bar"),
            (["s", "t", "u"], ["v", "w", "x", "y", "z", "aa"], "split|me|up"),
        ],
        ["col1", "col2", "col3"],
    )
    print("Created fake data")
    fake_data.show()

    # Setup transformers, can use set methods or just pass in the args to the constructor.
    string_list_to_string_1 = StringListToStringTransformer(
        inputCol="col1",
        outputCol="string_list_to_string_col1",
        layerName="string_list_to_string_col1_layer",
        separator=" ",
    )

    string_list_to_string_2 = StringListToStringTransformer(
        inputCol="col2",
        outputCol="string_list_to_string_col2",
        layerName="string_list_to_string_col2_layer",
        separator="__",
    )

    string_to_string_list = StringToStringListTransformer(
        inputCol="col3",
        outputCol="string_to_string_list_col3",
        layerName="string_to_string_list_col3_layer",
        separator="|",
        listLength=3,
        defaultValue="DEFAULT",
    )

    print("Creating pipeline and writing to disk")
    test_pipeline = KamaeSparkPipeline(
        stages=[
            string_list_to_string_1,
            string_list_to_string_2,
            string_to_string_list,
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
            "dtype": tf.string,
            "shape": (None, 3),
        },
        {
            "name": "col2",
            "dtype": tf.string,
            "shape": (None, 3),
        },
        {
            "name": "col3",
            "dtype": tf.string,
            "shape": (1,),
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
        tf.constant([[["a", "b", "c"], ["d", "e", "f"]]]),
        tf.constant([[["g", "h", "i"], ["j", "k", "l"]]]),
        tf.constant([["hello|world"]]),
    ]
    print("Predicting with loaded keras model")
    print(loaded_keras_model.predict(inputs))
