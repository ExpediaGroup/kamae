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

from kamae.spark.pipeline import KamaeSparkPipeline, KamaeSparkPipelineModel
from kamae.spark.transformers import (
    StringContainsTransformer,
    SubStringDelimAtIndexTransformer,
)

if __name__ == "__main__":
    print("Starting test of Spark pipeline and integration with Tensorflow")

    spark = SparkSession.builder.getOrCreate()

    # Create a spark dataframe some fake data
    fake_data = spark.createDataFrame(
        [
            (1, 2, 3, "EXPEDIA", "US"),
            (4, 5, 6, "EXPEDIA_UK", "UK"),
            (7, 8, 9, "EXPEDIA_UK_4EVA", "UK"),
            (7, 8, 9, None, None),
        ],
        ["col1", "col2", "col3", "col4", "col5"],
    )
    print("Created fake data")
    fake_data.show()

    # Setup transformers, can use set methods or just pass in the args to the constructor.

    string_contains_layer_constant = StringContainsTransformer(
        inputCol="col4",
        outputCol="col4_contains_UK",
        stringConstant="UK",
    )

    string_contains_layer_variable = StringContainsTransformer(
        inputCols=["col4", "col5"],
        outputCol="col4_contains_col5",
    )

    sub_str_delim = SubStringDelimAtIndexTransformer(
        inputCol="col4",
        outputCol="col4_sub_str_delim",
        delimiter="",
        index=0,
        defaultValue="NOT_FOUND",
    )

    print("Creating pipeline and writing to disk")
    test_pipeline = KamaeSparkPipeline(
        stages=[
            string_contains_layer_constant,
            string_contains_layer_variable,
            sub_str_delim,
        ]
    )

    print("Transforming data with loaded pipeline")
    fit_pipeline = test_pipeline.fit(fake_data)
    fit_pipeline.transform(fake_data).show(20, False)

    print("Saving pipeline to disk")
    fit_pipeline.write().overwrite().save("./output/test_pipeline/")
    loaded_fit_pipeline = KamaeSparkPipelineModel.load("./output/test_pipeline/")

    # Create input schema for keras model. A list of tf.TypeSpec objects.
    # Or a list of dicts.
    tf_input_schema = [
        {
            "name": "col4",
            "dtype": tf.string,
            "shape": (None, None, 1),
        },
        {
            "name": "col5",
            "dtype": tf.string,
            "shape": (None, None, 1),
        },
    ]
    keras_model = loaded_fit_pipeline.build_keras_model(tf_input_schema=tf_input_schema)
    print(keras_model.summary())
    keras_model.save("./output/test_keras_model/")

    print("Loading keras model from disk")
    loaded_keras_model = tf.keras.models.load_model("./output/test_keras_model/")
    inputs = [
        tf.constant(
            [
                [["EXPEDIA"], ["EXPEDIA_UK"], ["EXPEDIA_UK_4EVA"]],
                [["EXPEDIA"], ["EXPEDIA_UK"], ["EXPEDIA_UK_4EVA"]],
                [["EXPEDIA"], ["EXPEDIA_UK"], ["EXPEDIA_UK_4EVA"]],
            ]
        ),
        tf.constant(
            [
                [["US"], ["UK"], ["_4V"]],
                [["US"], ["UK"], ["_4V"]],
                [["US"], ["UK"], ["_4V"]],
            ]
        ),
    ]

    print("Predicting with loaded keras model")
    print(keras_model.predict(inputs))

    print(keras_model.outputs)
    print(keras_model.inputs)
