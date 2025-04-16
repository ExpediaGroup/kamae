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
from kamae.spark.transformers import StringReplaceTransformer
from kamae.tensorflow.layers import StringReplaceLayer

if __name__ == "__main__":
    print("Starting test of Spark pipeline and integration with Tensorflow")

    spark = SparkSession.builder.getOrCreate()

    # Create a spark dataframe some fake data
    fake_data = spark.createDataFrame(
        [
            (1, 2, 3, "EXPEDIA", "US", "JEFF"),
            (4, 5, 6, "EXPEDIA_UK", "UK", "STEPH"),
            (7, 8, 9, "EXPEDIA_UK_4EVA", "UK", "BETH"),
            (7, 8, 9, None, None, None),
        ],
        ["col1", "col2", "col3", "col4", "col5", "col6"],
    )
    print("Created fake data")
    fake_data.show()

    # Setup transformers, can use set methods or just pass in the args to the constructor.

    string_replace_layer_constant = StringReplaceTransformer(
        inputCol="col4",
        outputCol="col4_contains_UK_replace_JEFF",
        stringMatchConstant="UK",
        stringReplaceConstant="JEFF",
    )

    string_replace_layer_variable = StringReplaceTransformer(
        inputCols=["col4", "col5", "col6"],
        outputCol="col4_contains_col5_replace_col6",
    )

    print("Creating pipeline and writing to disk")
    test_pipeline = KamaeSparkPipeline(
        stages=[string_replace_layer_constant, string_replace_layer_variable]
    )

    print("Transforming data with loaded pipeline")
    fit_pipeline = test_pipeline.fit(fake_data)
    fit_pipeline.transform(fake_data).show(20, False)

    # Create input schema for keras model. A list of tf.TypeSpec objects.
    tf_input_schema = [
        tf.TensorSpec(name="col4", dtype=tf.string, shape=(None, None, 1)),
        tf.TensorSpec(name="col5", dtype=tf.string, shape=(None, None, 1)),
        tf.TensorSpec(name="col6", dtype=tf.string, shape=(None, None, 1)),
    ]
    keras_model = fit_pipeline.build_keras_model(tf_input_schema=tf_input_schema)
    print(keras_model.summary())
    keras_model.save("./output/test_keras_model/")

    # print("Loading keras model from disk")
    # loaded_keras_model = tf.keras.models.load_model("./output/test_keras_model/")
    inputs = [
        tf.constant(
            [[["EXPEDIA"], ["EXPEDIA.._UK"], ["EXPEDIA_.UK_4EVA.UK_4EV_WHEHEIW"]]]
        ),
        tf.constant([[["US"], ["._UK"], [".UK_4EV"]]]),
        tf.constant([[["JEFF"], ["STEPH"], ["BETH"]]]),
    ]
    print("Predicting with loaded keras model")
    print(keras_model.predict(inputs))

    print(keras_model.outputs)
    print(keras_model.inputs)
