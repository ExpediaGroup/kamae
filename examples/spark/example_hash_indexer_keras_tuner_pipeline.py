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
import keras_tuner as kt
import tensorflow as tf
from packaging.version import Version
from pyspark.sql import SparkSession

from kamae.spark.pipeline import KamaeSparkPipeline, KamaeSparkPipelineModel
from kamae.spark.transformers import HashIndexTransformer

is_keras_3 = Version(keras.__version__) >= Version("3.0.0")

if __name__ == "__main__":
    print(
        """
    Starting test of Spark pipeline, 
    integration with Tensorflow and Keras Tuner
    """
    )

    spark = SparkSession.builder.getOrCreate()

    # Create a spark dataframe some fake data
    fake_data = spark.createDataFrame(
        [
            ("hello", "world", "again"),
            ("this", "is", "a keras"),
            ("tuner", "example", "pipeline"),
        ],
        ["col1", "col2", "col3"],
    )
    print("Created fake data")
    fake_data.show()

    # Setup transformers, can use set methods or just pass in the args to the constructor.
    hash_indexer1 = (
        HashIndexTransformer()
        .setInputCol("col1")
        .setOutputCol("hash_col1")
        .setNumBins(100)
        .setLayerName("hash_indexer_col1")
    )
    hash_indexer2 = (
        HashIndexTransformer()
        .setInputCol("col2")
        .setOutputCol("hash_col2")
        .setNumBins(1000)
        .setLayerName("hash_indexer_col2")
    )
    hash_indexer3 = (
        HashIndexTransformer()
        .setInputCol("col3")
        .setOutputCol("hash_col3")
        .setNumBins(500)
        .setLayerName("hash_indexer_col3")
    )

    print("Creating pipeline and writing to disk")
    test_pipeline = KamaeSparkPipeline(
        stages=[
            hash_indexer1,
            hash_indexer2,
            hash_indexer3,
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

    print("Building keras tuner model builder function from fit pipeline")
    # Create input schema for keras model.
    tf_input_schema = [
        {
            "name": "col1",
            "dtype": "string",
            "shape": (1,),
        },
        {
            "name": "col2",
            "dtype": "string",
            "shape": (1,),
        },
        {
            "name": "col3",
            "dtype": "string",
            "shape": (1,),
        },
    ]

    # In order to use the keras tuner we need to define a dictionary of hyperparameters
    # The format is as follows:
    #  {
    #     "layer_name": [
    #         {
    #             "arg_name": <NAME_OF_LAYER_ARGUMENT>,
    #              "method": <NAME_OF_KERAS_HYPERPARAMETER_METHOD>, e.g. "choice"
    #              "kwargs": {
    #                         <KWARGS_TO_PASS_TO_KERAS_HYPERPARAMETER_METHOD>
    #                   }
    #       }
    #    ]
    # }

    hyper_param_dict = {
        "hash_indexer_col1": [
            {
                "arg_name": "num_bins",
                "method": "choice",
                "kwargs": {
                    "name": "hash_indexer_col1_num_bins",
                    "values": [100, 500, 1000],
                },
            }
        ],
        "hash_indexer_col2": [
            {
                "arg_name": "num_bins",
                "method": "int",
                "kwargs": {
                    "name": "hash_indexer_col2_num_bins",
                    "min_value": 100,
                    "max_value": 1000,
                },
            }
        ],
        "hash_indexer_col3": [
            {
                "arg_name": "num_bins",
                "method": "int",
                "kwargs": {
                    "name": "hash_indexer_col3_num_bins",
                    "min_value": 1000,
                    "max_value": 5000,
                },
            }
        ],
    }

    build_prepro_model = loaded_fitted_pipeline.get_keras_tuner_model_builder(
        tf_input_schema=tf_input_schema,
        hp_dict=hyper_param_dict,
    )

    # Next we setup the model builder function. Here we will use the function
    # we just got for the preprocessing hyperparameters and then add
    # embedding and dense layers. Note that we can get the hyperparameters
    # from the hashing layer to use as the input dimension for the embedding layer.

    def build_model(hp):
        prepro_model = build_prepro_model(hp)
        num_bins_params = [
            prepro_model.get_layer(layer).get_config()["num_bins"]
            for layer in ["hash_indexer_col1", "hash_indexer_col2", "hash_indexer_col3"]
        ]

        # Add embedding layers with vocab = num_bins + 1
        col1_embedding_layer = tf.keras.layers.Embedding(
            name="hash_col1_embedding",
            input_dim=num_bins_params[0] + 1,
            output_dim=hp.Int(
                "hash_col1_embedding_dim",
                min_value=32,
                max_value=num_bins_params[0] + 1,
            ),
        )(prepro_model.outputs[0])

        col2_embedding_layer = tf.keras.layers.Embedding(
            name="hash_col2_embedding",
            input_dim=num_bins_params[1] + 1,
            output_dim=hp.Int(
                "hash_col2_embedding_dim",
                min_value=32,
                max_value=num_bins_params[1] + 1,
            ),
        )(prepro_model.outputs[1])

        col3_embedding_layer = tf.keras.layers.Embedding(
            name="hash_col3_embedding",
            input_dim=num_bins_params[2] + 1,
            output_dim=hp.Int(
                "hash_col3_embedding_dim",
                min_value=32,
                max_value=num_bins_params[2] + 1,
            ),
        )(prepro_model.outputs[2])

        # Concatenate the embedding layers
        concat_embeds = tf.keras.layers.Concatenate(
            name="concat_embedding_layers",
        )([col1_embedding_layer, col2_embedding_layer, col3_embedding_layer])

        # Add dense layer with hyperparameter for number of units
        dense_layer = tf.keras.layers.Dense(
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation="relu",
            name="dense_layer",
        )(concat_embeds)
        output_layer = tf.keras.layers.Dense(
            units=1,
            activation="relu",
            name="output_layer",
        )(dense_layer)

        # We need to be careful not to end up with a disconnected graph when combining
        # the preprocessing model and the rest of the training model
        model = tf.keras.Model(
            inputs=prepro_model.inputs,
            outputs=output_layer,
        )

        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(
                hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
            ),
            loss="mse",
            metrics=["mse"],
        )
        return model

    print("Creating keras tuner object")
    tuner = kt.RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=5,
        project_name="output/test_keras_tuner_hash",
    )

    # Create some fake data for training and validation. This will be used in the keras
    # tuner to train and evaluate the model.
    x_train = [
        tf.constant(
            [
                ["Once"],
                ["again"],
                ["we"],
                ["create"],
                ["some"],
                ["fake"],
            ]
        ),
        tf.constant(
            [
                ["data"],
                ["for"],
                ["keras"],
                ["Once"],
                ["again"],
                ["we"],
            ]
        ),
        tf.constant(
            [
                ["create"],
                ["some"],
                ["fake"],
                ["data"],
                ["for"],
                ["keras"],
            ]
        ),
    ]

    y_train = tf.constant(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [52.0],
            [53.0],
        ]
    )

    # Thanks GH copilot for the below
    x_val = [
        tf.constant(
            [
                ["Once"],
                ["upon"],
                ["a"],
            ]
        ),
        tf.constant(
            [
                ["time"],
                ["there"],
                ["was"],
            ]
        ),
        tf.constant(
            [
                ["a"],
                ["king"],
                ["and"],
            ]
        ),
    ]

    y_val = tf.constant(
        [
            [1.0],
            [3.0],
            [53.0],
        ]
    )

    print("Running keras tuner search")
    tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

    print("Best model summary")
    best_model = tuner.get_best_models()[0]
    print(best_model.summary())

    print("Best hyperparameters")
    best_hp = tuner.get_best_hyperparameters()[0]
    print(best_hp.values)

    print("Saving best model")
    model_path = "./output/test_keras_tuner_hash_best_model"
    if is_keras_3:
        model_path += ".keras"
    best_model.save(model_path)

    print("Loading best model")
    loaded_best_model = tf.keras.models.load_model(model_path)

    print("Predict with best model")
    print(loaded_best_model.predict(x_val))

    print("Printing all layer params to check they are the same as the best model")
    for layer in loaded_best_model.layers:
        print("Layer name: ", layer.name)
        print("Layer params: ", layer.get_config())
