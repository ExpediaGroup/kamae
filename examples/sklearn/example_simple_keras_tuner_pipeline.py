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

import joblib
import keras_tuner as kt
import pandas as pd
import tensorflow as tf

from kamae.sklearn.estimators import StandardScaleEstimator
from kamae.sklearn.pipeline import KamaeSklearnPipeline
from kamae.sklearn.transformers import ArrayConcatenateTransformer, LogTransformer

if __name__ == "__main__":
    print(
        """Starting test of Spark pipeline, 
    integration with Tensorflow and Keras Tuner"""
    )

    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

    # Create some dummy pandas data
    df = pd.DataFrame(
        {
            "col1": [10, 4.8, 7.3],
            "col2": [2.5, 5.3, 8.2],
            "col3": [3.7, 6.4, 9.4],
            "col4": ["a", "b", "c"],
        },
    )

    print("Original dataframe:")
    print(df.head())

    # Setup transformers, can use set methods or just pass in the args to the constructor.
    log_transformer = LogTransformer(
        input_col="col1",
        output_col="log_col1",
        alpha=1,
        layer_name="log_col1_one_plus_x",
    )
    log_transformer2 = LogTransformer(
        input_col="col2",
        output_col="log_col2",
        alpha=1,
        layer_name="log_col2_one_plus_x",
    )
    vector_assembler = ArrayConcatenateTransformer(
        input_cols=["log_col1", "log_col2", "col3"],
        output_col="features",
        layer_name="vec_assemble_log_col1_col2_col3",
    )

    standard_scalar_layer = StandardScaleEstimator(
        input_col="features",
        output_col="scaled_features",
        layer_name="standard_scaler",
    )

    print("Creating pipeline and writing to disk")
    test_pipeline = KamaeSklearnPipeline(
        steps=[
            ("log_transformer_1", log_transformer),
            ("log_transformer_2", log_transformer2),
            ("vector_assembler", vector_assembler),
            ("standard_scaler", standard_scalar_layer),
        ]
    )

    joblib.dump(test_pipeline, "./output/test_pipeline.joblib")

    print("Loading pipeline from disk")
    loaded_pipeline = joblib.load("./output/test_pipeline.joblib")

    print("Transforming data with loaded pipeline")
    fit_pipeline = loaded_pipeline.fit(df)
    print(fit_pipeline.transform(df).head())

    print("Building keras tuner model builder function from fit pipeline")
    # Create input schema for keras model
    # The values here will be inserted into tf.keras.Input layers
    # using kwargs ** syntax.
    tf_input_schema = [
        {
            "name": "col1",
            "dtype": "float32",
            "shape": (1,),
        },
        {
            "name": "col2",
            "dtype": "float32",
            "shape": (1,),
        },
        {
            "name": "col3",
            "dtype": "float32",
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
    #                            <KWARGS_TO_PASS_TO_KERAS_HYPERPARAMETER_METHOD>
    #                   }
    #       }
    #    ]
    # }

    hyper_param_dict = {
        "log_col1_one_plus_x": [
            {
                "arg_name": "alpha",
                "method": "choice",
                "kwargs": {
                    "name": "log_one_plus_x_alpha",
                    "values": [1, 10, 20],
                },
            }
        ],
        "log_col2_one_plus_x": [
            {
                "arg_name": "alpha",
                "method": "float",
                "kwargs": {
                    "name": "log2_one_plus_x_alpha",
                    "min_value": 1.0,
                    "max_value": 20.0,
                },
            }
        ],
    }

    build_prepro_model = fit_pipeline.get_keras_tuner_model_builder(
        tf_input_schema=tf_input_schema,
        hp_dict=hyper_param_dict,
    )

    # Next we setup the model builder function. Here we will use the function
    # we just got for the preprocessing hyperparameters and then add a dense layer
    # with a hyperparameter for the number of units.

    def build_model(hp):
        prepro_model = build_prepro_model(hp)
        prepro_output_layer = prepro_model.outputs[0]
        # Add dense layer with hyperparameter on top of prepro model output.
        dense_layer = tf.keras.layers.Dense(
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation="relu",
            name="dense_layer",
        )(prepro_output_layer)
        output_layer = tf.keras.layers.Dense(
            units=1,
            activation="relu",
            name="output_layer",
        )(dense_layer)

        # We need to be careful not to end up with a disconnected graph when combining
        # the preprocessing model and the rest of the training.

        model = tf.keras.Model(
            inputs=prepro_model.inputs,
            outputs=output_layer,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
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
        project_name="output/test_keras_tuner_simple",
    )

    # Create some fake data for training and validation. This will be used in the keras
    # tuner to train and evaluate the model.
    x_train = [
        tf.constant(
            [
                [1.0],
                [2.0],
                [3.0],
                [4.0],
                [5.0],
                [6.0],
            ]
        ),
        tf.constant(
            [
                [45.0],
                [48.0],
                [51.0],
                [54.0],
                [57.0],
                [60.0],
            ]
        ),
        tf.constant(
            [
                [5.0],
                [8.0],
                [1.0],
                [4.0],
                [7.0],
                [0.0],
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

    x_val = [
        tf.constant(
            [
                [3.0],
                [5.0],
                [6.0],
            ]
        ),
        tf.constant(
            [
                [45.0],
                [48.0],
                [54.0],
            ]
        ),
        tf.constant(
            [
                [5.0],
                [8.0],
                [4.0],
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
    best_model.save("output/test_keras_tuner_simple_best_model")

    print("Loading best model")
    loaded_best_model = tf.keras.models.load_model(
        "output/test_keras_tuner_simple_best_model"
    )

    print("Predict with best model")
    print(loaded_best_model.predict(x_val))
