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

import jax.numpy as jnp
import optax
import tensorflow as tf
from flax import nnx as nnx
from jax.experimental import jax2tf
from pyspark.sql import SparkSession

from kamae.spark.estimators import StandardScaleEstimator
from kamae.spark.pipeline import KamaeSparkPipeline
from kamae.spark.transformers import ArrayConcatenateTransformer, LogTransformer

# Create some fake data for fitting, training and validation.
x_schema = ["col1", "col2", "col3"]

x_fit = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
]

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

y_train = jnp.array(
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

y_val = jnp.array(
    [
        [0.8104625],
        [0.7484849],
        [0.95002884],
    ]
)

x_predict = [
    tf.constant(
        [
            [3.0],
        ]
    ),
    tf.constant(
        [
            [45.0],
        ]
    ),
    tf.constant(
        [
            [5.0],
        ]
    ),
]

if __name__ == "__main__":
    print("""Starting test of Spark pipeline, integration with TensorFlow & Jax/Flax""")

    print("\n* Creating processing pipeline\n")

    spark = SparkSession.builder.getOrCreate()
    fit_data = spark.createDataFrame(x_fit, x_schema)

    # Setup transformers, can use set methods or just pass in the args to the constructor.
    log_transformer = (
        LogTransformer()
        .setInputCol("col1")
        .setOutputCol("log_col1")
        .setAlpha(1)
        .setLayerName("log_one_plus_x")
    )
    log_transformer2 = (
        LogTransformer()
        .setInputCol("col2")
        .setOutputCol("log_col2")
        .setAlpha(1)
        .setLayerName("log2_one_plus_x")
    )
    vector_assembler = ArrayConcatenateTransformer(
        inputCols=["log_col1", "log_col2", "col3"],
        outputCol="features",
    ).setLayerName("vec_assemble_log_col1_col2_col3")

    standard_scalar_layer = StandardScaleEstimator(
        inputCol="features",
        outputCol="scaled_features",
    ).setLayerName("numericals_scaled")

    test_pipeline = KamaeSparkPipeline(
        stages=[
            log_transformer,
            log_transformer2,
            vector_assembler,
            standard_scalar_layer,
        ]
    )

    print("\n* Fit preprocessing pipeline and convert it to a Keras model\n")

    fit_pipeline = test_pipeline.fit(fit_data)
    fit_pipeline.transform(fit_data).show()

    tf_input_schema = [
        {
            "name": col,
            "dtype": tf.float32,
            "shape": (1,),
        }
        for col in x_schema
    ]

    tf_preproc_model = fit_pipeline.build_keras_model(tf_input_schema=tf_input_schema)
    tf_preproc_model.summary()

    print("\n* Build and train a JAX neural network\n")

    # Create a JAX neural network using Flax NNX
    #
    # This model will serve both for training and for inference
    #
    class Model(nnx.Module):
        def __init__(self, input_units, dense_units, output_units):
            def create_dense(inp, out):
                return nnx.Linear(
                    in_features=inp,
                    out_features=out,
                    rngs=nnx.Rngs(1),
                    kernel_init=nnx.initializers.glorot_uniform(),
                )

            self.dense = create_dense(input_units, dense_units)
            self.output = create_dense(dense_units, output_units)

        def __call__(self, x):
            x = self.dense(x)
            x = nnx.relu(x)
            x = self.output(x)
            return nnx.relu(x)

    # Method for supervised training of the JAX neural network
    #
    # It executes the model, calculate the loss based on the expected output, calculate the gradients with
    # auto-differentiation, and apply the gradient descent using the provided optimizer
    #
    # For speeding up its execution, the method is serialized to XLA (JIT) before being executed
    #
    @nnx.jit
    def train_step(model, optimizer, x, y):
        def loss_fn(mdl):
            preds = mdl(x)
            return jnp.mean(jnp.square(y - preds))  # mse

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)  # in-place updates
        return loss

    # Instantiate the model and its optimizer, then train the model.
    #
    # Model is trained with a single batch of fake data, but in the real world that training step should
    # be invoked in a loop for every batch of data, repeatedly for each epoch
    #
    jax_nn_model = Model(3, 256, 1)
    jax_optimizer = nnx.Optimizer(jax_nn_model, optax.adam(1e-3))
    num_epochs = 50

    for epoch in range(num_epochs):
        x_transformed = tf_preproc_model(x_train)["scaled_features"]
        loss = train_step(
            jax_nn_model,
            jax_optimizer,
            jnp.array(x_transformed),
            y_train,
        )
        print(f"Epoch: {epoch} Loss: {loss}")

    print("\n* Save preprocessing model and neural network as a Keras model\n")

    # Bundle our Keras preprocessing model and our JAX neural network together in a single new Keras model
    #
    # We use jax2tf to convert the JAX/Flax model to a TF function, then map this function into a Keras Lambda
    # that receives in input the output of the preprocessing layer.
    #
    # Interesting facts:
    #
    #   - Normally we would need to fix the shapes of the preprocessing layer, since JAX function only works
    #     with fully-known shape, but there is a feature in development (polymorphic_shapes) that allow us to continue
    #     to pass an unknown batch size. So far, looks like this is working...
    #
    #   - We cannot enable native serialization (XLA) because TensorFlow will complain that the resulting op sequence
    #     is not compatible with the version of XLA supported by our version of TensorFlow (Kamae requires TF <= 2.14).
    #     Though disabling native serialization is being deprecated and will stop working in a future release of JAX.
    #
    #   - Here, the JAX model weights will be frozen as constants into the TensorFlow saved model. It is
    #     fine to do so, since there is no need to retrain that model. Though if there are too many parameters, we might
    #     blow up the limits allowed by the saved model protobuf definition, in which case we need to preserve the
    #     weights as variables.
    #
    k_preproc_output = tf_preproc_model.outputs[0]
    tf_nn_fn = tf.function(
        jax2tf.convert(
            jax_nn_model,
            polymorphic_shapes=str(k_preproc_output.shape).replace("None", "b"),
            native_serialization=False,
        ),
        autograph=False,  # That must always be disabled when using JAX
    )
    k_model = tf.keras.Model(
        inputs=tf_preproc_model.inputs,
        outputs=tf.keras.layers.Lambda(tf_nn_fn)(k_preproc_output),
    )
    k_model.compile()
    k_model.summary()

    k_model.save("./output/test_model", save_format="tf")

    # Save and reload the model, then run a few predictions validating that the model state remains the same
    # after serialization
    #
    print("\n* Predict with JAX (original model)\n")

    def predict_with_jax(x):
        x_transformed = tf_preproc_model(x)["scaled_features"]
        return jax_nn_model(jnp.array(x_transformed))

    print(predict_with_jax(x_val))
    print(predict_with_jax(x_predict))

    print("\n* Predict with Keras\n")

    k_loaded_model = tf.keras.saving.load_model("./output/test_model")

    print(k_loaded_model(x_val))
    print(k_loaded_model(x_predict))

    print("\n* Predict with TensorFlow\n")

    tf_loaded_model = tf.saved_model.load("./output/test_model")

    print(tf_loaded_model(x_val))
    print(tf_loaded_model(x_predict))
