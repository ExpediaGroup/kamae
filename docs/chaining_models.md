# Model Chaining: Combining Kamae and Trained Keras Models

Below we will explain how you can combine your Kamae pre/post processing model with a trained Keras model.

## Kamae Processing Models

When building your Kamae model from the pipeline interfaces, you will call the `build_keras_model` method on the pipeline object.
This method will return a Keras model that you can use to process your data.

### Accessing model inputs

The way in which you specify the `tf_input_schema` to this method can influence how you access your model inputs.

#### 1. **List of dictionary config.** 

This is the standard way of specifying the `tf_input_schema`. 
In this case, you would pass the `tf_input_schema` as a list of dictionaries, where each dictionary specifies (at least) the name, shape and type of the input.
These dictionaries will be passed directly into the [`tf.keras.layers.Input`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/InputLayer) via ** kwargs, and so the names of the arguments will be the keys specified in the dictionary.

   
In this case, when accessing your model inputs, you can use the `inputs` attribute of the model, which is a list of `tf.keras.Input` objects.
You can access the `name` attribute of each of these objects to get the name of the input.
These will match the names specified in the `tf_input_schema` dictionary.

#### 2. **List of tf.TypeSpec.**

If you have more complex inputs (e.g. a [`RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor)) then you may find using [`tf.TypeSpec`](https://www.tensorflow.org/api_docs/python/tf/TypeSpec?hl=en) objects easier.
In this case, you would pass the `tf_input_schema` as a list of `tf.TypeSpec` objects.
Under the hood, these will be passed to the [`tf.keras.layers.Input`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/InputLayer) via the `typespec` argument.

However, in this case, accessing the inputs of your model via the `inputs` attribute will return inputs with missing names (i.e. `None`). This is detailed in this [GitHub issue](https://github.com/keras-team/tf-keras/issues/406).

In order to fix this you will need to zip the `input_names` attribute of your model with the `inputs` attribute, to assign the names to the inputs.

```python
inputs_with_names = list(zip(model.input_names, model.inputs))
```

### Accessing model outputs

With your Kamae model, you can access the outputs of the model via the `outputs` attribute of the model.
We add an Identity layer to each output (to preserve the name of the output), but this means that the `name`
attribute of the output will be `<OUTPUT_NAME>/Identity:0`.

Therefore, you can either split these strings, or zip the `output_names` attribute of your model with the `outputs` attribute, to assign the names to the outputs.


## Combining your Kamae processing model with a trained Keras model

### Preprocessing example

Assuming we have two models, `prepro_model` and `trained_model` which we want to chain together, we can do the following:

```python
import tensorflow as tf

# Get the inputs of the prepro model
prepro_inputs = prepro_model.inputs

# If you need to access the names of the inputs, you can do the following
prepro_inputs_dict = {
    input_name: input 
    for input_name, input in zip(prepro_model.input_names, prepro_model.inputs)
}

# Get the outputs of the prepro model as a dictionary
prepro_outputs_dict = {
    output_name: output 
    for output_name, output in zip(prepro_model.output_names, prepro_model.outputs)
}

# Apply trained model to prepro outputs
combined_outputs = trained_model(prepro_outputs_dict)

# Create a new model with the prepro inputs and combined outputs
combined_model = tf.keras.Model(inputs=prepro_inputs, outputs=combined_outputs)
```

### Postprocessing example

Postprocessing works in a very similar way, you just change which model is applied to the other:

```python
import tensorflow as tf

# Get the inputs of the trained model
trained_inputs = trained_model.inputs

# If you need to access the names of the inputs, you can do the following
trained_inputs_dict = {
    input_name: input 
    for input_name, input in zip(trained_model.input_names, trained_model.inputs)
}

# Get the outputs of the trained model as a dictionary
trained_outputs_dict = {
    output_name: output 
    for output_name, output in zip(trained_model.output_names, trained_model.outputs)
}

# Apply postpro model to trained outputs
combined_outputs = postpro_model(trained_outputs_dict)

# Create a new model with the trained inputs and combined outputs
combined_model = tf.keras.Model(inputs=trained_inputs, outputs=combined_outputs)
```
