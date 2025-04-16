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

from typing import Any, Dict, List, Tuple, Union

import keras
import networkx as nx
import tensorflow as tf
from packaging.version import Version

from kamae.tensorflow.layers import IdentityLayer

keras_version = Version(keras.__version__)


class PipelineGraph:
    """
    PipelineGraph is a class that constructs a graph of the pipeline stages.
    This is used to determine the order in which the layers should be constructed.
    The graph is built by adding edges between layers that have the same input column
    as the output column of a previous layer. If the input is not an output of any other
    layer, then it is assumed to be an input to the pipeline.

    The graph is then topologically sorted to determine the order in which the layers
    should be constructed. Iterating through this order, the layers are constructed by
    calling the get_tf_layer method of each stage. The inputs to the layer are
    determined by the outputs of the previous layers.
    """

    def __init__(self, stage_dict: Dict[str, Any]) -> None:
        """
        Initialize the PipelineGraph class with a dictionary of stage information.

        :param stage_dict: Dictionary of stages to add to the graph.
        :returns: None - class instance is initialized.
        """
        self.stage_dict = stage_dict
        self.graph = self.add_stage_edges(nx.DiGraph())
        self.transform_order = [
            node for node in nx.topological_sort(self.graph) if node in self.stage_dict
        ]

        # We keep a dictionary of layers to keep track of which have been reused.
        # This allows us to easily get the output layers
        self.layer_store = {}
        self.inputs = {}

    def update_layer_store_with_key(self, layer_key, layer_output) -> None:
        """
        Updates the layer store at a specific key with the layer output and whether
        it was reused. A layer is deemed to be reused if it is already present in
        the layer store.

        :param layer_key: Key to update the layer store with.
        :param layer_output: Layer output to update.
        :returns: None - layer store is updated.
        """
        if layer_key in self.layer_store:
            self.layer_store[layer_key] = {"output": layer_output, "reused": True}
        else:
            self.layer_store[layer_key] = {"output": layer_output, "reused": False}

    def update_layer_store(self, layer_dict: Dict[str, tf.Tensor]) -> None:
        """
        Given a dictionary of layer output names and tensor outputs,
        update the layer store.

        :param layer_dict: Dictionary of layer names and outputs.
        :returns: None - layer store is updated.
        """
        for name, output in layer_dict.items():
            self.update_layer_store_with_key(layer_key=name, layer_output=output)

    def get_layer_output_from_layer_store(self, layer_output_name: str) -> tf.Tensor:
        """
        Given a layer name and index, get the output from the layer store.

        :param layer_output_name: Name of the layer output
        :returns: Tensor output of the layer
        """
        return self.layer_store[layer_output_name]["output"]

    def add_stage_edges(self, graph):
        """
        Add edges to the graph based on the inputs and outputs of each stage.
        Specifically for each stage, we connect the inputs of a stage to itself and the
        stage to its outputs.

        :param graph: NetworkX DAG to add edges to.
        :returns: Graph with edges added.
        """
        edges_to_add = []
        for layer_name, layer_info in self.stage_dict.items():
            # Add edges for all inputs
            edges_to_add.extend(
                [(input_name, layer_name) for input_name in layer_info["inputs"]]
            )
            # Add edges for all outputs
            edges_to_add.extend(
                [(layer_name, output_name) for output_name in layer_info["outputs"]]
            )

        graph.add_edges_from(edges_to_add)
        return graph

    def get_model_outputs(self) -> Dict[str, tf.Tensor]:
        """
        Get the outputs of the model. This is all the layer store outputs that are not
        reused and not inputs. We also apply an identity layer to the outputs so we
        can rename them with the same name as the output columns of the layer.

        :returns: Dictionary of model output tensors.
        """
        return {
            k: IdentityLayer(name=k)(v["output"])
            for k, v in self.layer_store.items()
            if not v["reused"] and k not in self.inputs
        }

    def build_keras_inputs(
        self, tf_input_schema: Union[List[tf.TypeSpec], List[Dict[str, Any]]]
    ) -> None:
        """
        Builds a Keras input layer for the given node.

        Specifically, we get a single node from the out edges of the current node.
        The out edges are a tuple of edges `(u, v)` where `u` is the current node and
        `v` is a node takes the current node (`u`) as input.

        Using `v` we can get the input dimension and type and therefore construct the
        keras input layer. We then store this layer as an input and update the
        layer store.

        :param tf_input_schema: List of tf.TypeSpec objects containing the input schema
        for the model or a list of dict config to be passed to the Input constructor.
        :returns: None - layer store is updated and input layer is stored in the
        inputs dict.
        """

        if isinstance(tf_input_schema, list) and all(
            isinstance(x, tf.TypeSpec) for x in tf_input_schema
        ):
            if keras_version >= Version("3.0.0"):
                raise ValueError(
                    "tf.TypeSpec is not supported in Keras 3, please use a dict config"
                )
            input_config = [
                {
                    "name": spec.name,
                    "type_spec": spec,
                }
                for spec in tf_input_schema
            ]
        elif isinstance(tf_input_schema, list) and all(
            isinstance(x, dict) for x in tf_input_schema
        ):
            input_config = tf_input_schema
        else:
            raise ValueError("tf_input_schema must be a list of tf.TypeSpec or dict!")

        for conf in input_config:
            name = conf.get("name", None)
            if name is None:
                raise ValueError(
                    "Input schema must have names for all inputs, but found None"
                )
            input_layer = tf.keras.layers.Input(**conf)
            self.inputs[name] = input_layer
            self.update_layer_store_with_key(layer_key=name, layer_output=input_layer)

    def sort_inputs(
        self, layer_name: str, input_dict: Dict[str, tf.Tensor]
    ) -> List[tf.Tensor]:
        """
        Sorts the inputs for a given layer based on the order of the inputs in the
        stage dict. This is needed because layers with multiple inputs are not
        guaranteed to be in the correct order when built from the graph as the
        topological sort can be different to the order in the stage dict.

        :param layer_name: Name of the layer
        :param input_dict: Dictionary of inputs for the layer.
        :returns: Sorted list of inputs for the layer.
        """
        stage_inputs = self.stage_dict[layer_name]["inputs"]
        return [input_dict[i] for i in stage_inputs]

    def build_transform_layer_inputs(
        self, node: str, in_edges: List[Tuple[str, str]]
    ) -> List[tf.Tensor]:
        """
        Constructs all the layers that are connected to the current node.
        These are either input layers or the outputs of previous layers.

        The in edges are a tuple of edges `(u, v)` where `v` is the current node and
        `u` is a node that is an input to the current node (`v`).

        Using `u` we can tell if this is the output of another transformation or an
        input layer. If it's an input layer, we retrieve it from the inputs dictionary.
        If it's the output of another transformation, we retrieve it from the
        layer store.

        :param node: Current node (name of layer).
        :param in_edges: List of in edges for the current node.
        :returns: List of layer inputs for the current node/layer.
        """
        # Here we get all layer outputs that are connected to this node.
        # We need these so we can apply the current node's layer to
        # the output of the previous layers.
        # Since we topologically sorted the nodes,
        # all previous layers will have already been created.

        # Get the in edge node names
        in_edge_node_names = [in_edge[0] for in_edge in in_edges]
        # For each in edge, find the output that maps to this node.
        layer_output_from_in_edge = [
            (stage_name, in_edge_node_name)
            for in_edge_node_name in in_edge_node_names
            for stage_name, stage_info in self.stage_dict.items()
            if in_edge_node_name in stage_info["outputs"]
        ]
        # Get any input layers that are connected to the node via the in edges
        input_layers_from_in_edge = {
            name: layer
            for name, layer in self.inputs.items()
            if name in in_edge_node_names
        }

        # For each layer output in edge,
        # get its corresponding layer output from the layer store.
        # Update the store to indicate we have reused this layer.
        # Thus, it is not an output layer
        in_edge_layers_inputs = {}
        for name, output_name in layer_output_from_in_edge:
            layer = self.get_layer_output_from_layer_store(output_name)
            self.update_layer_store_with_key(layer_key=output_name, layer_output=layer)
            in_edge_layers_inputs[output_name] = layer

        # Sort the inputs according to the order set in the Spark transformers
        input_dict = {**in_edge_layers_inputs, **input_layers_from_in_edge}
        layer_inputs = self.sort_inputs(
            layer_name=node,
            input_dict=input_dict,
        )
        return layer_inputs

    @staticmethod
    def override_hyperparameters(
        layer: Union[tf.keras.layers.Layer, List[tf.keras.layers.Layer]],
        hp_override: Dict[str, Any] = None,
    ) -> Union[tf.keras.layers.Layer, List[tf.keras.layers.Layer]]:
        """
        Overrides layer arguments with hyperparameters provided in the
        hyperparameter override dictionary.

        :param layer: Layer to override hyperparameters for.
        :param hp_override: Optional dictionary of hyperparameters to override.
        :returns: Layer with hyperparameters overridden.
        """

        def update_layer(
            layer: tf.keras.layers.Layer, hp_override: Dict[str, Any]
        ) -> tf.keras.layers.Layer:
            config = layer.get_config()
            config.update(hp_override)
            updated_layer = type(layer).from_config(config)
            return updated_layer

        if hp_override is None:
            return layer
        elif isinstance(layer, list):
            overriden_layer = []
            for layer_elem in layer:
                overriden_layer.append(update_layer(layer_elem, hp_override))
            return overriden_layer
        else:
            overriden_layer = update_layer(layer, hp_override)
            return overriden_layer

    def build_keras_transform_layer(
        self,
        node: str,
        in_edges: List[Tuple[str, str]],
        hp_override: Dict[str, Any] = None,
    ) -> None:
        """
        Builds a Keras transformation layer for the given node.
        Gets the layer inputs using the in edges and then applies the layer to
        the inputs. Updates the layer store.

        :param node: Current node (name of layer).
        :param in_edges: List of in edges for the current node.
        :param hp_override: Optional dictionary of hyperparameters to override.
        Used for building Keras tuner model builder functions.
        :returns: None - layer store is updated.
        """
        layer_inputs = self.build_transform_layer_inputs(node=node, in_edges=in_edges)
        layer = self.stage_dict[node]["layer"]
        layer = self.override_hyperparameters(layer=layer, hp_override=hp_override)

        if isinstance(layer, list):
            # If we have a list of layers, we assume that each layer needs to operate
            # on the corresponding input idx in the list of inputs.
            layer_outputs = [
                layer_elem(layer_input)
                for layer_elem, layer_input in zip(layer, layer_inputs)
            ]

        else:
            # If we have a single layer, we assume that it needs to operate on all
            # the inputs.
            layer_outputs = (
                layer(layer_inputs) if len(layer_inputs) > 1 else layer(*layer_inputs)
            )

        layer_output_names = self.stage_dict[node]["outputs"]
        # Make layer outputs a list if it isn't already
        layer_outputs = (
            layer_outputs if isinstance(layer_outputs, list) else [layer_outputs]
        )

        # Zip the output names with the ouputs themselves
        layer_outputs_with_name = {
            layer_output_name: layer_output
            for layer_output_name, layer_output in zip(
                layer_output_names, layer_outputs
            )
        }

        self.update_layer_store(layer_dict=layer_outputs_with_name)

    @staticmethod
    def get_keras_hyperparam_from_config(hp, config):
        """
        Returns a keras hyperparameter object from a config dictionary.

        :param hp: keras_tuner.HyperParameters object passed through from the
        keras tuner model builder function.
        :param config: Config supplied by the user.
        :returns: keras_tuner.HyperParameters method for the given config.
        """

        method_dict = {
            "int": hp.Int,
            "choice": hp.Choice,
            "boolean": hp.Boolean,
            "float": hp.Float,
            "fixed": hp.Fixed,
        }

        hyperparam_dict = {}
        for hyperparams in config:
            try:
                method = method_dict[hyperparams["method"].lower()]
            except KeyError:
                raise ValueError(
                    f"""
                    Hyperparameter method {hyperparams['method']} not supported.
                    Must be one of {",".join(list(method_dict.keys()))}"""
                )

            hyperparam_dict.update(
                {hyperparams["arg_name"]: method(**hyperparams["kwargs"])}
            )
        return hyperparam_dict

    def get_keras_tuner_model_builder(
        self,
        tf_input_schema: Union[List[tf.TypeSpec], List[Dict[str, Any]]],
        hp_dict: Dict[str, List[Dict[str, Any]]],
    ):
        """
        Returns a Keras tuner model builder function for the current graph.
        This allows the user to tune the hyperparameters of the preprocessing model.
        Useful for scenarios where the best preprocessing variables are not known
        a priori. For example, the num_bins to use for a HashIndexLayer.

        :param tf_input_schema: List of tf.TypeSpec objects containing the input schema
        for the model. Specifically the name, shape and dtype of each input.
        These will be passed as is to the Keras Input layer.
        :param hp_dict: Dictionary of possible hyperparameters for each layer.
        Should be of the format:
        {
            "<LAYER_NAME>": [
                {
                    "arg_name": <NAME_OF_LAYER_ARGUMENT>,
                    "method": <NAME_OF_KERAS_HYPERPARAMETER_METHOD>, e.g. "choice"
                    "kwargs": {
                        <KWARGS_TO_PASS_TO_KERAS_HYPERPARAMETER_METHOD>
                    }
                }
            ]
        }
        :returns: Model builder function that takes a keras_tuner.HyperParameters class
        and returns a model.
        """

        transform_order = self.transform_order

        def keras_model_builder(hp):
            # We need to clear the layer store and inputs each time we build a model.
            self.layer_store = {}
            self.inputs = {}
            # Build the input layers
            self.build_keras_inputs(tf_input_schema=tf_input_schema)

            for node in transform_order:
                in_edges = list(self.graph.in_edges(node))

                # Try and get the hyperparameter override.
                try:
                    hp_override = self.get_keras_hyperparam_from_config(
                        hp, hp_dict[node]
                    )
                except KeyError:
                    hp_override = None
                self.build_keras_transform_layer(
                    node=node, in_edges=in_edges, hp_override=hp_override
                )

            sorted_inputs = [self.inputs[k] for k in sorted(self.inputs)]
            return tf.keras.Model(
                inputs=sorted_inputs, outputs=self.get_model_outputs()
            )

        return keras_model_builder

    def build_keras_model(
        self, tf_input_schema: Union[List[tf.TypeSpec], List[Dict[str, Any]]]
    ) -> tf.keras.Model:
        """
        Builds a Keras model from the graph.

        :param tf_input_schema: List of tf.TypeSpec objects containing the input schema
        for the model. Each TypeSpec object must define a unique `name` attribute.
        These will be passed as is to the Keras Input layer.
        :returns: Keras model to be applied to a tensors dictionary: {name: Tensor}.
        """
        # Build the input layers
        self.build_keras_inputs(tf_input_schema=tf_input_schema)

        for node in self.transform_order:
            in_edges = list(self.graph.in_edges(node))
            self.build_keras_transform_layer(node=node, in_edges=in_edges)

        # All the layers are now stored in the layer store,
        # with all inputs/outputs specified.
        # We can now build the model by specifying the inputs and outputs.
        sorted_inputs = {k: self.inputs[k] for k in sorted(self.inputs)}
        return tf.keras.Model(inputs=sorted_inputs, outputs=self.get_model_outputs())
