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

from typing import Any, Callable, Dict, List, Union

import keras_tuner as kt
import tensorflow as tf
from sklearn.pipeline import Pipeline

from kamae.graph import PipelineGraph


class KamaeSklearnPipeline(Pipeline):
    """
    KamaeSklearnPipeline is a subclass of sklearn.pipeline.Pipeline that is used to
    chain together BaseTransformers. It maintains the same functionality
    as sklearn.pipeline.Pipeline e.g. serialisation.
    """

    def __init__(self, steps, *, memory=None, verbose=False) -> None:
        """
        Initializes a KamaeSklearnPipeline object.

        :param steps: List of tuples containing the name and LayerTransformer
        :param memory: str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. The last step
        will never be cached, even if it is a transformer. By default, no
        caching is performed. If a string is given, it is the path to the
        caching directory. Enabling caching triggers a clone of the transformers
        before fitting. Therefore, the transformer instance given to the
        pipeline cannot be inspected directly. Use the attribute ``named_steps``
        or ``steps`` to inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.
        :param verbose: If True, the time elapsed while fitting each step
        will be printed as it is completed.
        """
        super().__init__(steps, memory=memory, verbose=verbose)

    def get_all_tf_layers(self) -> List[tf.keras.layers.Layer]:
        """
        Gets a list of all tensorflow layers in the pipeline model.

        :returns: List of tensorflow layers within the pipeline model.
        """
        return [step[1].get_tf_layer() for step in self.steps]

    def build_keras_model(
        self, tf_input_schema: Union[List[tf.TypeSpec], List[Dict[str, Any]]]
    ) -> tf.keras.Model:
        """
        Builds a keras model from the pipeline using the PipelineGraph
        helper class.

        :param tf_input_schema: List of dictionaries containing the input schema for
        the model. Specifically the name, shape and dtype of each input.
        These will be passed as is to the Keras Input layer.
        :returns: Keras model.
        """
        stage_dict = {
            step[1].layer_name: step[1].construct_layer_info() for step in self.steps
        }
        pipeline_graph = PipelineGraph(stage_dict=stage_dict)
        return pipeline_graph.build_keras_model(tf_input_schema=tf_input_schema)

    def get_keras_tuner_model_builder(
        self,
        tf_input_schema: Union[List[tf.TypeSpec], List[Dict[str, Any]]],
        hp_dict: Dict[str, List[Dict[str, Any]]],
    ) -> Callable[[kt.HyperParameters], tf.keras.Model]:
        """
        Builds a keras tuner model builder (function) from the pipeline model
        using the PipelineGraph helper class.

        :param tf_input_schema: List of dictionaries containing the input schema for
        the model. Specifically the name, shape and dtype of each input.
        These will be passed as is to the Keras Input layer.
        :param hp_dict: Dictionary containing the hyperparameters for the model.
        :returns: Keras tuner model builder (function).
        """
        stage_dict = {
            step[1].layer_name: step[1].construct_layer_info() for step in self.steps
        }
        pipeline_graph = PipelineGraph(stage_dict=stage_dict)
        return pipeline_graph.get_keras_tuner_model_builder(
            tf_input_schema=tf_input_schema, hp_dict=hp_dict
        )
