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

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin

from kamae.sklearn.params import InputOutputExtractor, LayerNameMixin


class BaseTransformerMixin(ABC, LayerNameMixin, InputOutputExtractor):
    """
    Mixin abstract class defining methods needed for all kamae scikit-learn
    transformers.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the transformer.
        """
        super().__init__()

    @abstractmethod
    def get_tf_layer(self) -> Union[tf.keras.layers.Layer, List[tf.keras.layers.Layer]]:
        """
        Gets the tensorflow layer to be used in the model.
        This is the only abstract method that must be implemented.
        :returns: Tensorflow Layer
        """
        raise NotImplementedError

    def construct_layer_info(self) -> Dict[str, Any]:
        """
        Constructs the layer info dictionary.
        Contains the layer name, the tensorflow layer, and the inputs and outputs.
        This is used when constructing the pipeline graph.

        :returns: Dictionary containing layer information such as
        name, tensorflow layer, inputs, and outputs.
        """
        inputs, outputs = self.get_layer_inputs_outputs()
        return {
            "name": self.layer_name,
            "layer": self.get_tf_layer(),
            "inputs": inputs,
            "outputs": outputs,
        }


class BaseTransformer(BaseTransformerMixin, BaseEstimator, TransformerMixin, ABC):
    """
    Abstract class for all scikit-learn transformers. Specifically, this class extends
    the required scikit-learn classes BaseEstimator and TransformerMixin adding in the
    kamae BaseTransformerMixin which defines the methods needed to work with the kamae
    pipeline graph.

    The reason we keep this separate from the BaseTransformerMixin (which is not done
    for Spark) is because on the scikit-learn side we want to allow the ability to
    inherit from existing scikit-learn classes (such as the StandardScaler). In these
    cases the existing class already inherits from BaseEstimator and TransformerMixin
    and so only needs the BaseTransformerMixin (to add kamae specific functionality).
    If you try and inherit these classes twice (once from the existing scikit-learn
    class and once from BaseTransformer) you will get an error. Therefore, we keep
    these separate.

    If you are building an entirely new transformer, then you can inherit from this
    class directly, to save you from having to inherit from BaseEstimator and
    TransformerMixin.

    In Spark, all existing (core) implementations are built in Scala and ported to
    Python. In this case, the ability to re-use existing Spark transformers is very
    difficult and not worth the effort. You can see that for the StandardScaleEstimator
    the logic does not depend on the existing Spark StandardScaler.

    Therefore, we have a single BaseTransformer class for use by all Spark
    transformers.
    """
