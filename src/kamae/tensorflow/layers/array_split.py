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

from typing import Any, Dict, List, Optional

import tensorflow as tf

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import enforce_single_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(kamae.__name__)
class ArraySplitLayer(BaseLayer):
    """
    Performs a splitting of the input tensor into a list of tensors.
    Expands dimensions to ensure the output tensors are the same shape as the input.
    """

    def __init__(
        self,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        axis: int = -1,
        **kwargs,
    ) -> None:
        """
        Initialises the ArraySplitLayer layer.

        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param axis: Axis to split on. Defaults to -1.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.axis = axis

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return None

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> List[Tensor]:
        """
        Splits the input tensor along the specified axis.

        Decorated with `@enforce_single_tensor_input` to ensure that the input
        is a single tensor. Raises an error if an iterable of tensors is passed
        in.

        :param inputs: Tensor to split.
        :returns: List of split tensors.
        """
        return [
            tf.expand_dims(y, axis=self.axis)
            for y in tf.unstack(inputs, axis=self.axis)
        ]

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the VectorSplit layer.
        Used for saving and loading from a model.

        Specifically, adds the `axis` to the config.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update({"axis": self.axis})
        return config
