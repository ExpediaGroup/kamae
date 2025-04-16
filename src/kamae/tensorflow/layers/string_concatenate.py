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

from typing import Any, Dict, Iterable, List, Optional

import tensorflow as tf

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import enforce_multiple_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(kamae.__name__)
class StringConcatenateLayer(BaseLayer):
    """
    Performs a concatenation of the input tensors.
    """

    def __init__(
        self,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        separator: str = "_",
        **kwargs,
    ) -> None:
        """
        Initialises the Concat layer.
        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param separator: The separator to use when joining the input tensors.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.separator = separator

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [tf.string]

    @enforce_multiple_tensor_input
    def _call(self, inputs: Iterable[Tensor], **kwargs) -> Tensor:
        """
        Concatenates the input tensors.

        Decorated with `@enforce_multiple_tensor_input` to ensure that the input is an
        iterable of multiple tensors. Raises an error if a single tensor is passed in.

        :param inputs: Input tensors that will be concatenated on the last axis.
        Must be string tensors.
        :returns: A tensor with the concatenated values - same shape as each of
        the input tensors.
        """
        return tf.strings.join(inputs, separator=self.separator)

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the StringConcatenate layer.
        Used for saving and loading from a model.

        Specifically adds the `separator` to the config.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update({"separator": self.separator})
        return config
