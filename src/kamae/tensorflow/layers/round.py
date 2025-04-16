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


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class RoundLayer(BaseLayer):
    """
    Performs a standard rounding operation on the input tensor.
    Supported rounding types are 'ceil', 'floor' and 'round'.

    - 'ceil' rounds up to the nearest integer.
    - 'floor' rounds down to the nearest integer.
    - 'round' rounds to the nearest integer.
    """

    def __init__(
        self,
        round_type: str = "round",
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        **kwargs,
    ) -> None:
        """
        Initialises the RoundLayer layer.

        :param round_type: The type of rounding to perform.
        Supported types are 'ceil', 'floor' and 'round'. Defaults to 'round'.
        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        if round_type not in ["ceil", "floor", "round"]:
            raise ValueError("""roundType must be one of 'ceil', 'floor' or 'round'.""")
        self.round_type = round_type

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [tf.float16, tf.float32, tf.float64]

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> Tensor:
        """
        Performs the rounding operation on the input tensor.

        Decorated with `@enforce_single_tensor_input` to ensure that the input is a
        single tensor. Raises an error if multiple tensors are passed in as an iterable.

        :param inputs: Input tensor to perform the rounding on.
        :returns: The input tensor with the rounding applied.
        """
        if self.round_type == "ceil":
            return tf.math.ceil(inputs)
        elif self.round_type == "floor":
            return tf.math.floor(inputs)
        elif self.round_type == "round":
            return tf.math.round(inputs)
        else:
            raise ValueError("""roundType must be one of 'ceil', 'floor' or 'round'.""")

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the Round layer.
        Used for saving and loading from a model.

        Specifically adds the `round_type` value to the configuration.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update({"round_type": self.round_type})
        return config
