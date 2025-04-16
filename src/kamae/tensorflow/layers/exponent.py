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
from kamae.tensorflow.utils import allow_single_or_multiple_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class ExponentLayer(BaseLayer):
    """
    Performs the x^exponent operation on a given input tensor
    """

    def __init__(
        self,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        exponent: float = None,
        **kwargs,
    ) -> None:
        """
        Initializes the exponent layer

        :param name: Name of the layer, defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param exponent: The exponent to raise the input to, defaults to `None`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.exponent = exponent

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [
            tf.float16,
            tf.float32,
            tf.float64,
            tf.complex64,
            tf.complex128,
        ]

    @allow_single_or_multiple_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> Tensor:
        """
        Performs the x^exponent operation on a given input tensor.

        Decorated with `@allow_single_or_multiple_tensor_input` to ensure that the input
        is either a single tensor or an iterable of tensors. Returns this result as a
        list of tensors for easier use here.

        :param inputs: Single tensor or iterable of tensors to perform the x^pow
         operation on.
        :returns: The tensor raised to the power of the exponent.
        """
        if self.exponent is not None:
            if len(inputs) > 1:
                raise ValueError("If exponent is set, cannot have multiple inputs")
            return tf.math.pow(
                inputs[0],
                self._cast(tf.constant(self.exponent), cast_dtype=inputs[0].dtype.name),
            )
        else:
            if not len(inputs) == 2:
                raise ValueError("If exponent is not set, must have exactly 2 inputs")
            return tf.math.pow(inputs[0], inputs[1])

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the exp layer.
        Used for saving and loading from a model.

        Specifically adds the `exponent` to the config dictionary

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update({"exponent": self.exponent})
        return config
