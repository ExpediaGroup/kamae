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

from functools import reduce
from typing import Any, Dict, Iterable, List, Optional, Union

import tensorflow as tf

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import allow_single_or_multiple_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class MultiplyLayer(BaseLayer):
    """
    Performs the multiply(x, y) operation on a given input tensor.
    If multiplier is not set, inputs are assumed to be a list of tensors and multiplied.
    If multiplier is set, inputs must be a tensor.
    """

    def __init__(
        self,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        multiplier: float = None,
        **kwargs,
    ) -> None:
        """
        Initializes the MultiplyLayer layer

        :param name: Name of the layer, defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param multiplier: The multiplier to multiply the input by, defaults to `None`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.multiplier = multiplier

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [
            tf.bfloat16,
            tf.float16,
            tf.float32,
            tf.float64,
            tf.uint8,
            tf.int8,
            tf.uint16,
            tf.int16,
            tf.int32,
            tf.int64,
            tf.complex64,
            tf.complex128,
        ]

    @allow_single_or_multiple_tensor_input
    def _call(self, inputs: Union[Tensor, Iterable[Tensor]], **kwargs) -> Tensor:
        """
        Performs the multiply(x, y) operation on either an iterable of input tensors or
        a single input tensor and a constant.

        Decorated with `@allow_single_or_multiple_tensor_input` to ensure that the input
        is either a single tensor or an iterable of tensors. Returns this result as a
        list of tensors for easier use here.

        :param inputs: Single tensor or iterable of tensors to perform the
        multiply(x, y) operation on.
        :returns: The tensor resulting from the multiply(x, y) operation.
        """
        if self.multiplier is not None:
            if len(inputs) > 1:
                raise ValueError("If multiplier is set, cannot have multiple inputs")
            cast_input, cast_multiplier = self._force_cast_to_compatible_numeric_type(
                inputs[0], self.multiplier
            )
            return tf.math.multiply(
                cast_input,
                cast_multiplier,
            )
        else:
            if not len(inputs) > 1:
                raise ValueError("If multiplier is not set, must have multiple inputs")

            return reduce(tf.math.multiply, inputs)

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the Multiply layer.
        Used for saving and loading from a model.

        Specifically adds the `multiplier` to the config dictionary.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update({"multiplier": self.multiplier})
        return config
