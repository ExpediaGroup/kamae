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
class SumLayer(BaseLayer):
    """
    Performs the sum(x, y) operation on a given input tensor.
    If added is not set, inputs are assumed to be a list of tensors and summed.
    If added is set, inputs must be a tensor.
    """

    def __init__(
        self,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        addend: float = None,
        **kwargs,
    ) -> None:
        """
        Initializes the SumLayer layer

        :param name: Name of the layer, defaults to `None`.
        :param addend: The addend to add to the input, defaults to `None`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.addend = addend

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
            tf.uint16,
            tf.uint32,
            tf.uint64,
            tf.int8,
            tf.int16,
            tf.int32,
            tf.int64,
            tf.complex64,
            tf.complex128,
        ]

    @allow_single_or_multiple_tensor_input
    def _call(self, inputs: Union[Tensor, Iterable[Tensor]], **kwargs) -> Tensor:
        """
        Performs the sum(x, y) operation on either an iterable of input tensors or
        a single input tensor and a constant.

        Decorated with `@allow_single_or_multiple_tensor_input` to ensure that the input
        is either a single tensor or an iterable of tensors. Returns this result as a
        list of tensors for easier use here.

        :param inputs: Single tensor or iterable of tensors to perform the
        sum(x, y) operation on.
        :returns: The tensor resulting from the sum(x, y) operation.
        """
        if self.addend is not None:
            if len(inputs) > 1:
                raise ValueError("If addend is set, cannot have multiple inputs")
            cast_input, cast_addend = self._force_cast_to_compatible_numeric_type(
                inputs[0], self.addend
            )
            return tf.math.add(
                cast_input,
                cast_addend,
            )
        else:
            if not len(inputs) > 1:
                raise ValueError("If addend is not set, must have multiple inputs")
            return reduce(tf.math.add, inputs)

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the Sum layer.
        Used for saving and loading from a model.

        Specifically adds the `addend` to the config dictionary.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update({"addend": self.addend})
        return config
