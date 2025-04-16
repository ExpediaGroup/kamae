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
class DivideLayer(BaseLayer):
    """
    Performs the divide(x, y) operation on a given input tensor. If divisor is not set,
    inputs must be a list. If divisor is set, inputs must be a tensor.
    """

    def __init__(
        self,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        divisor: float = None,
        **kwargs,
    ) -> None:
        """
        Initializes the DivideLayer layer

        :param name: Name of the layer, defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param divisor: The divisor to divide the input by, defaults to `None`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.divisor = divisor

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        # No int support here because when dividing two ints the result is a float64.
        # And when we have multiple inputs we perform a reduce operation, which will
        # error for the any inputs of size > 2 since we then try to divide a float64
        # by an int.
        return [
            tf.bfloat16,
            tf.float16,
            tf.float32,
            tf.float64,
        ]

    @allow_single_or_multiple_tensor_input
    def _call(self, inputs: Union[Tensor, Iterable[Tensor]], **kwargs) -> Tensor:
        """
        Performs the divide(x, y) operation on either an iterable of input tensors or
        a single input tensor and a constant.

        Decorated with `@allow_single_or_multiple_tensor_input` to ensure that the input
        is either a single tensor or an iterable of tensors. Returns this result as a
        list of tensors for easier use here.

        :param inputs: Single tensor or iterable of tensors to perform the
        divide(x, y) operation on.
        :returns: The tensor resulting from the divide(x, y) operation.
        """
        if self.divisor is not None:
            if len(inputs) > 1:
                raise ValueError("If divisor is set, cannot have multiple inputs")
            return tf.math.divide_no_nan(
                inputs[0], tf.constant(self.divisor, dtype=inputs[0].dtype)
            )
        else:
            if not len(inputs) > 1:
                raise ValueError("If divisor is not set, must have multiple inputs")
            return reduce(tf.math.divide_no_nan, inputs)

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the Divide layer.
        Used for saving and loading from a model.

        Specifically adds the `divisor` to the config dictionary.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update({"divisor": self.divisor})
        return config
