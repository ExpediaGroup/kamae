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

import keras
from keras import ops

import kamae
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import allow_single_or_multiple_tensor_input
from kamae.keras.core.utils.ops_utils import divide_no_nan


@keras.saving.register_keras_serializable(package=kamae.__name__)
class DivideLayer(BaseLayer):
    """
    Performs the divide(x, y) operation on a given input tensor. If divisor is not set,
    inputs must be a list. If divisor is set, inputs must be a tensor.

    This is a backend-agnostic layer that works with TensorFlow, JAX, and PyTorch.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        input_dtype: Optional[str] = None,
        output_dtype: Optional[str] = None,
        divisor: Optional[float] = None,
        **kwargs: Any,
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
    def compatible_dtypes(self) -> Optional[List[str]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        # No int support here because when dividing two ints the result is a float64.
        # And when we have multiple inputs we perform a reduce operation, which will
        # error for the any inputs of size > 2 since we then try to divide a float64
        # by an int.
        return [
            "bfloat16",
            "float16",
            "float32",
            "float64",
        ]

    @allow_single_or_multiple_tensor_input
    def _call(self, inputs: Union[Tensor, Iterable[Tensor]], **kwargs: Any) -> Tensor:
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
            divisor_tensor = ops.cast(self.divisor, dtype=inputs[0].dtype)
            return divide_no_nan(inputs[0], divisor_tensor)
        else:
            if not len(inputs) > 1:
                raise ValueError("If divisor is not set, must have multiple inputs")
            return reduce(divide_no_nan, inputs)

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
