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

from typing import Any, Dict, List, Optional, Union

import keras
from keras import ops

import kamae
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input
from kamae.keras.core.utils.tensor_utils import get_dtype_max


@keras.saving.register_keras_serializable(package=kamae.__name__)
class ArraySubtractMinimumLayer(BaseLayer):
    """
    Computes the difference across an axis from the minimum non-padded element
    in the input tensor.

    It takes a tensor of numerical value and calculates the differences between
    each value and the minimum value in the tensor. The calculation preserves
    the pad value elements.

    The principal use case for this layer is to calculate the time difference
    from the first event to all events in a sequence, where the tensor is an array of
    timestamps.
    """

    jit_compatible = True

    def __init__(
        self,
        name: Optional[str] = None,
        input_dtype: Optional[str] = None,
        output_dtype: Optional[str] = None,
        axis: int = -1,
        pad_value: Optional[Union[int, float]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialises the ArraySubtractMinimum layer.

        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param axis: The axis along which the differences are calculated.
        Defaults to -1.
        :param pad_value: The value to be considered as padding. Defaults to `None`.
        :returns: None
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.axis = axis
        self.pad_value = pad_value

    @property
    def compatible_dtypes(self) -> Optional[List[str]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [
            "bfloat16",
            "float16",
            "float32",
            "float64",
            "uint8",
            "int8",
            "uint16",
            "int16",
            "int32",
            "int64",
            "uint32",
            "uint64",
        ]

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Performs the calculation of the differences on the input tensor.

        Example:
         input_tensor = [[19, 18, 13, 11, 10, -1, -1, -1],
                        [12, 2, 1, -1, -1, -1, -1, -1]]
         layer = ArraySubtractMinimumLayer(pad_value=-1)
         differences = layer(input_tensor)
         Output: [[9, 8, 3, 1, 0, -1, -1, -1],
                 [11, 1, 0, -1, -1, -1, -1, -1]]

        :param inputs: The input tensor.
        :returns: Tensor of differences from the minimum (non-padded) value.
        """
        if self.pad_value is None:
            # If pad value is not defined, then the smallest value in the tensor is
            # considered as the first value and subtracted from all the values.
            first_value = ops.min(inputs, axis=self.axis)
            subtracted_val = ops.subtract(
                inputs, ops.expand_dims(first_value, self.axis)
            )
            return subtracted_val

        # Otherwise, we find the smallest non padded value and subtract it from all
        # the values. Padded values are preserved.
        inputs, pad_tensor = self._force_cast_to_compatible_numeric_type(
            inputs, self.pad_value
        )

        # Get the dtype max value for masking
        dtype_str = keras.backend.standardize_dtype(inputs.dtype)
        dtype_max = get_dtype_max(dtype_str)
        dtype_max_tensor = ops.convert_to_tensor(dtype_max, dtype=inputs.dtype)

        first_non_pad_value = ops.min(
            ops.where(ops.equal(inputs, pad_tensor), dtype_max_tensor, inputs),
            axis=self.axis,
        )
        subtracted_val = ops.subtract(
            inputs, ops.expand_dims(first_non_pad_value, self.axis)
        )
        return ops.where(ops.equal(inputs, pad_tensor), inputs, subtracted_val)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the layer.
        Used for saving and loading from a model.

        :returns: Dictionary of the configuration of the layer
        """
        config = super().get_config()
        config.update(
            {
                "pad_value": self.pad_value,
                "axis": self.axis,
            }
        )
        return config
