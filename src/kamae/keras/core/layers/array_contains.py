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

from typing import Any, Dict, Iterable, List, Optional, Union

import keras
from keras import KerasTensor, ops

import kamae
from kamae.keras.core.backend import ALL_BACKENDS
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.utils.input_utils import allow_single_or_multiple_tensor_input


@keras.saving.register_keras_serializable(package=kamae.__name__)
class ArrayContainsLayer(BaseLayer):
    """
    Computes whether a value is contained in an array along a given axis.

    With `value_constant` set, takes a single `array` input and checks whether
    that constant is present. Otherwise takes two inputs `(array, value)`.
    Returns a boolean; set `output_dtype` to cast the result.

    Example:
        >>> layer = ArrayContainsLayer(value_constant=2)
        >>> layer([[1, 2, 3], [4, 5, 6]])
        [[True], [False]]
    """

    supported_backends = ALL_BACKENDS
    jit_compatible = True

    def __init__(
        self,
        name: Optional[str] = None,
        input_dtype: Optional[str] = None,
        output_dtype: Optional[str] = None,
        axis: int = -1,
        keepdims: bool = True,
        value_constant: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the ArrayContainsLayer layer.

        :param name: Name of the layer, defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param axis: The axis along which to check for the value. Defaults to
        `-1`.
        :param keepdims: Whether to keep the reduced axis as a size 1 dimension
        in the output. Defaults to `True`.
        :param value_constant: Optional constant to check for. If set, the layer
        takes a single `array` input. If `None`, two inputs `(array, value)` are
        required. Defaults to `None`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.axis = axis
        self.keepdims = keepdims
        self.value_constant = value_constant

    @property
    def compatible_dtypes(self) -> Optional[List[str]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: List of compatible dtype names.
        """
        return [
            "bfloat16",
            "float16",
            "float32",
            "float64",
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "int64",
            "uint64",
        ]

    @allow_single_or_multiple_tensor_input
    def _call(
        self, inputs: Union[KerasTensor, Iterable[KerasTensor]], **kwargs: Any
    ) -> KerasTensor:
        """
        Computes whether `value` is present in `array` along the given axis.

        :param inputs: Single `array` tensor (when `value_constant` is set) or a
        list of two tensors `(array, value)`.
        :returns: A boolean tensor, `True` where the value is found.
        """
        if self.value_constant is not None and len(inputs) == 1:
            # Constant value given upon initialization
            array, value = inputs[0], self.value_constant
            array, value = self._force_cast_to_compatible_numeric_type(array, value)
        elif self.value_constant is None and len(inputs) == 2:
            # Dynamic value given
            array, value = inputs
        elif self.value_constant is not None:
            raise ValueError("Expected 1 input when `value_constant` is set")
        else:
            raise ValueError(f"Expected 2 inputs, got {len(inputs)} inputs instead")

        return ops.any(ops.equal(array, value), axis=self.axis, keepdims=self.keepdims)

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the ArrayContains layer.
        Used for saving and loading from a model.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
                "keepdims": self.keepdims,
                "value_constant": self.value_constant,
            }
        )
        return config
