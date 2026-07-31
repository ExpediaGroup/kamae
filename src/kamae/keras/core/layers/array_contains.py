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

import keras
from keras import KerasTensor, ops

import kamae
from kamae.keras.core.backend import ALL_BACKENDS
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.utils.input_utils import enforce_multiple_tensor_input


@keras.saving.register_keras_serializable(package=kamae.__name__)
class ArrayContainsLayer(BaseLayer):
    """
    Computes whether a value is contained in an array along the last axis.

    Expects two inputs `(array, value)` that broadcast on every axis except the
    last. The `array` tensor holds the dimension to search over (e.g. shape
    `(B, 1, N)`), while the `value` tensor has size 1 on the last axis (e.g.
    shape `(B, L, 1)`). The output is the broadcast shape with the last axis
    collapsed to 1 (e.g. `(B, L, 1)`), containing `1.0` where the value is found
    and `0.0` otherwise. Both inputs must share the same dtype; use `input_dtype`
    to cast them to a common dtype.
    """

    supported_backends = ALL_BACKENDS
    jit_compatible = True

    def __init__(
        self,
        name: Optional[str] = None,
        input_dtype: Optional[str] = None,
        output_dtype: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the ArrayContainsLayer layer.

        :param name: Name of the layer, defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )

    @property
    def compatible_dtypes(self) -> Optional[List[str]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: List of compatible dtype names.
        """
        return [
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "int64",
            "uint64",
            "float16",
            "float32",
            "float64",
        ]

    @enforce_multiple_tensor_input
    def _call(self, inputs: Iterable[KerasTensor], **kwargs: Any) -> KerasTensor:
        """
        Computes membership of `value` within `array` along the last axis.

        Decorated with `@enforce_multiple_tensor_input` to ensure that the input
        is an iterable of tensors. Raises an error if a single tensor is passed.

        After decoration, we check the length of the inputs to ensure we have the
        right number of input tensors.

        :param inputs: List of two tensors `(array, value)` to compute membership
        over.
        :returns: The tensor resulting from the membership operation, with `1.0`
        where the value is found and `0.0` otherwise.
        """
        if len(inputs) != 2:
            raise ValueError(
                f"Expected 2 inputs, received {len(inputs)} inputs instead."
            )
        array, value = inputs
        any_match = ops.any(ops.equal(array, value), axis=-1, keepdims=True)
        return ops.cast(any_match, "float32")

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the ArrayContains layer.
        Used for saving and loading from a model.

        :returns: Dictionary of the configuration of the layer.
        """
        return super().get_config()
