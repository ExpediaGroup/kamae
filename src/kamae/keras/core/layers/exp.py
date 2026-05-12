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

import keras
from keras import ops

import kamae
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input


@keras.saving.register_keras_serializable(package=kamae.__name__)
class ExpLayer(BaseLayer):
    """
    Performs the exp(x) operation on a given input tensor.
    """

    jit_compatible = True

    def __init__(
        self,
        name: Optional[str] = None,
        input_dtype: Optional[str] = None,
        output_dtype: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the exp layer

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

        :returns: List of compatible dtype names
        """
        return [
            "bfloat16",
            "float16",
            "float32",
            "float64",
            "complex64",
            "complex128",
        ]

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Performs the exp(x) operation on a given input tensor.

        Decorated with `@enforce_single_tensor_input` to ensure that the input
        is a single tensor. Raises an error if multiple tensors are passed
        in as an iterable.

        :param inputs: Tensor to perform the exp(x) operation on.
        :returns: The exp of the input tensor.
        """
        return ops.exp(inputs)

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the exp layer.
        Used for saving and loading from a model.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        return config
