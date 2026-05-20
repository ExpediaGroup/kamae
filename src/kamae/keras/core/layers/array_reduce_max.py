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
from keras import KerasTensor, ops

import kamae
from kamae.keras.core.backend import ALL_BACKENDS
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input


@keras.saving.register_keras_serializable(package=kamae.__name__)
class ArrayReduceMaxLayer(BaseLayer):
    """
    Reduces the last dimension of a tensor by taking the maximum.

    Input:  (..., N)
    Output: (...)

    NaN values in the result are replaced with the configured default_value.
    """

    supported_backends = ALL_BACKENDS
    jit_compatible = True

    def __init__(
        self,
        name: Optional[str] = None,
        input_dtype: Optional[str] = None,
        output_dtype: Optional[str] = None,
        default_value: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.default_value = default_value

    @property
    def compatible_dtypes(self) -> Optional[List[str]]:
        return [
            "bfloat16",
            "float16",
            "float32",
            "float64",
        ]

    @enforce_single_tensor_input
    def _call(self, inputs: KerasTensor, **kwargs: Any) -> KerasTensor:
        result = ops.max(inputs, axis=-1)
        return ops.where(
            ops.isnan(result),
            ops.cast(self.default_value, dtype=result.dtype),
            result,
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"default_value": self.default_value})
        return config
