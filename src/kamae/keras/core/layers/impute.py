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

from typing import Any

import keras
from keras import ops

from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input
from kamae.params import _REQUIRED, ParamSpec


class ImputeLayer(BaseLayer):
    """
    Performs imputation on the input.

    Where the input data is equal to the specified mask value, this layer will replace
    the data with the impute value calculated at preprocessing time.

    The impute value is either the mean or median and is computed while ignoring rows
    in the data which are equal to the mask value or are null.
    """

    jit_compatible = True

    _compatible_dtypes = None
    _params = {
        "impute_value": ParamSpec(
            default=_REQUIRED,
            doc="The value to use for imputation",
        ),
        "mask_value": ParamSpec(
            default=_REQUIRED,
            doc="Value which should be replaced by the impute value at inference",
        ),
    }

    def _post_init(self):
        if not isinstance(self.mask_value, type(self.impute_value)):
            raise ValueError(
                "The mask value and impute value must be of the same type."
            )

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Performs imputation on the input tensor(s). It imputes over values which
        are equal to the mask_value.


        :param inputs: Input tensor to perform the imputation on.
        :returns: The input tensor with the imputation applied.
        """
        input_dtype_str = keras.backend.standardize_dtype(inputs.dtype)

        # Check if dtype is numeric (floating or integer)
        if "float" in input_dtype_str or "int" in input_dtype_str:
            inputs, mask = self._force_cast_to_compatible_numeric_type(
                inputs, self.mask_value
            )
            inputs, impute_value = self._force_cast_to_compatible_numeric_type(
                inputs, self.impute_value
            )
        else:
            # For non-numeric types (like strings)
            mask = self._cast(ops.convert_to_tensor(self.mask_value), input_dtype_str)
            impute_value = self._cast(
                ops.convert_to_tensor(self.impute_value), input_dtype_str
            )

        mask = ops.equal(inputs, mask)
        imputed_outputs = ops.where(
            mask,
            impute_value,
            inputs,
        )

        return imputed_outputs
