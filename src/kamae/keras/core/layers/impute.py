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


@keras.saving.register_keras_serializable(package=kamae.__name__)
class ImputeLayer(BaseLayer):
    """
    Performs imputation on the input.

    Where the input data is equal to the specified mask value, this layer will replace
    the data with the impute value calculated at preprocessing time.

    The impute value is either the mean or median and is computed while ignoring rows
    in the data which are equal to the mask value or are null.

    This is a backend-agnostic layer that works with TensorFlow, JAX, and PyTorch.
    """

    def __init__(
        self,
        impute_value: Union[float, str, int],
        mask_value: Union[float, str, int],
        name: Optional[str] = None,
        input_dtype: Optional[str] = None,
        output_dtype: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise the ImputeLayer layer.

        :param impute_value: The value to use for imputation.
        :param mask_value: Value which should be replaced by the
        impute value at inference.
        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.impute_value = impute_value
        self.mask_value = mask_value
        if not isinstance(self.mask_value, type(self.impute_value)):
            raise ValueError(
                "The mask value and impute value must be of the same type."
            )

    @property
    def compatible_dtypes(self) -> Optional[List[str]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return None

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Performs imputation on the input tensor(s). It imputes over values which
        are equal to the mask_value.

        Decorated with `@enforce_single_tensor_input` to ensure that
        the input is a single tensor. Raises an error if multiple tensors are passed
        in as an iterable.

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

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the ImputeLayer layer.
        Used for saving and loading from a model.
        Specifically adds additional parameters to the base configuration.
        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "impute_value": self.impute_value,
                "mask_value": self.mask_value,
            }
        )
        return config
