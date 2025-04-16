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

import tensorflow as tf

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import enforce_single_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class ImputeLayer(BaseLayer):
    """
    Performs imputation on the input.
    Where the input data is equal to the specified mask value, this layer will replace
    the data with the impute value calculated at preprocessing time.
    The impute value is either the mean or median and is computed while ignoring rows
    in the data which are equal to the mask value or are null.
    """

    def __init__(
        self,
        impute_value: Union[float, str, int],
        mask_value: Union[float, str, int],
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        **kwargs,
    ) -> None:
        """
        Initialise the ImputeLayer layer.
        :param impute_value: The value to use for imputation.
        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param mask_value: Value which should be replaced by the
        impute value at inference.
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
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return None

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> Tensor:
        """
        Performs imputation on the input tensor(s) by calling the keras
        ImputeLayer layer. It imputes over values which are equal to the
        mask_value.
        Decorated with `@enforce_single_tensor_input` to ensure that
        the input is a single tensor. Raises an error if multiple tensors are passed
        in as an iterable.
        :param inputs: Input tensor to perform the imputation on.
        :returns: The input tensor with the imputation applied.
        """
        if inputs.dtype.is_floating or inputs.dtype.is_integer:
            inputs, mask = self._force_cast_to_compatible_numeric_type(
                inputs, self.mask_value
            )
            inputs, impute_value = self._force_cast_to_compatible_numeric_type(
                inputs, self.impute_value
            )
        else:
            mask = self._cast(tf.constant(self.mask_value), inputs.dtype.name)
            impute_value = self._cast(tf.constant(self.impute_value), inputs.dtype.name)

        mask = tf.equal(inputs, mask)
        imputed_outputs = tf.where(
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
