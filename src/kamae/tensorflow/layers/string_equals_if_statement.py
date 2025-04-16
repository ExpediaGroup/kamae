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

import tensorflow as tf

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import allow_single_or_multiple_tensor_input

from .base import BaseLayer


# TODO: Deprecate this in favor of IfStatementLayer in next major release.
@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class StringEqualsIfStatementLayer(BaseLayer):
    """
    Performs a string if equals statement on the input tensor,
    returning a tensor of the same shape as the input tensor.

    The value to compare must be a string. We will cast the input tensor to a string
    if it is not already a string. This could cause unexpected behaviour if the input
    tensor is not a string.

    If the condition is true, the result is the result_if_true value.
    If the condition is false, the result is the result_if_false value.

    If any of [value_to_compare, result_if_true, result_if_false] are None, we assume
    they are passed in as inputs to the layer in the above order. If all of them are
    not None, then inputs is expected to be a tensor.
    """

    def __init__(
        self,
        value_to_compare: str = None,
        result_if_true: str = None,
        result_if_false: str = None,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        **kwargs,
    ) -> None:
        """
        Initialises the StringIfEqualStatement layer.

        :param value_to_compare: String value to compare the input tensor to.
        If None, we assume it is passed in as an input to the layer.
        :param result_if_true: String value to return if the condition is true.
        If None, we assume it is passed in as an input to the layer.
        :param result_if_false: String value to return if the condition is false.
        If None, we assume it is passed in as an input to the layer.
        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.value_to_compare = value_to_compare
        self.result_if_true = result_if_true
        self.result_if_false = result_if_false

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [tf.string]

    def _construct_input_tensors(self, inputs: List[Tensor]) -> List[Tensor]:
        """
        Constructs the input tensors for the layer in the case where all the optional
        parameters are not specified. We need to run through the provided inputs and
        either select an input or the specified parameter.

        Specifically for this layer, we assume the inputs are in the following order:
        [input_tensor, value_to_compare, result_if_true, result_if_false]

        Any but the input tensor can be None.

        :param inputs: List of input tensors.
        :returns: List of input tensors potentially containing constant tensors for the
        optional parameters.
        """
        optional_params = [
            self.value_to_compare,
            self.result_if_true,
            self.result_if_false,
        ]
        # Setup the inputs. Keep a counter to know how many tensors from inputs have
        # been used.
        input_col_counter = 1
        # First input is always the input tensor
        multiple_inputs = [inputs[0]]
        for param in optional_params:
            if param is None:
                # If the param is None, we assume it is an input tensor at the next
                # index
                multiple_inputs.append(inputs[input_col_counter])
                input_col_counter += 1
            else:
                # Otherwise, we create a constant tensor for the parameter
                # and do not increment the counter.
                multiple_inputs.append(tf.constant(param, dtype=tf.string))
        return multiple_inputs

    @allow_single_or_multiple_tensor_input
    def _call(self, inputs: Union[Tensor, Iterable[Tensor]], **kwargs) -> Tensor:
        """
        Performs the string if equals statement on the inputs. If the inputs are a
        tensor, we assume that the value_to_compare, result_if_true, and
        result_if_false are provided. If the inputs are not a tensor, we assume any
        not provided are provided as inputs to the layer.

        Decorated with `@allow_single_or_multiple_tensor_input` to ensure that the input
        is either a single tensor or an iterable of tensors. Returns this result as a
        list of tensors for easier use here.

        :param inputs: Tensor or iterable of tensors.
        :returns: Tensor after computing the string if equal statement.
        """
        if len(inputs) == 1:
            # If the input is a tensor, we assume that the value_to_compare,
            # result_if_true, and result_if_false are provided
            if any(
                [
                    v is None
                    for v in [
                        self.value_to_compare,
                        self.result_if_true,
                        self.result_if_false,
                    ]
                ]
            ):
                raise ValueError(
                    "If inputs is a tensor, value_to_compare, result_if_true, and "
                    "result_if_false must be specified."
                )
            string_inputs = (
                tf.strings.as_string(inputs[0])
                if inputs[0].dtype != tf.string
                else inputs[0]
            )
            cond = tf.where(
                string_inputs == self.value_to_compare,
                tf.constant(self.result_if_true, dtype=tf.string),
                tf.constant(self.result_if_false, dtype=tf.string),
            )
            return cond
        else:
            # If the input is a list, we assume that the value_to_compare,
            # result_if_true, and result_if_false are potentially provided in the inputs
            string_inputs = [
                tf.strings.as_string(i) if i.dtype != tf.string else i for i in inputs
            ]
            input_tensors = self._construct_input_tensors(string_inputs)
            cond = tf.where(
                input_tensors[0] == input_tensors[1],
                input_tensors[2],
                input_tensors[3],
            )
            return cond

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the StringEqualsIfStatement layer.
        Used for saving and loading from a model.

        Specifically adds the following to the config dictionary:
        - value_to_compare
        - result_if_true
        - result_if_false

        :returns: Dictionary configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "value_to_compare": self.value_to_compare,
                "result_if_true": self.result_if_true,
                "result_if_false": self.result_if_false,
            }
        )
        return config
