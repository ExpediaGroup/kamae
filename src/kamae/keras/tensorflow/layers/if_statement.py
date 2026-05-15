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
from numbers import Number
from typing import Any, Iterable, Union

import tensorflow as tf

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import allow_single_or_multiple_tensor_input
from kamae.params import _REQUIRED, ParamSpec
from kamae.utils import get_condition_operator


class IfStatementLayer(BaseLayer):
    """
    Performs an if statement on the input tensor.

    This layer requires TensorFlow backend as it supports string operations.

    Performs an if statement on the input tensor,
    returning a tensor of the same shape as the input tensor.

    The condition operator can be one of the following:
    - "eq": Equal to
    - "neq": Not equal to
    - "lt": Less than
    - "le": Less than or equal to
    - "gt": Greater than
    - "ge": Greater than or equal to

    If the condition is true, the result is the result_if_true value.
    If the condition is false, the result is the result_if_false value.

    If any of [value_to_compare, result_if_true, result_if_false] are None, we assume
    they are passed in as inputs to the layer in the above order. If all of them are
    not None, then inputs is expected to be a tensor.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = None
    _params = {
        "condition_operator": ParamSpec(
            default=_REQUIRED,
            doc="Operator to use in the if statement (eq, neq, lt, leq, gt, geq)",
        ),
        "value_to_compare": ParamSpec(
            default=None,
            doc="Value to compare the input tensor to",
        ),
        "result_if_true": ParamSpec(
            default=None,
            doc="Value to return if the condition is true",
        ),
        "result_if_false": ParamSpec(
            default=None,
            doc="Value to return if the condition is false",
        ),
    }

    @staticmethod
    def _post_init(self):
        if (
            self.value_to_compare is not None
            and not isinstance(self.value_to_compare, Number)
            and self.condition_operator not in ["eq", "neq"]
        ):
            raise TypeError(
                """value_to_compare must be a number for condition operators
                other than eq and neq."""
            )

        if self.result_if_true is not None and self.result_if_false is not None:
            if not isinstance(self.result_if_true, type(self.result_if_false)):
                raise TypeError(
                    """If provided, result_if_true and result_if_false must be of the
                    same type."""
                )

    def _construct_input_tensors(
        self, inputs: Iterable[tf.Tensor]
    ) -> Iterable[tf.Tensor]:
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
                multiple_inputs.append(param)
        return multiple_inputs

    def _create_casted_tensor_from_tensor_or_constant(
        self, value: Union[tf.Tensor, Any]
    ) -> tf.Tensor:
        """
        Creates a tensor from a tensor or constant value.
        If the input value is not a tensor, we assume it is a constant and create a
        tensor from it. If self.input_dtype is not None, we cast the tensor to the
        specified dtype.
        """
        if not isinstance(value, tf.Tensor):
            value = tf.constant(value)
        return (
            value
            if self._input_dtype is None
            else self._cast(tf.constant(value), self._input_dtype)
        )

    @allow_single_or_multiple_tensor_input
    def _call(self, inputs: Union[Tensor, Iterable[Tensor]], **kwargs: Any) -> Tensor:
        """
        Performs the numerical if statement on the inputs. If the inputs are a tensor,
        we assume that the value_to_compare, result_if_true, and result_if_false are
        provided. If the inputs are not a tensor, we assume any not provided are
        provided as inputs to the layer.


        :param inputs: Tensor or list of tensors.
        :returns: Tensor after computing the numerical if statement.
        """
        condition_op = get_condition_operator(self.condition_operator)
        if not len(inputs) > 1:
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
            dtype_str = keras.backend.standardize_dtype(inputs[0].dtype)
            if "float" in dtype_str or "int" in dtype_str:
                inputs, value_to_compare = self._force_cast_to_compatible_numeric_type(
                    inputs[0], self.value_to_compare
                )
            else:
                inputs = inputs[0]
                value_to_compare = tf.constant(
                    self.value_to_compare, dtype=inputs.dtype
                )
            cond = tf.where(
                condition_op(inputs, value_to_compare),
                tf.constant(self.result_if_true),
                tf.constant(self.result_if_false),
            )
            return cond
        else:
            # If the input is a list, we assume that the value_to_compare,
            # result_if_true, and result_if_false are potentially provided in the inputs
            input_tensors = self._construct_input_tensors(inputs)
            # Ensure the results are the casted to the input dtype if specified
            result_if_true = self._create_casted_tensor_from_tensor_or_constant(
                input_tensors[2]
            )
            result_if_false = self._create_casted_tensor_from_tensor_or_constant(
                input_tensors[3]
            )

            if isinstance(input_tensors[1], tf.Tensor):
                # If the value to compare is a tensor, we cast it to the input dtype
                inputs = input_tensors[0]
                value_to_compare = self._cast(
                    input_tensors[1],
                    cast_dtype=keras.backend.standardize_dtype(input_tensors[0].dtype),
                )
            elif "float" in keras.backend.standardize_dtype(
                input_tensors[0].dtype
            ) or "int" in keras.backend.standardize_dtype(input_tensors[0].dtype):
                # If the inputs are numeric we force cast it to a compatible dtype
                inputs, value_to_compare = self._force_cast_to_compatible_numeric_type(
                    input_tensors[0], input_tensors[1]
                )
            else:
                # The inputs are not numeric, so we just do the regular casting
                inputs = input_tensors[0]
                value_to_compare = self._cast(
                    tf.constant(input_tensors[1]),
                    keras.backend.standardize_dtype(inputs.dtype),
                )

            cond = tf.where(
                condition_op(
                    inputs,
                    value_to_compare,
                ),
                result_if_true,
                result_if_false,
            )
            return cond
