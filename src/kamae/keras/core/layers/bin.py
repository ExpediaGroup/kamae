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

from keras import ops

from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input
from kamae.params import _REQUIRED, ParamSpec
from kamae.utils import get_condition_operator


class BinLayer(BaseLayer):
    """
    Performs a binning operation on a given input tensor.

    The binning operation is performed by comparing the input tensor to a list of
    values using a list of operators. The bin label corresponding to the first
    condition that evaluates to True is returned.

    If no conditions evaluate to True, the default label is returned.
    """

    jit_compatible = True

    _compatible_dtypes = [
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
    _params = {
        "condition_operators": ParamSpec(
            default=_REQUIRED,
            doc="List of operators to use in the if statement (eq, neq, lt, leq, gt, geq)",
        ),
        "bin_values": ParamSpec(
            default=_REQUIRED,
            doc="List of values to compare the input tensor to",
        ),
        "bin_labels": ParamSpec(
            default=_REQUIRED,
            doc="List of labels to use for each bin",
        ),
        "default_label": ParamSpec(
            default=_REQUIRED,
            doc="Label to use if none of the conditions are met",
        ),
    }

    def _post_init(self):
        if len(self.condition_operators) != len(self.bin_labels) or len(
            self.condition_operators
        ) != len(self.bin_values):
            raise ValueError(
                f"condition_operators, bin_labels and bin_values must be the same "
                f"length. Got lengths: {len(self.condition_operators)}, {len(self.bin_labels)}, "
                f"{len(self.bin_values)}"
            )

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Performs a binning operation on a given input tensor.

        Creates a tensor of the same shape as the input tensor, where each
        element is the label of the bin that the corresponding element in the input
        tensor belongs to. The bin labels are determined by successively applying
        the condition operators to the input tensor, and returning the label of the
        first bin that the element belongs to.


        :param inputs: Tensor to perform the binning operation on.
        :returns: The binned input tensor.
        """
        cond_op_fns = [get_condition_operator(op) for op in self.condition_operators]

        # Build default output tensor
        outputs = ops.convert_to_tensor(self.default_label)

        # Loop through the conditions.
        # Reverse the list of conditions so that we start from the last condition
        # and work backwards. This ensures that the first condition that is met
        # is the one that is used.
        conds = zip(cond_op_fns[::-1], self.bin_values[::-1], self.bin_labels[::-1])

        for cond_op, value, label in conds:
            # Ensure that the inputs and value are compatible dtypes
            cast_input, cast_value = self._force_cast_to_compatible_numeric_type(
                inputs, value
            )
            outputs = ops.where(
                cond_op(
                    cast_input,
                    cast_value,
                ),
                ops.convert_to_tensor(label),
                outputs,
            )

        return outputs
