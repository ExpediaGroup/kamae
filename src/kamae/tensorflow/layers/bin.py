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
from kamae.utils import get_condition_operator

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class BinLayer(BaseLayer):
    """
    Performs a binning operation on a given input tensor.

    The binning operation is performed by comparing the input tensor to a list of
    values using a list of operators. The bin label corresponding to the first
    condition that evaluates to True is returned.

    If no conditions evaluate to True, the default label is returned.
    """

    def __init__(
        self,
        condition_operators: List[str],
        bin_values: List[float],
        bin_labels: List[Union[float, int, str]],
        default_label: Union[float, int, str],
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        **kwargs,
    ) -> None:
        """
        Initializes the BinLayer layer

        :param name: Name of the layer, defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param condition_operators: List of operators to use in the if statement.
        Can be one of:
            - "eq": Equal to
            - "neq": Not equal to
            - "lt": Less than
            - "leq": Less than or equal to
            - "gt": Greater than
            - "geq": Greater than or equal to
        :param bin_values: List of values to compare the input tensor to. Must be the
        same length as condition_operators.
        :param bin_labels: List of labels to use for each bin. Must be the same length
        as condition_operators.
        :param default_label: Label to use if none of the conditions are met.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        if len(condition_operators) != len(bin_labels) != len(bin_values):
            raise ValueError(
                f"""condition_operators, bin_labels and bin_values must be the same
                length. Got lengths: {len(condition_operators)}, {len(bin_labels)},
                {len(bin_values)}"""
            )
        self.condition_operators = condition_operators
        self.bin_values = bin_values
        self.bin_labels = bin_labels
        self.default_label = default_label

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [
            tf.bfloat16,
            tf.float16,
            tf.float32,
            tf.float64,
            tf.int8,
            tf.uint8,
            tf.int16,
            tf.uint16,
            tf.int32,
            tf.uint32,
            tf.int64,
            tf.uint64,
        ]

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> Tensor:
        """
        Performs a binning operation on a given input tensor.

        Creates a string tensor of the same shape as the input tensor, where each
        element is the label of the bin that the corresponding element in the input
        tensor belongs to. The bin labels are determined by successively applying
        the condition operators to the input tensor, and returning the label of the
        first bin that the element belongs to.

        Decorated with `@enforce_single_tensor_input` to ensure that the input
        is a single tensor. Raises an error if multiple tensors are passed
        in as an iterable.

        :param inputs: Tensor to perform the binning operation on.
        :returns: The binned input tensor.
        """
        cond_op_fns = [get_condition_operator(op) for op in self.condition_operators]

        # Build default output tensor
        outputs = tf.constant(self.default_label)

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
            outputs = tf.where(
                cond_op(
                    cast_input,
                    cast_value,
                ),
                tf.constant(label),
                outputs,
            )

        return outputs

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the Bin layer.
        Used for saving and loading from a model.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "condition_operators": self.condition_operators,
                "bin_values": self.bin_values,
                "bin_labels": self.bin_labels,
                "default_label": self.default_label,
            }
        )
        return config
