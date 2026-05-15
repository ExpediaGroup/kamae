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

import tensorflow as tf

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input
from kamae.keras.tensorflow.utils.transform_utils import map_fn_w_axis
from kamae.params import ParamSpec


class OrdinalArrayEncodeLayer(BaseLayer):
    """
    Transformer that encodes an array of strings into an array of integers.

    The transformer will map each unique string in the array to an integer,
    according to the order in which they appear in the array. It will also
    ignore the pad value if specified.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = ["string"]
    _params = {
        "pad_value": ParamSpec(
            default=None,
            doc="The value which pad the array and as a result should be ignored in the encoding process.",
        ),
        "axis": ParamSpec(
            default=-1,
            doc="The axis along which to encode the array. Defaults to -1.",
        ),
    }

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Performs the ordinal encoding on the input dataset.
        Example:
         input_tensor = tf.Tensor([
            ['a', 'a', 'a', 'b', 'c', '-1', '-1', '-1'],
            ['x', 'x', 'x', 'x', 'y', 'z', '-1', '-1'],
            ]
         )

        Output: tf.Tensor([[
            [0, 0, 0, 1, 2, -1, -1, -1],
            [0, 0, 0, 0, 1, 2, -1, -1],
            ]
        )

        :param inputs: The input tensor.
        :returns: Transformed tensor.
        """

        @tf.function
        def _transform_row(input_row: Tensor) -> Tensor:
            if self.pad_value is None:
                converted_tensor = tf.unique(input_row).idx
            else:
                not_pad_mask = tf.where(
                    tf.not_equal(input_row, self.pad_value),
                    tf.constant(True),
                    tf.constant(False),
                )
                # If all values are the pad value return -1s
                if not tf.reduce_any(not_pad_mask):
                    converted_tensor = tf.fill(tf.shape(input_row), -1)
                else:
                    non_pad_values = tf.boolean_mask(input_row, not_pad_mask)
                    first_non_pad_value = non_pad_values[0]
                    replace_pad_with_first = tf.where(
                        tf.equal(input_row, self.pad_value),
                        first_non_pad_value,
                        input_row,
                    )
                    converted_tensor = tf.where(
                        not_pad_mask,
                        tf.unique(replace_pad_with_first).idx,
                        tf.constant(-1),
                    )
            return self._cast(converted_tensor, cast_dtype=tf.int32.name)

        output = map_fn_w_axis(
            elems=inputs,
            fn=_transform_row,
            axis=self.axis,
            fn_output_signature=tf.int32,
        )

        return output
