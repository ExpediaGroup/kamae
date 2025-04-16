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

from typing import List, Optional

import tensorflow as tf

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import enforce_single_tensor_input, map_fn_w_axis

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class OrdinalArrayEncodeLayer(BaseLayer):
    """
    Transformer that encodes an array of strings into an array of integers.

    The transformer will map each unique string in the array to an integer,
    according to the order in which they appear in the array. It will also
    ignore the pad value if specified.
    """

    def __init__(
        self,
        pad_value: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        axis: int = -1,
        name: str = None,
        **kwargs,
    ) -> None:
        """
        Initializes the OrdinalArrayEncodeLayer layer

        :param name: Name of the layer, defaults to `None`.
        :param pad_value: The value which pad the array and as a result should be
        ignored in the encoding process.

        :returns: None
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.pad_value = pad_value
        self.axis = axis

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [tf.string]

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> Tensor:
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

    def get_config(self):
        """
        Gets the configuration of the OrdinalArrayEncoder layer.
        Used for saving and loading from a model.

        Specifically adds the `pad_value` value to the configuration.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update({"pad_value": self.pad_value, "axis": self.axis})
        return config
