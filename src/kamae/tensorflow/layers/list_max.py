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

from typing import Any, Dict, Iterable, List, Optional

import tensorflow as tf

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import allow_single_or_multiple_tensor_input, get_top_n

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class ListMaxLayer(BaseLayer):
    """
    Calculate the max across the axis dimension.
    - If one tensor is passed, the transformer calculates the max of the tensor
    based on all the items in the given axis dimension.
    - If inputCols is set, the transformer calculates the max of the first tensor
    based on second tensor's topN items in the same given axis dimension.

    By using the topN items to calculate the statistics, we can better approximate
    the real statistics in production. It is suggested to use a large enough topN to
    get a good approximation of the statistics, and an important feature to sort on,
    such as item's past production.

    Example: calculate the max price in the same query, based only on the top N
    items sorted by descending production.
    """

    def __init__(
        self,
        name: str,
        input_dtype: str = None,
        output_dtype: str = None,
        top_n: int = None,
        sort_order: str = "asc",
        min_filter_value: float = None,
        nan_fill_value: float = 0.0,
        axis: int = 1,
        **kwargs,
    ):
        """
        Initializes the Listwise Max layer.

        WARNING: The code is fully tested for axis=1 only. Further testing is needed.

        WARNING: The code can be affected by the value of the padding items. Always
        make sure to filter out the padding items value with min_filter_value.

        :param name: Name of the layer, defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param top_n: The number of top items to consider when calculating the max.
        :param sort_order: The order to sort the second tensor by. Defaults to `asc`.
        :param min_filter_value: The minimum filter value to ignore values during
        calculation. Defaults to None (no filter).
        :param nan_fill_value: The value to fill NaNs results with. Defaults to 0.
        :param axis: The axis to calculate the statistics across. Defaults to 1.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.top_n = top_n
        self.sort_order = sort_order
        self.min_filter_value = min_filter_value
        self.nan_fill_value = nan_fill_value
        self.axis = axis

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
            tf.uint8,
            tf.int8,
            tf.uint16,
            tf.int16,
            tf.int32,
            tf.int64,
            tf.complex64,
            tf.complex128,
        ]

    @allow_single_or_multiple_tensor_input
    def _call(self, inputs: Iterable[Tensor], **kwargs) -> Tensor:
        """
        Calculate the listwise max, optionally sorting and
        filtering based on the second input tensor.

        :param inputs: The iterable tensor for the feature.
        :returns: Thew new tensor result column.
        """
        val_tensor = inputs[0]
        output_shape = tf.shape(val_tensor)

        with_sort = True if len(inputs) == 2 else False
        sort_tensor = inputs[1] if with_sort else None

        if with_sort and self.top_n is None:
            raise ValueError("topN must be specified when using a sort column.")

        if with_sort:
            # Get the values corresponding to the top N item in the sort tensor
            filtered_tensor = get_top_n(
                val_tensor=val_tensor,
                axis=self.axis,
                sort_tensor=sort_tensor,
                sort_order=self.sort_order,
                top_n=self.top_n,
            )
        else:
            filtered_tensor = val_tensor

        # Apply the mask to filter out elements less than or equal to the threshold
        if self.min_filter_value is not None:
            mask = tf.greater_equal(filtered_tensor, self.min_filter_value)
            neg_inf = val_tensor.dtype.min
            cond = tf.where(mask, filtered_tensor, neg_inf)
            listwise_stat = tf.reduce_max(cond, axis=self.axis, keepdims=True)
            is_integer = listwise_stat.dtype.is_integer
            nan_val = int(self.nan_fill_value) if is_integer else self.nan_fill_value
            listwise_max = tf.where(listwise_stat != neg_inf, listwise_stat, nan_val)

        else:
            # Calculate the mean without filtering
            listwise_max = tf.reduce_max(filtered_tensor, axis=self.axis, keepdims=True)

        # Broadcast the stat to each item in the list
        # WARNING: If filter creates empty items list, the result will be NaN
        listwise_max = tf.broadcast_to(listwise_max, output_shape)

        return listwise_max

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the layer.
        Used for saving and loading from a model.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "top_n": self.top_n,
                "sort_order": self.sort_order,
                "min_filter_value": self.min_filter_value,
                "nan_fill_value": self.nan_fill_value,
                "axis": self.axis,
            }
        )
        return config
