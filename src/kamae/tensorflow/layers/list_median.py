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
class ListMedianLayer(BaseLayer):
    """
    Calculate the median across the axis dimension.
    - If one tensor is passed, the transformer calculates the median of the tensor
    based on all the items in the given axis dimension.
    - If inputCols is set, the transformer calculates the median of the first tensor
    based on second tensor's topN items in the same given axis dimension.

    By using the topN items to calculate the statistics, we can better approximate
    the real statistics in production. It is suggested to use a large enough topN to
    get a good approximation of the statistics, and an important feature to sort on,
    such as item's past production.

    Example: calculate the median price in the same query, based only on the top N
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
        Initializes the Listwise Median layer.

        WARNING: The code is fully tested for axis=1 only. Further testing is needed.

        WARNING: The code can be affected by the value of the padding items. Always
        make sure to filter out the padding items value with min_filter_value.

        :param name: Name of the layer, defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param top_n: The number of top items to consider when calculating the median.
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

    def sort_with_nans_last(self, tensor: Tensor):
        """
        Sorts a tensor while placing NaN values at the end along the specified axis.

        :param tensor: The input tensor.
        :param axis: The axis along which to sort.
        :returns: The sorted tensor with NaN values placed at the end.
        """
        # Replace NaNs with a very large value to move them to the end
        masked_tensor = tf.where(tf.math.is_nan(tensor), tensor.dtype.max, tensor)

        # Sort the tensor along the specified axis
        sorted_masked_tensor = tf.sort(masked_tensor, axis=self.axis)

        # Replace the very large values back with NaN after sorting
        sorted_masked_tensor = tf.where(
            tf.equal(sorted_masked_tensor, tensor.dtype.max),
            tf.constant(float("nan"), dtype=tensor.dtype),
            sorted_masked_tensor,
        )

        return sorted_masked_tensor

    @allow_single_or_multiple_tensor_input
    def _call(self, inputs: Iterable[Tensor], **kwargs) -> Tensor:
        """
        Calculate the listwise median, optionally sorting and
        filtering based on the second input tensor.

        :param inputs: The iterable tensor for the feature.
        :returns: The new tensor result column.
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

        # Assign nan to elements less than or equal to the threshold
        if self.min_filter_value is not None:
            filtered_tensor = tf.where(
                filtered_tensor >= self.min_filter_value,
                filtered_tensor,
                tf.constant(float("nan"), dtype=val_tensor.dtype),
            )
        else:
            filtered_tensor = filtered_tensor

        # Get the number of non-nan values
        num_valid_values = tf.reduce_sum(
            tf.cast(tf.math.is_finite(filtered_tensor), tf.int32), axis=self.axis
        )

        # Sort the values along the list dimension
        sorted_filtered_tensor = self.sort_with_nans_last(filtered_tensor)

        # Calculate the indices of the median values
        lower_index = (num_valid_values - 1) // 2
        upper_index = tf.minimum(lower_index + 1, num_valid_values - 1)

        # Gather the median values for each feature
        batch_size = tf.shape(filtered_tensor)[0]
        batch_indices = tf.range(batch_size)[:, tf.newaxis, tf.newaxis]
        lower_indices = tf.concat([batch_indices, lower_index[:, tf.newaxis]], axis=-1)
        lower_medians = tf.gather_nd(sorted_filtered_tensor, lower_indices)
        upper_indices = tf.concat([batch_indices, upper_index[:, tf.newaxis]], axis=-1)
        upper_medians = tf.gather_nd(sorted_filtered_tensor, upper_indices)

        # Calculate the average of lower and upper medians for even cases
        listwise_median = tf.where(
            tf.math.mod(num_valid_values[:, tf.newaxis], 2) == 0,
            (lower_medians + upper_medians) / 2.0,
            lower_medians,
        )

        # Fill nan
        is_integer = listwise_median.dtype.is_integer
        nan_val = int(self.nan_fill_value) if is_integer else self.nan_fill_value
        listwise_median = tf.where(
            tf.math.is_nan(listwise_median),
            tf.constant(nan_val, dtype=listwise_median.dtype),
            listwise_median,
        )

        # Broadcast the stat to each item in the list
        # WARNING: If filter creates empty items list, the result will be NaN
        listwise_median = tf.broadcast_to(listwise_median, output_shape)

        return listwise_median

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
