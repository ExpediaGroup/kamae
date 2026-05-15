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

from typing import Any, Iterable

import keras
import tensorflow as tf

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import allow_single_or_multiple_tensor_input
from kamae.keras.tensorflow.utils.list_utils import get_top_n
from kamae.params.shared_specs import LISTWISE_FILTER_PARAMS, LISTWISE_PARAMS


class ListStdDevLayer(BaseLayer):
    """
    Calculate the average across the axis dimension.
    - If one tensor is passed, the transformer calculates the average of the tensor
    based on all the items in the given axis dimension.
    - If inputCols is set, the transformer calculates the average of the first tensor
    based on second tensor's topN items in the same given axis dimension.

    By using the topN items to calculate the statistics, we can better approximate
    the real statistics in production. It is suggested to use a large enough topN to
    get a good approximation of the statistics, and an important feature to sort on,
    such as item's past production.

    Example: calculate the average price in the same query, based only on the top N
    items sorted by descending production.
    """

    supported_backends = TENSORFLOW_ONLY
    jit_compatible = True

    _compatible_dtypes = [
        "bfloat16",
        "float16",
        "float32",
        "float64",
    ]
    _params = {
        **{k: v for k, v in LISTWISE_PARAMS.items() if k != "queryIdCol"},
        **LISTWISE_FILTER_PARAMS,
    }

    @allow_single_or_multiple_tensor_input
    def _call(self, inputs: Iterable[Tensor], **kwargs: Any) -> Tensor:
        """
        Calculate the listwise average, optionally sorting and
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

        # Apply the mask to filter out elements less than or equal to the threshold
        if self.min_filter_value is not None:
            mask = tf.greater_equal(filtered_tensor, self.min_filter_value)
            nan_tensor = tf.constant(float("nan"), dtype=val_tensor.dtype)
            filtered_tensor = tf.where(mask, filtered_tensor, nan_tensor)
            mask = tf.math.is_finite(filtered_tensor)
            numerator = tf.reduce_sum(
                tf.where(mask, filtered_tensor, tf.zeros_like(filtered_tensor)),
                axis=self.axis,
                keepdims=True,
            )
            denominator = tf.reduce_sum(
                tf.cast(mask, dtype=numerator.dtype),
                axis=self.axis,
                keepdims=True,
            )
            listwise_mean = tf.truediv(numerator, denominator)

        else:
            # Calculate the mean without filtering
            listwise_mean = tf.reduce_mean(
                filtered_tensor,
                axis=self.axis,
                keepdims=True,
            )

        # Calculate the squared differences from the mean
        squared_diff = tf.square(filtered_tensor - listwise_mean)

        # Calculate the sample variance by dividing the sum of squared diff by (N - 1)
        mask = tf.math.is_finite(squared_diff)
        listwise_sum = tf.reduce_sum(
            tf.where(mask, squared_diff, tf.zeros_like(squared_diff)),
            axis=self.axis,
            keepdims=True,
        )
        listwise_count = tf.reduce_sum(
            tf.cast(mask, dtype=listwise_sum.dtype),
            axis=self.axis,
            keepdims=True,
        )
        listwise_variance = tf.math.divide_no_nan(listwise_sum, (listwise_count - 1))
        listwise_stddev = tf.sqrt(listwise_variance)

        # Fill nan
        is_integer = "int" in keras.backend.standardize_dtype(listwise_stddev.dtype)
        nan_val = int(self.nan_fill_value) if is_integer else self.nan_fill_value
        listwise_stddev = tf.where(
            tf.math.is_nan(listwise_stddev),
            tf.constant(nan_val, dtype=listwise_mean.dtype),
            listwise_stddev,
        )

        # Broadcast the stat to each item in the list
        # WARNING: If filter creates empty items list, the result will be NaN
        listwise_stddev = tf.broadcast_to(listwise_stddev, output_shape)

        return listwise_stddev
