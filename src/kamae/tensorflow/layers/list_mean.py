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
from kamae.tensorflow.utils import (
    allow_single_or_multiple_tensor_input,
    get_top_n,
    map_fn_w_axis,
)

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class ListMeanLayer(BaseLayer):
    """
    Calculate the mean across the axis dimension.
    - If one tensor is passed, the transformer calculates the mean of the tensor
    based on all the items in the given axis dimension.
    - If inputCols is set,
        - If with_segment = True: the layer calculates the mean of the first tensor
        segmented by values of the second tensor.
        Example: calculate the mean price of hotels within star ratings

        - If with_segment = False: the layer calculates the mean of the first tensor
    based on second tensor's topN items in the same given axis dimension.
    By using the topN items to calculate the statistics, we can better approximate
    the real statistics in production. It is suggested to use a large enough topN to
    get a good approximation of the statistics, and an important feature to sort on,
    such as item's past production.

    Example: calculate the mean price in the same query, based only on the top N
    items sorted by descending production.
    """

    def __init__(
        self,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        top_n: int = None,
        sort_order: str = "asc",
        with_segment: bool = False,
        min_filter_value: float = None,
        nan_fill_value: float = 0.0,
        axis: int = 1,
        **kwargs,
    ):
        """
        Initializes the Listwise Mean layer.

        WARNING: The code is fully tested for axis=1 only. Further testing is needed.

        WARNING: The code can be affected by the value of the padding items. Always
        make sure to filter out the padding items value with min_filter_value.

        :param name: Name of the layer, defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param top_n: The number of top items to consider when calculating the mean.
        :param sort_order: The order to sort the second tensor by. Defaults to `asc`.
        :param with_segment: Whether the second tensor should be used for segmentation (True)
        or sorting (False). Defaults to False.
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
        self.with_segment = with_segment

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
            tf.string,
        ]

    @allow_single_or_multiple_tensor_input
    def _call(self, inputs: Iterable[Tensor], **kwargs) -> Tensor:
        """
        Calculate the listwise mean, optionally sorting and
        filtering based on the second input tensor, or segmenting
        based on the second input tensor. Behaviour is set by with_segment.

        :param inputs: The iterable tensor for the feature.
        :returns: Thew new tensor result column.
        """
        val_tensor = inputs[0]
        output_shape = tf.shape(val_tensor)

        # Define use of second input
        if len(inputs) == 2:
            if self.with_segment:
                segment_tensor = inputs[1]
            else:
                sort_tensor = inputs[1]
                if self.top_n is None:
                    raise ValueError("topN must be specified when using a sort column.")
                val_tensor = get_top_n(
                    val_tensor=val_tensor,
                    axis=self.axis,
                    sort_tensor=sort_tensor,
                    sort_order=self.sort_order,
                    top_n=self.top_n,
                )

        # Apply the mask to filter out elements less than or equal to the threshold
        if self.min_filter_value is not None:
            mask = tf.greater_equal(val_tensor, self.min_filter_value)
            nan_tensor = tf.constant(float("nan"), dtype=val_tensor.dtype)
            val_tensor = tf.where(mask, val_tensor, nan_tensor)

        if self.with_segment:

            def segment_mean(values):
                unique_segments, segment_indices = tf.unique(values[1])
                num_segments = tf.size(unique_segments)
                # Cast
                flat_v = tf.cast(values[0], tf.float32)
                mask = tf.math.is_finite(flat_v)
                sum_vals = tf.math.unsorted_segment_sum(
                    tf.where(mask, flat_v, tf.zeros_like(flat_v)),
                    segment_indices,
                    num_segments,
                )

                count_vals = tf.math.unsorted_segment_sum(
                    tf.cast(mask, dtype=sum_vals.dtype), segment_indices, num_segments
                )

                gathered = tf.gather(
                    tf.math.divide_no_nan(sum_vals, count_vals), segment_indices
                )
                return tf.reshape(gathered, tf.shape(values[0]))

            listwise_mean = map_fn_w_axis(
                elems=[val_tensor, segment_tensor],
                fn=segment_mean,
                axis=self.axis,
                fn_output_signature=tf.TensorSpec(
                    shape=val_tensor.shape[self.axis], dtype=tf.float32
                ),
            )
        else:
            if self.min_filter_value is not None:
                mask = tf.math.is_finite(val_tensor)
                listwise_sum = tf.reduce_sum(
                    tf.where(mask, val_tensor, tf.zeros_like(val_tensor)),
                    axis=self.axis,
                    keepdims=True,
                )
                listwise_count = tf.reduce_sum(
                    tf.cast(mask, dtype=listwise_sum.dtype),
                    axis=self.axis,
                    keepdims=True,
                )
                listwise_mean = tf.math.divide_no_nan(listwise_sum, listwise_count)
            else:
                # Calculate the mean without filtering
                listwise_mean = tf.reduce_mean(
                    val_tensor,
                    axis=self.axis,
                    keepdims=True,
                )
            # Broadcast the stat to each item in the list
            # WARNING: If filter creates empty items list, the result will be NaN
            listwise_mean = tf.broadcast_to(listwise_mean, output_shape)

        # Fill nan
        is_integer = listwise_mean.dtype.is_integer
        nan_val = int(self.nan_fill_value) if is_integer else self.nan_fill_value
        listwise_mean = tf.where(
            tf.math.is_nan(tf.cast(listwise_mean, tf.float32)),
            tf.constant(nan_val, dtype=listwise_mean.dtype),
            listwise_mean,
        )

        return listwise_mean

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
                "with_segment": self.with_segment,
            }
        )
        return config
