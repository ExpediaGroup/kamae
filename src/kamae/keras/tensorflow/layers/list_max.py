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

import tensorflow as tf

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import allow_single_or_multiple_tensor_input
from kamae.keras.tensorflow.utils.list_utils import get_top_n, segmented_operation
from kamae.keras.tensorflow.utils.transform_utils import map_fn_w_axis
from kamae.params.shared_specs import (
    LISTWISE_FILTER_PARAMS,
    LISTWISE_PARAMS,
    LISTWISE_SEGMENT_PARAMS,
)


class ListMaxLayer(BaseLayer):
    """
    Calculate the max across the axis dimension.
    - If one tensor is passed, the transformer calculates the max of the tensor
    based on all the items in the given axis dimension.
    - If inputCols is set,
        - If with_segment = True: the layer calculates the maximum of the first tensor
        segmented by values of the second tensor.
        Example: calculate the maximum price of hotels within star ratings

        - If with_segment = False: the layer calculates the maximum of the first tensor
    based on second tensor's topN items in the same given axis dimension.


    By using the topN items to calculate the statistics, we can better approximate
    the real statistics in production. It is suggested to use a large enough topN to
    get a good approximation of the statistics, and an important feature to sort on,
    such as item's past production.

    Example: calculate the maximum price in the same query, based only on the top N
    items sorted by descending production.
    """

    supported_backends = TENSORFLOW_ONLY
    jit_compatible = True

    _compatible_dtypes = [
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "string",
    ]
    _params = {
        **{k: v for k, v in LISTWISE_PARAMS.items() if k != "queryIdCol"},
        **LISTWISE_FILTER_PARAMS,
        **LISTWISE_SEGMENT_PARAMS,
    }

    @allow_single_or_multiple_tensor_input
    def _call(self, inputs: Iterable[Tensor], **kwargs: Any) -> Tensor:
        """
        Calculate the listwise max, optionally sorting and
        filtering based on the second input tensor, or segmenting
        based on the second input tensor. Behaviour is set by with_segment.

        :param inputs: The iterable tensor for the feature.
        :returns: The new tensor result column.
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
        else:
            if self.with_segment:
                raise ValueError("with_segment set to True, expected two inputs.")

        # Apply the mask to filter out elements less than or equal to the threshold
        if self.min_filter_value is not None:
            mask = tf.greater_equal(val_tensor, self.min_filter_value)
            neg_inf = val_tensor.dtype.min
            val_tensor = tf.where(mask, val_tensor, neg_inf)
        else:
            val_tensor = val_tensor

        # Apply segmented calculation
        if self.with_segment:
            listwise_max = map_fn_w_axis(
                elems=[val_tensor, segment_tensor],
                fn=lambda x: segmented_operation(x, tf.math.unsorted_segment_max),
                axis=self.axis,
                fn_output_signature=tf.TensorSpec(
                    shape=val_tensor.shape[self.axis :], dtype=val_tensor.dtype
                ),
            )
            listwise_max = tf.ensure_shape(listwise_max, val_tensor.shape)
        else:
            listwise_max = tf.reduce_max(val_tensor, axis=self.axis, keepdims=True)
            listwise_max = tf.broadcast_to(listwise_max, output_shape)

        if self.min_filter_value is not None:
            fill_val = tf.constant(self.nan_fill_value, dtype=listwise_max.dtype)
            listwise_max = tf.where(listwise_max != neg_inf, listwise_max, fill_val)

        return listwise_max
