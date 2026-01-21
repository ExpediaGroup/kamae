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
from typing import Any, Callable, List, Union

import numpy as np
import tensorflow as tf

from kamae.tensorflow.typing import Tensor


def get_top_n(
    val_tensor: Tensor,
    axis: int,
    sort_tensor: Tensor,
    top_n: int,
    sort_order: str = "asc",
) -> Tensor:
    """
    Get the top N items from the value tensor based on their position in
    the sort tensor, ordered by the sort order ('asc' or 'desc').

    :param val_tensor: Value tensor.
    :param axis: Axis to get the top N items.
    :param sort_tensor: Sort tensor.
    :param top_n: Number of top values to consider.
    :param sort_order: Order to sort the values by. Default is "asc".
    :returns: Tensor of the top N items
    """

    # If K is less than the number of items at real time,
    # replace K with the number of items in the list
    top_n = tf.minimum(top_n, tf.shape(sort_tensor)[axis])

    # Define sort direction
    sort_tensor_with_order = None
    if sort_order == "desc":
        sort_tensor_with_order = sort_tensor
    elif sort_order == "asc":
        sort_tensor_with_order = -sort_tensor
    else:
        ValueError(f"Invalid sort_order: {sort_order}")

    # If value of shape at position (axis + 1) is equal to 1, squeeze this dimension,
    # otherwise the top_k would complain about the shape mismatch
    # If we apply squeeze without axis, the inference when batch_size=1 would fail
    if len(sort_tensor_with_order.shape) > axis + 1:
        if sort_tensor_with_order.shape[axis + 1] == 1:
            sort_tensor_with_order = tf.squeeze(sort_tensor_with_order, axis=axis + 1)

    # Get the indices of the top N items, using the sort tensor
    _, sorted_indices = tf.math.top_k(sort_tensor_with_order, k=top_n, sorted=True)

    # Gather elements from the value tensor using the top-k indices
    return tf.gather(
        val_tensor,
        sorted_indices,
        batch_dims=axis,
        axis=axis,
    )


def listify_tensors(x: Union[tf.Tensor, np.ndarray, List[Any]]) -> List[Any]:
    """
    Converts any tensors or numpy arrays to lists for config serialization.

    :param x: The input tensor or numpy array.
    :returns: The input as a list.
    """
    if tf.is_tensor(x):
        x = x.numpy()
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return x


def segmented_operation(values: List[Tensor], fn: Callable) -> Tensor:
    """
    Function for applying an operation to one tensor, segmented by the values of another.

    Primarily intended for use with Tensorflow's unsorted segment operations, which require flattened inputs.
    e.g. tf.math.unsorted_segment_min
    :param values: List of two tensors, the first containing values, the second containing segment identifiers.
    :param fn: Function to apply an operation taking the two tensors as inputs.

    :returns: Single tensor in shape of the first of the original inputs.
    """
    segment_ids = values[1]

    # Segment ids are expected to be 1D. In some pipelines they arrive with a trailing
    # "feature" dimension, e.g. (items, 1) or (items, feature). When feature > 1 we
    # only support the common case where the segment ids are duplicated across the
    # feature dimension (so we can safely take the first column).
    if segment_ids.shape.rank is not None:
        if segment_ids.shape.rank > 1:
            if segment_ids.shape[-1] == 1:
                segment_ids = tf.squeeze(segment_ids, axis=-1)
            else:
                first = segment_ids[..., 0]
                tf.debugging.assert_equal(
                    segment_ids,
                    tf.broadcast_to(
                        tf.expand_dims(first, axis=-1), tf.shape(segment_ids)
                    ),
                    message=(
                        "Segment identifiers must be 1D, or duplicated across the trailing "
                        "feature dimension."
                    ),
                )
                segment_ids = first
    else:

        def _normalize_segment_ids() -> Tensor:
            rank = tf.rank(segment_ids)
            feature_dim = tf.shape(segment_ids)[-1]

            def _squeeze() -> Tensor:
                return tf.squeeze(segment_ids, axis=-1)

            def _take_first() -> Tensor:
                first = segment_ids[..., 0]
                tf.debugging.assert_equal(
                    segment_ids,
                    tf.broadcast_to(
                        tf.expand_dims(first, axis=-1), tf.shape(segment_ids)
                    ),
                    message=(
                        "Segment identifiers must be 1D, or duplicated across the trailing "
                        "feature dimension."
                    ),
                )
                return first

            return tf.cond(
                tf.equal(rank, 1),
                lambda: segment_ids,
                lambda: tf.cond(tf.equal(feature_dim, 1), _squeeze, _take_first),
            )

        segment_ids = _normalize_segment_ids()
    tf.debugging.assert_rank(
        segment_ids, 1, message="Segment identifiers must be a 1D tensor."
    )

    # Get segment indices and their IDs
    unique_segments, segment_indices = tf.unique(segment_ids)
    num_segments = tf.size(unique_segments)

    # Apply segment function
    vals = fn(values[0], segment_indices, num_segments)

    # Reshape and return
    gathered = tf.gather(vals, segment_indices)
    result = tf.reshape(gathered, tf.shape(values[0]))

    return result
