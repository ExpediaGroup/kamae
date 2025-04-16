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

from typing import Callable

import tensorflow as tf

from kamae.tensorflow.typing import Tensor


def map_fn_w_axis(
    elems: Tensor,
    fn: Callable[[Tensor], Tensor],
    fn_output_signature: tf.dtypes.DType,
    axis: int = -1,
    parallel_iterations=None,
    swap_memory=False,
    infer_shape=True,
    name=None,
) -> Tensor:
    """
    Applies a function to a specific axis of a tensor using map_fn.
    Specifically uses `tf.transpose` and `tf.reshape` to rearrange the tensor so that
    the specified axis is preserved, the tensor is 2D and thus can be used with map_fn.

    After applying map_fn, the tensor is reshaped and transposed back to the original
    shape.

    :param elems: The input tensor.
    :param fn: The function to apply to the tensor. Must take a single tensor as input
    and return a tensor.
    :param fn_output_signature: The output signature of the function.
    :param axis: The axis to apply the function to. Defaults to -1.
    :param parallel_iterations: The number of iterations to run in parallel. Defaults to
    None.
    :param swap_memory: Whether to use memory swapping. Defaults to False.
    :param infer_shape: Whether to infer the shape of the output. Defaults to True.
    :param name: The name of the operation. Defaults to None.
    """
    # Permutation tensor that does nothing/identity
    identity_perm = tf.range(start=0, limit=tf.rank(elems))
    # Mod the axis param by the rank of the tensor and add 1. To resolve the positive
    # axis value when axis is negative.
    # Create the shift axis. We will roll the identity permutation by this amount to
    # transpose the input
    shift_axis = tf.math.mod(axis, tf.rank(elems)) + 1
    # Roll by negative shift axis. For example if
    # axis=0, shift_axis=1, identity_perm=[0, 1, 2]
    # Then transpose_perm = [1, 2, 0]

    # Transpose and reshape
    transpose_perm = tf.roll(identity_perm, shift=-shift_axis, axis=0)
    transposed_input = tf.transpose(elems, perm=transpose_perm)
    reshaped_input = tf.reshape(transposed_input, tf.stack([-1, tf.shape(elems)[axis]]))

    # Apply map_fn
    output = tf.map_fn(
        fn=fn,
        elems=reshaped_input,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        infer_shape=infer_shape,
        name=name,
        fn_output_signature=fn_output_signature,
    )

    # Undo reshape and transpose
    undo_reshaped_output = tf.reshape(output, tf.shape(transposed_input))
    undo_transpose_perm = tf.roll(identity_perm, shift=shift_axis, axis=0)
    return tf.transpose(undo_reshaped_output, perm=undo_transpose_perm)
