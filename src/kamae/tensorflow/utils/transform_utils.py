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

from typing import Callable, List, Optional, Union

import tensorflow as tf

from kamae.tensorflow.typing import Tensor


def map_fn_w_axis(
    elems: Union[Tensor, List[Tensor]],
    fn: Callable[[Tensor], Tensor],
    fn_output_signature: Union[tf.dtypes.DType, tf.TypeSpec],
    axis: int = -1,
    parallel_iterations: Optional[int] = None,
    swap_memory: bool = False,
    infer_shape: bool = True,
    name: Optional[str] = None,
) -> Tensor:
    """
    Applies a function to a specific axis of a tensor using `tf.map_fn`.

    Backward-compatible behavior (when `fn_output_signature` is a `tf.dtypes.DType`):
    preserves only the `axis` length when passing slices into `fn`.

    When `fn_output_signature` is a `tf.TypeSpec` (e.g. `tf.TensorSpec`), preserves
    all dimensions from `axis` onwards when passing slices into `fn`.

    :param elems: The input tensor or list of tensors.
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

    if not isinstance(fn_output_signature, (tf.dtypes.DType, tf.TypeSpec)):
        raise TypeError(
            "`fn_output_signature` must be a `tf.dtypes.DType` or `tf.TypeSpec`, "
            f"got {type(fn_output_signature).__name__}."
        )

    if isinstance(fn_output_signature, tf.TypeSpec):

        def reshape_for_map(
            tensor: Tensor, axis_pos: tf.Tensor, rank: tf.Tensor
        ) -> Tensor:
            shape = tf.shape(tensor)
            tail_shape = tf.slice(
                shape, begin=tf.stack([axis_pos]), size=tf.stack([rank - axis_pos])
            )
            return tf.reshape(
                tensor,
                tf.concat([tf.expand_dims(head_size, axis=0), tail_shape], axis=0),
            )

        if isinstance(elems, list):
            if len(elems) > 2:
                raise ValueError("Passing 3 or more tensors as input is not supported.")
            ref = elems[0]
        else:
            ref = elems

        rank = tf.rank(ref)
        axis_pos = tf.math.floormod(tf.cast(axis, dtype=rank.dtype), rank)

        ref_shape = tf.shape(ref)
        head_shape = tf.slice(ref_shape, begin=[0], size=tf.stack([axis_pos]))
        head_size = tf.reduce_prod(head_shape)

        if isinstance(elems, list):
            reshaped_input = (
                reshape_for_map(elems[0], axis_pos=axis_pos, rank=rank),
                reshape_for_map(elems[1], axis_pos=axis_pos, rank=rank),
            )
        else:
            reshaped_input = reshape_for_map(elems, axis_pos=axis_pos, rank=rank)

        output = tf.map_fn(
            fn=fn,
            elems=reshaped_input,
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory,
            infer_shape=infer_shape,
            name=name,
            fn_output_signature=fn_output_signature,
        )

        output_shape = tf.shape(output)
        output_rank = tf.rank(output)
        output_tail = tf.slice(
            output_shape, begin=[1], size=tf.stack([output_rank - 1])
        )
        return tf.reshape(output, tf.concat([head_shape, output_tail], axis=0))

    def apply_transpose_and_reshape(tensor: Tensor) -> Tensor:
        transposed = tf.transpose(tensor, perm=transpose_perm)
        reshaped = tf.reshape(transposed, tf.stack([-1, tf.shape(tensor)[axis]]))
        return reshaped

    def apply_undo_transpose_and_reshape(
        output: Tensor, transposed_shape: Tensor, identity_perm: Tensor, shift_axis: int
    ) -> Tensor:
        reshaped = tf.reshape(output, transposed_shape)
        perm = tf.roll(identity_perm, shift=shift_axis, axis=0)
        return tf.transpose(reshaped, perm=perm)

    if isinstance(elems, list):
        if len(elems) > 2:
            raise ValueError("Passing 3 or more tensors as input is not supported.")
        elems_rank = tf.rank(elems[0])
        original_shape = tf.shape(elems[0])
    else:
        elems_rank = tf.rank(elems)
        original_shape = tf.shape(elems)

    identity_perm = tf.range(start=0, limit=elems_rank)
    shift_axis = tf.math.mod(axis, elems_rank) + 1
    transpose_perm = tf.roll(identity_perm, shift=-shift_axis, axis=0)

    if isinstance(elems, list):
        reshaped_input = (
            apply_transpose_and_reshape(elems[0]),
            apply_transpose_and_reshape(elems[1]),
        )
    else:
        reshaped_input = apply_transpose_and_reshape(elems)

    output = tf.map_fn(
        fn=fn,
        elems=reshaped_input,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        infer_shape=infer_shape,
        name=name,
        fn_output_signature=fn_output_signature,
    )

    transposed_shape = tf.gather(original_shape, transpose_perm)
    return apply_undo_transpose_and_reshape(
        output, transposed_shape, identity_perm, shift_axis
    )
