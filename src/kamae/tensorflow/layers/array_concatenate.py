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
from kamae.tensorflow.utils import enforce_multiple_tensor_input, reshape_to_equal_rank

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(kamae.__name__)
class ArrayConcatenateLayer(BaseLayer):
    """
    Performs a concatenation of the input tensors.
    """

    def __init__(
        self,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        axis: int = -1,
        auto_broadcast: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialises the ArrayConcatenateLayer layer.

        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param axis: Axis to concatenate on. Defaults to -1.
        :param auto_broadcast: If `True`, will broadcast the input tensors to the
        biggest rank before concatenating. Defaults to `False`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        if auto_broadcast and axis != -1:
            raise ValueError("auto_broadcast is only supported for axis=-1")
        self.axis = axis
        self.auto_broadcast = auto_broadcast

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer. Returns `None` as the
        compatible dtypes are not restricted.

        :returns: The compatible dtypes of the layer.
        """
        return None

    @enforce_multiple_tensor_input
    def _call(self, inputs: Iterable[Tensor], **kwargs) -> Tensor:
        """
        Concatenates the input tensors along the specified axis.
        If auto_broadcast is set to True, the tensors are broadcasted to the
        same rank before concatenating.

        Decorated with `@enforce_multiple_tensor_input` to ensure that the input
        is an iterable of tensors. Raises an error if a single tensor is passed
        in.

        :param inputs: Iterable of tensors to concatenate.
        :returns: Concatenated tensor.
        """
        if self.auto_broadcast:
            # Determine the maximum rank statically
            max_rank = max([len(tensor.shape) for tensor in inputs])

            # Reshape all tensors to the same rank, so to calculate later the max_shape
            # WARNING: It assumes that order of inputs and reshaped_inputs is the same!
            reshaped_inputs = reshape_to_equal_rank(inputs)

            # Check the maximum static shape (i.e. with None being the biggest number)
            # except the last one to concat. Here we use the static tensor.shape.
            max_static_shape = []
            for i in range(max_rank - 1):
                shapes = [x.shape[i] for x in reshaped_inputs]
                if None in shapes:
                    max_static_shape.append(None)
                else:
                    max_static_shape.append(max(shapes))

            # Determine the maximum dynamic shape for each dimension, except last one
            # Since shapes can be dynamic (None), we need to use tf.shape
            max_dynamic_shape = []
            for i in range(max_rank - 1):
                shapes = [tf.shape(x)[i] for x in reshaped_inputs]
                max_dynamic_shape.append(tf.reduce_max(shapes))

            # Broadcast tensors to the maximum dynamic shape if the static is different
            # WARNING: It assumes that when the static shapes of two tensors are None
            # at a given rank, the dynamic shapes are the same.
            for idx, x in enumerate(reshaped_inputs):
                x_static_shape = x.shape[:-1]
                if x_static_shape != max_static_shape:
                    last_dim = x.shape[-1]
                    broadcast_shape = tf.concat([max_dynamic_shape, [last_dim]], axis=0)
                    broadcasted_x = tf.broadcast_to(x, broadcast_shape)
                    reshaped_inputs[idx] = broadcasted_x
            inputs = reshaped_inputs

        return tf.concat(inputs, axis=self.axis)

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the VectorConcat layer.
        Used for saving and loading from a model.

        Specifically, adds the `axis` to the config.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
                "auto_broadcast": self.auto_broadcast,
            }
        )
        return config
