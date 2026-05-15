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

from keras import ops

from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_multiple_tensor_input
from kamae.keras.core.utils.shape_utils import reshape_to_equal_rank
from kamae.params import ParamSpec


class ArrayConcatenateLayer(BaseLayer):
    """
    Performs a concatenation of the input tensors.
    """

    jit_compatible = True

    _compatible_dtypes = None
    _params = {
        "axis": ParamSpec(default=-1, doc="Axis to concatenate on"),
        "auto_broadcast": ParamSpec(
            default=False,
            doc="If True, broadcast input tensors to biggest rank before concatenating",
        ),
    }

    def _post_init(self):
        if self.auto_broadcast and self.axis != -1:
            raise ValueError("auto_broadcast is only supported for axis=-1")

    @enforce_multiple_tensor_input
    def _call(self, inputs: Iterable[Tensor], **kwargs: Any) -> Tensor:
        """
        Concatenates the input tensors along the specified axis.
        If auto_broadcast is set to True, the tensors are broadcasted to the
        same rank before concatenating.


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
            # Since shapes can be dynamic (None), we need to use ops.shape
            max_dynamic_shape = []
            for i in range(max_rank - 1):
                shapes = [ops.shape(x)[i] for x in reshaped_inputs]
                max_dynamic_shape.append(ops.max(ops.stack(shapes)))

            # Broadcast tensors to the maximum dynamic shape if the static is different
            # WARNING: It assumes that when the static shapes of two tensors are None
            # at a given rank, the dynamic shapes are the same.
            for idx, x in enumerate(reshaped_inputs):
                x_static_shape = x.shape[:-1]
                if x_static_shape != max_static_shape:
                    last_dim = x.shape[-1]
                    broadcast_shape = ops.concatenate(
                        [
                            ops.stack(max_dynamic_shape),
                            ops.convert_to_tensor([last_dim]),
                        ],
                        axis=0,
                    )
                    broadcasted_x = ops.broadcast_to(x, broadcast_shape)
                    reshaped_inputs[idx] = broadcasted_x
            inputs = reshaped_inputs

        return ops.concatenate(inputs, axis=self.axis)
