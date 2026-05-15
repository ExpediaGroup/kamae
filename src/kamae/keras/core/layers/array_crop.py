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

from typing import Any

from keras import ops

from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input
from kamae.params import ParamSpec


class ArrayCropLayer(BaseLayer):
    """
    Performs a cropping of the input tensor to a certain length.
    If the tensor is shorter than the specified length, it is
    padded with specified pad value.


    TODO: Currently only supports cropping the final dimension of the tensor.
    """

    jit_compatible = True

    _compatible_dtypes = None
    _params = {
        "array_length": ParamSpec(
            default=128,
            doc="The length to crop or pad the arrays to",
        ),
        "pad_value": ParamSpec(
            default=None,
            doc="The value to pad the arrays with",
        ),
    }

    def _post_init(self):
        if self.array_length < 1:
            raise ValueError("Array length must be greater than 0.")
        if self.pad_value is None:
            raise ValueError("Pad value must be provided and not None.")

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Crops the tensor to specified length and pads with specified value.

        :param inputs: Tensor to split.
        :returns: Cropped and padded tensor
        """
        # Crop final dimension of tensor
        # Use static shape for slicing if available, otherwise dynamic
        if inputs.shape[-1] is not None:
            crop_length = min(self.array_length, inputs.shape[-1])
            cropped = inputs[..., :crop_length]
            padding_needed = max(self.array_length - inputs.shape[-1], 0)
        else:
            # Dynamic shape - need runtime computation
            dynamic_last_dim = ops.shape(inputs)[-1]
            crop_length = ops.minimum(self.array_length, dynamic_last_dim)
            cropped = inputs[..., :crop_length]
            padding_needed = ops.maximum(self.array_length - dynamic_last_dim, 0)

        # Pad final dim of tensor if necessary
        rank = len(inputs.shape)
        paddings = [[0, 0]] * (rank - 1) + [[0, padding_needed]]
        padded = ops.pad(cropped, paddings, constant_values=self.pad_value)

        # Build target shape tuple for reshape
        # Use static shape dimensions where available, dynamic where needed
        new_shape_list = []
        for i in range(rank - 1):
            if padded.shape[i] is not None:
                new_shape_list.append(padded.shape[i])
            else:
                new_shape_list.append(ops.shape(padded)[i])
        new_shape_list.append(self.array_length)

        return ops.reshape(padded, new_shape_list)
