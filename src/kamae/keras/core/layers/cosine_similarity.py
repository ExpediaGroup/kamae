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
from kamae.keras.core.utils.ops_utils import l2_normalize
from kamae.params import ParamSpec


class CosineSimilarityLayer(BaseLayer):
    """
    Computes the cosine similarity between two input tensors.
    """

    jit_compatible = True

    _compatible_dtypes = [
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ]
    _params = {
        "axis": ParamSpec(
            default=-1, doc="The axis along which to compute the cosine similarity"
        ),
        "keepdims": ParamSpec(
            default=False, doc="Whether to keep the shape of the input tensor"
        ),
    }

    @enforce_multiple_tensor_input
    def _call(self, inputs: Iterable[Tensor], **kwargs: Any) -> Tensor:
        """
        Computes the cosine similarity between two input tensors. If `keepdims` is
        `True`, the shape is retained. Otherwise, the shape is reduced along the
        specified axis.



        :param inputs: List of two tensors to compute the cosine similarity between.
        :returns: The tensor resulting from the cosine similarity.
        """
        if len(inputs) != 2:
            raise ValueError(
                f"Expected 2 inputs, received {len(inputs)} inputs instead."
            )
        x = l2_normalize(inputs[0], axis=self.axis)
        y = l2_normalize(inputs[1], axis=self.axis)

        return ops.sum(ops.multiply(x, y), axis=self.axis, keepdims=self.keepdims)
