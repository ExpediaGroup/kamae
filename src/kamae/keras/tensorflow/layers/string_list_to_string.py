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

import tensorflow as tf

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input
from kamae.params import ParamSpec


class StringListToStringLayer(BaseLayer):
    """
    A layer that converts a list of strings to a single string along the specified
    axis.
    If `keepdims` is `True`, the shape is retained.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = ["string"]
    _params = {
        "axis": ParamSpec(
            default=-1,
            doc="The axis along which to join the strings.",
        ),
        "separator": ParamSpec(
            default="",
            doc="The separator to use when joining the strings.",
        ),
        "keepdims": ParamSpec(
            default=False,
            doc="Whether to keep the shape of the input tensor.",
        ),
    }

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Joins the strings along the specified axis with the specified separator.
        If `keepdims` is `True`, the shape is retained. Otherwise the shape is
        reduced along the specified axis.

        :param inputs: Input tensor.
        :returns: Tensor with strings joined along the specified axis.
        """
        return tf.strings.reduce_join(
            inputs, axis=self.axis, separator=self.separator, keepdims=self.keepdims
        )
