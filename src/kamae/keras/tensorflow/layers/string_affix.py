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


class StringAffixLayer(BaseLayer):
    """
    Performs a prefixing and suffing on the input tensor.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = ["string"]
    _params = {
        "prefix": ParamSpec(
            default=None,
            doc="The prefix to apply to tensor.",
        ),
        "suffix": ParamSpec(
            default=None,
            doc="The suffix to apply to tensor.",
        ),
    }

    @staticmethod
    def _post_init(self):
        if (self.prefix is None or self.prefix == "") and (
            self.suffix is None or self.suffix == ""
        ):
            raise ValueError(
                "Either prefix or suffix must be set. Otherwise nothing to affix."
            )

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Prefixes and suffixes a given input tensor.


        :param inputs: Input tensor to affix. Must be string tensors.
        :returns: A tensor with affixed values - same shape as input.
        """
        x = inputs
        if self.prefix:
            x = tf.strings.join([self.prefix, x])
        if self.suffix:
            x = tf.strings.join([x, self.suffix])
        return x
