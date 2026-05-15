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
from tensorflow.keras.layers import Hashing

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input
from kamae.params import _REQUIRED, ParamSpec
from kamae.params.shared_specs import MASK_VALUE_PARAMS


class HashIndexLayer(BaseLayer):
    """
    Wrapper around the Keras Hashing layer which hashes and bins categorical features.

    This layer transforms categorical inputs to hashed output. It element-wise
    converts ints or strings to ints in a fixed range. The stable hash
    function uses `tensorflow::ops::Fingerprint` to produce the same output
    consistently across all platforms.

    This layer uses [FarmHash64](https://github.com/google/farmhash),
    which provides a consistent hashed output across different platforms and is
    stable across invocations, regardless of device and context, by mixing the
    input bits thoroughly.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = ["string"]
    _params = {
        "num_bins": ParamSpec(
            default=_REQUIRED,
            doc="Number of hash bins. Note that this includes the `mask_value` bin, so the effective number of bins is `(num_bins - 1)` if `mask_value` is set.",
        ),
        **MASK_VALUE_PARAMS,
    }

    @staticmethod
    def _post_init(self):
        if self.mask_value is not None:
            self.hash_indexer = Hashing(
                name=self.name, num_bins=self.num_bins, mask_value=self.mask_value
            )
        else:
            self.hash_indexer = Hashing(name=self.name, num_bins=self.num_bins - 1)

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Performs the hash indexing on the input tensor by calling the underlying
        Hashing layer.


        :param inputs: Input tensor to be hashed.
        :returns: Hashed and bucketed tensor.
        """
        result = self.hash_indexer(inputs)
        if self.mask_value is None:
            result = result + 1
        return result
