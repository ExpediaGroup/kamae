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
from kamae.params import ParamSpec
from kamae.params.shared_specs import MASK_VALUE_PARAMS


class MinHashIndexLayer(BaseLayer):
    """
    Performs min hashing of the input tensor as described here:
    https://en.wikipedia.org/wiki/MinHash

    MinHash approximates the Jaccard similarity between sets by hashing the elements of
    the sets and returning a fixed-length signature. This length is determined by the
    num_permutations parameter, which defaults to 128. The output is an array of integer
    bits.

    Setting the mask_value parameter allows you to ignore a specific value in the
    input column when computing the min hash. This is useful if you have padded arrays
    as then a padded array with the same unique elements as another non-padded array
    will be considered equal.

    The minimum is computed across the last dimension of the input tensor.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = ["string"]
    _params = {
        "num_permutations": ParamSpec(
            default=128,
            doc="Number of permutations to use for the min hashing. Defaults to 128.",
        ),
        **MASK_VALUE_PARAMS,
        "axis": ParamSpec(
            default=-1,
            doc="The axis along which to compute the min hash. Defaults to -1 (last axis).",
        ),
    }

    @staticmethod
    def _post_init(self):
        self.hash_fn = Hashing(num_bins=tf.int32.max - 1)

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Performs the min hash indexing on the input tensor.


        :param inputs: Input tensor to be encoded.
        :returns: Encoded tensor.
        """
        min_hash_signature = []
        for i in range(self.num_permutations):
            # Salt the input
            salted_inputs = tf.strings.join(
                [inputs, tf.zeros_like(inputs)], separator=str(i)
            )
            # Hash the salted inputs and add 1 to reserve index 0 for nulls.
            if self.mask_value is not None:
                hashed_inputs = tf.where(
                    tf.equal(salted_inputs, f"{self.mask_value}{i}"),
                    # Use the maximum integer value for masked inputs, therefore it is
                    # never selected as the minimum.
                    tf.ones_like(salted_inputs, dtype=tf.int64) * tf.int32.max,
                    self.hash_fn(salted_inputs) + 1,
                )
            else:
                hashed_inputs = self.hash_fn(salted_inputs) + 1
            min_hash_value = tf.reduce_min(hashed_inputs, axis=self.axis, keepdims=True)
            min_hash_bit = min_hash_value & 1
            min_hash_signature.append(min_hash_bit)

        # Concatenate the min hash values to form the final signature.
        return tf.concat(min_hash_signature, axis=self.axis)
