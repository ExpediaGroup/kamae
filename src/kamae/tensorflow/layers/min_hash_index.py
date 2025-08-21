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

from typing import Any, Dict, List, Optional

import tensorflow as tf
from tensorflow.keras.layers import Hashing

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import enforce_single_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
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

    def __init__(
        self,
        name: Optional[str] = None,
        input_dtype: Optional[str] = None,
        output_dtype: Optional[str] = None,
        num_permutations: int = 128,
        mask_value: Optional[str] = None,
        axis: int = -1,
        **kwargs: Any,
    ) -> None:
        """
        Initialises the MinHashIndexLayer layer.

        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param num_permutations: Number of permutations to use for the min hashing.
            Defaults to 128.
        :param mask_value: A value that represents masked inputs, which are ignored when
        computing the min hash. Defaults to None, meaning no mask term will be added.
        :param axis: The axis along which to compute the min hash.
        Defaults to -1 (last axis).
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.num_permutations = num_permutations
        self.axis = axis
        self.mask_value = mask_value
        self.hash_fn = Hashing(
            # Set the number of bins to the maximum integer value. We just want to hash
            # the input without binning it, so we use the maximum integer value.
            num_bins=tf.int32.max
        )

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [tf.string]

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Performs the min hash indexing on the input tensor.

        Decorated with `@enforce_single_tensor_input` to ensure that the input
        is a single tensor. Raises an error if multiple tensors are passed
        in as an iterable.

        :param inputs: Input tensor to be encoded.
        :returns: Encoded tensor.
        """
        min_hash_signature = []
        for i in range(self.num_permutations):
            # Salt the input
            salted_inputs = tf.strings.join(
                [inputs, tf.zeros_like(inputs)], separator=str(i)
            )
            # Hash the salted inputs.
            if self.mask_value is not None:
                hashed_inputs = tf.where(
                    tf.equal(salted_inputs, f"{self.mask_value}{i}"),
                    # Use the maximum integer value for masked inputs, therefore it is
                    # never selected as the minimum.
                    tf.ones_like(salted_inputs, dtype=tf.int64) * tf.int32.max,
                    self.hash_fn(salted_inputs),
                )
            else:
                hashed_inputs = self.hash_fn(salted_inputs)
            min_hash_value = tf.reduce_min(hashed_inputs, axis=self.axis, keepdims=True)
            min_hash_bit = min_hash_value & 1
            min_hash_signature.append(min_hash_bit)

        # Concatenate the min hash values to form the final signature.
        return tf.concat(min_hash_signature, axis=self.axis)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the MinHashIndex layer.

        :returns: Configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "num_permutations": self.num_permutations,
                "mask_value": self.mask_value,
                "axis": self.axis,
            }
        )
        return config
