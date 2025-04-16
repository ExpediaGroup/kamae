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

from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
from tensorflow.keras.layers import Hashing

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import enforce_single_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
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

    def __init__(
        self,
        num_bins: int,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        mask_value: Union[int, str] = None,
        **kwargs,
    ) -> None:
        """
        Intialise the HashIndexLayer layer.

        :param num_bins: Number of hash bins. Note that this includes the `mask_value`
        bin, so the effective number of bins is `(num_bins - 1)` if `mask_value`
        is set.
        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param mask_value: A value that represents masked inputs, which are mapped to
        index 0. Defaults to None, meaning no mask term will be added and the
        hashing will start at index 0.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.num_bins = num_bins
        self.mask_value = mask_value
        self.hash_indexer = Hashing(name=name, num_bins=num_bins, mask_value=mask_value)

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [tf.string]

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> Tensor:
        """
        Performs the hash indexing on the input tensor by calling the underlying
        Hashing layer.

        Decorated with `@enforce_single_tensor_input` to ensure that the input
        is a single tensor. Raises an error if multiple tensors are passed
        in as an iterable.

        :param inputs: Input tensor to be hashed.
        :returns: Hashed and bucketed tensor.
        """
        return self.hash_indexer(inputs)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the HashIndexLayer layer.

        :returns: Configuration of the HashIndexLayer layer.
        """
        config = super().get_config()
        config.update({"num_bins": self.num_bins, "mask_value": self.mask_value})
        return config
