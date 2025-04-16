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
class BloomEncodeLayer(BaseLayer):
    """
    Performs a bloom encoding on the input tensor. Uses multiple hash functions to
    encode the input tensor, significantly reducing the dimensionality of the input
    and also avoiding collisions. See paper for more details.
    https://arxiv.org/pdf/1706.03993.pdf

    In Kamae we actually use the same hash function for all the hash functions,
    but we use a salt to make sure that the hash functions are different. Therefore,
    this can be seen as a psuedo-bloom encoding.
    """

    def __init__(
        self,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        num_hash_fns: int = 3,
        num_bins: int = None,
        mask_value: Union[int, str] = None,
        feature_cardinality: int = None,
        use_heuristic_num_bins: bool = False,
        **kwargs,
    ) -> None:
        """
        Intialises the BloomEncodeLayer layer.

        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param num_hash_fns: Number of hash functions to use. Defaults to 3.
        The paper suggests a range of 2-4 hash functions for optimal performance.
        :param num_bins: Number of hash bins. Note that this includes the `mask_value`
        bin, so the effective number of bins is `(num_bins - 1)` if `mask_value`
        is set. If `use_heuristic_num_bins` is set to True, then this parameter is
        ignored and the number of bins is automatically set. See the description of this
        parameter below for how the heuristic is built.
        :param mask_value: A value that represents masked inputs, which are mapped to
        index 0. Defaults to None, meaning no mask term will be added and the
        hashing will start at index 0.
        :param feature_cardinality: The cardinality of the input tensor. Needed to use
        the heuristic to set the number of bins. Defaults to None, meaning the number of
        bins will not be set using the heuristic and must be set manually.
        :param use_heuristic_num_bins: If set to True, the number of bins is
        automatically set by fixing the ratio of the feature dimensionality to the
        number of bins to be b/f = 0.2. This ratio was found to be optimal in the paper
        for a wide variety of usecases. Therefore, num_bins = feature_cardinality * 0.2.
        This reduces the cardinality of the input tensor by 5x.
        Requires the `feature_cardinality` parameter to be set. Defaults to False.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        if num_hash_fns < 2:
            raise ValueError("The number of hash functions must be at least 2.")
        self.num_hash_fns = num_hash_fns
        self.mask_value = mask_value
        self.feature_cardinality = feature_cardinality
        self.use_heuristic_num_bins = use_heuristic_num_bins

        if use_heuristic_num_bins and feature_cardinality is None:
            raise ValueError(
                """If use_heuristic_num_bins is set to True, then the
                feature_cardinality parameter must be set."""
            )
        if num_bins is None and not use_heuristic_num_bins:
            raise ValueError(
                """If use_heuristic_num_bins is set to False, then the
                num_bins parameter must be set."""
            )
        self.num_bins = (
            num_bins
            if not use_heuristic_num_bins
            else max(round(feature_cardinality * 0.2), 2)
        )
        # We need to create multiple hashing layers if we have a mask_value, as the
        # mask_value needs salting in the same manner as the input tensor. Hence it is
        # not constant across the hash functions. If the mask_value is None, then we
        # can use the same hash function for all the hash functions.
        if mask_value is None:
            hash_fn = Hashing(num_bins=self.num_bins)
            self.hash_fns = {f"{i}": hash_fn for i in range(self.num_hash_fns)}
        else:
            self.hash_fns = {
                f"{i}": Hashing(
                    num_bins=self.num_bins,
                    mask_value=f"{self.mask_value}{i}",
                )
                for i in range(self.num_hash_fns)
            }

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
        Performs the bloom encoding on the input tensor.

        Decorated with `@enforce_single_tensor_input` to ensure that the input
        is a single tensor. Raises an error if multiple tensors are passed
        in as an iterable.

        :param inputs: Input tensor to be encoded.
        :returns: Encoded tensor.
        """
        # Expand dimensions to add the bloom encoding dimension for two scenarios:
        # 1. If the final dimension is not 1, in which case we do not want to use
        # this dimension for the encoding.
        # 2. If the rank of the tensor is less than 2, then we have a single dimensional
        # tensor thus we add a dimension for the encoding.
        expanded_inputs = (
            tf.expand_dims(inputs, axis=-1)
            if inputs.shape[-1] != 1 or len(inputs.shape) < 2
            else inputs
        )
        # Salt the inputs to create multiple hash functions
        # Add `i` to the input tensor, where `i` represents the ith hash function.
        salted_inputs = [
            tf.strings.join(
                [expanded_inputs, tf.zeros_like(expanded_inputs)], separator=str(i)
            )
            for i in range(self.num_hash_fns)
        ]
        # Hash the salted inputs.
        hashed_inputs = [
            self.hash_fns[f"{i}"](salted_inputs[i]) for i in range(self.num_hash_fns)
        ]
        return tf.concat(hashed_inputs, axis=-1)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the BloomEncode layer.

        :returns: Configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "num_hash_fns": self.num_hash_fns,
                "num_bins": self.num_bins,
                "mask_value": self.mask_value,
                "feature_cardinality": self.feature_cardinality,
                "use_heuristic_num_bins": self.use_heuristic_num_bins,
            }
        )
        return config
