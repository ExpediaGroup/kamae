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

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import enforce_single_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class BucketizeLayer(BaseLayer):
    """
    Performs a bucketing operation on the input tensor.
    Given a list of splits, the input tensor is bucketed into
    the corresponding bucket. For example, if the splits are
    [0, 1, 2, 3], then the input tensor is bucketed into 4 buckets:
    (-inf, 0), [0, 1), [1, 2), [2, 3), [3, inf).
    These buckets are int64 values, starting from 1. The 0 index
    is reserved for padding values.
    """

    def __init__(
        self,
        splits: List[float],
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        **kwargs,
    ) -> None:
        """
        Initialises the BucketizeLayer layer.

        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param splits: The splits to use for bucketing.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        if splits != sorted(splits):
            raise ValueError("`splits` argument must be a sorted list!")
        self.splits = splits

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [tf.int32, tf.int64, tf.float32, tf.float64]

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> Tensor:
        """
        Performs the bucketing operation on the input tensor.

        Decorated with `@enforce_single_tensor_input` to ensure that
        the input is a single tensor. Raises an error if multiple tensors are passed
        in as an iterable.

        :param inputs: Input tensor to bucket.
        :returns: Bucketed tensor.
        """
        # We add 1 to the output of the bucket layer so that we can use
        # 0 index as a padding value.
        bucketed_outputs = tf.raw_ops.Bucketize(input=inputs, boundaries=self.splits)
        return self._cast(tf.math.add(bucketed_outputs, 1), "int64")

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the Bucketizer layer.
        Used for saving and loading from a model.

        Specifically adds the `splits` argument to the base config.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update({"splits": self.splits})
        return config
