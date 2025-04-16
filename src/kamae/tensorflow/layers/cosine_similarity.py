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

from typing import Any, Dict, Iterable, List, Optional

import tensorflow as tf

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import enforce_multiple_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class CosineSimilarityLayer(BaseLayer):
    """
    Computes the cosine similarity between two input tensors.
    """

    def __init__(
        self,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        axis: int = -1,
        keepdims: bool = False,
        **kwargs,
    ) -> None:
        """
        Initializes the CosineSimilarityLayer layer

        :param name: Name of the layer, defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param axis: The axis along which to compute the cosine similarity. Defaults to
        `-1`.
        :param keepdims: Whether to keep the shape of the input tensor. Defaults to
        `False`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.axis = axis
        self.keepdims = keepdims

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [
            tf.bfloat16,
            tf.float16,
            tf.float32,
            tf.float64,
            tf.complex64,
            tf.complex128,
        ]

    @enforce_multiple_tensor_input
    def _call(self, inputs: Iterable[Tensor], **kwargs) -> Tensor:
        """
        Computes the cosine similarity between two input tensors. If `keepdims` is
        `True`, the shape is retained. Otherwise, the shape is reduced along the
        specified axis.

        Decorated with @enforce_multiple_tensor_input to ensure that the input
        is an iterable of tensors. Raises an error if a single tensor is passed.

        After decoration, we check the length of the inputs to ensure we have the right
        number of input tensors.

        :param inputs: List of two tensors to compute the cosine similarity between.
        :returns: The tensor resulting from the cosine similarity.
        """
        if len(inputs) != 2:
            raise ValueError(
                f"Expected 2 inputs, received {len(inputs)} inputs instead."
            )
        x = tf.nn.l2_normalize(inputs[0], axis=self.axis)
        y = tf.nn.l2_normalize(inputs[1], axis=self.axis)

        return tf.reduce_sum(tf.multiply(x, y), axis=self.axis, keepdims=self.keepdims)

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the CosineSimilarity layer.
        Used for saving and loading from a model.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update({"axis": self.axis, "keepdims": self.keepdims})
        return config
