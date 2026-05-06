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
class PairwiseCosineSimilarityLayer(BaseLayer):
    """
    Computes pairwise cosine similarity between a query embedding and
    each candidate embedding packed in a flat array.

    Input 0: (..., D)       -- query embedding
    Input 1: (..., N * D)   -- flat candidate embeddings
    Output:  (..., N)       -- cosine similarity per candidate
    """

    def __init__(
        self,
        name: Optional[str] = None,
        input_dtype: Optional[str] = None,
        output_dtype: Optional[str] = None,
        embedding_dim: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.embedding_dim = embedding_dim

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        return [tf.bfloat16, tf.float16, tf.float32, tf.float64]

    @enforce_multiple_tensor_input
    def _call(self, inputs: Iterable[Tensor], **kwargs: Any) -> Tensor:
        if len(inputs) != 2:
            raise ValueError(f"Expected 2 inputs, received {len(inputs)} instead.")

        query = inputs[0]  # (..., D)
        flat_candidates = inputs[1]  # (..., N*D)

        # Reshape: (..., N*D) -> (..., N, D)
        orig_shape = tf.shape(flat_candidates)
        num_candidates = orig_shape[-1] // self.embedding_dim
        new_shape = tf.concat(
            [orig_shape[:-1], [num_candidates, self.embedding_dim]], axis=0
        )
        candidates = tf.reshape(flat_candidates, new_shape)

        # (..., D) -> (..., 1, D) for broadcasting
        query_expanded = tf.expand_dims(query, axis=-2)

        # L2 normalize along embedding dimension
        q_norm = tf.nn.l2_normalize(query_expanded, axis=-1)
        c_norm = tf.nn.l2_normalize(candidates, axis=-1)

        # Dot product along last axis: (..., N)
        similarities = tf.reduce_sum(tf.multiply(q_norm, c_norm), axis=-1)

        # Zero-vector → NaN from normalization → replace with 0.0
        return tf.where(
            tf.math.is_nan(similarities),
            tf.zeros_like(similarities),
            similarities,
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"embedding_dim": self.embedding_dim})
        return config
