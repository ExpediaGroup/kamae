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
from kamae.tensorflow.utils import allow_single_or_multiple_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class SegmentMeanLayer(BaseLayer):
    """
    Calculate the mean of the first passed tensor, segmented by the values of the second passed tensor.

    Example: calculate the mean price of rates within rooms in the same query.
    """

    def __init__(
        self,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        axis: int = 1,
        **kwargs,
    ):
        """
        Initializes the Segment Mean layer.

        WARNING: The code is fully tested for axis=1 only. Further testing is needed.

        :param name: Name of the layer, defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param axis: The axis to calculate the statistics across. Defaults to 1.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.axis = axis

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
            tf.uint8,
            tf.int8,
            tf.uint16,
            tf.int16,
            tf.int32,
            tf.int64,
            tf.complex64,
            tf.complex128,
            tf.string,
        ]

    @allow_single_or_multiple_tensor_input
    def _call(self, inputs: Iterable[Tensor], **kwargs) -> Tensor:
        """
        Calculate the mean of the first input, segmented by the values of the second input.

        :param inputs: The iterable tensors for the features.
        :returns: Thew new tensor result column.
        """

        def per_batch_mean(v, s):
            flat_v = tf.reshape(v, [-1])
            flat_s = tf.reshape(s, [-1])
            unique_segments, segment_indices = tf.unique(flat_s)
            num_segments = tf.size(unique_segments)

            # Cast
            flat_v = tf.cast(flat_v, tf.float32)

            min_vals = tf.math.unsorted_segment_mean(
                flat_v, segment_indices, num_segments
            )
            gathered = tf.gather(min_vals, segment_indices)
            return tf.reshape(gathered, tf.shape(v))

        values = inputs[0]
        segments = inputs[1]

        return tf.map_fn(
            lambda x: per_batch_mean(x[0], x[1]),
            (values, segments),
            fn_output_signature=tf.TensorSpec(shape=values.shape[1:], dtype=tf.float32),
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the layer.
        Used for saving and loading from a model.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
            }
        )
        return config
