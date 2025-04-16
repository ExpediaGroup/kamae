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

from typing import Any, Dict

import tensorflow as tf

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import NormalizeLayer, enforce_single_tensor_input


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class StandardScaleLayer(NormalizeLayer):
    """
    Performs the standard scaling of the input.
    This layer will shift and scale inputs into a distribution centered around
    0 with standard deviation 1. It accomplishes this by precomputing the mean
    and variance of the data, and calling `(input - mean) / sqrt(var)` at
    runtime. mask_value is used to ignore certain values in the standard scaling
    process. They will remain the same value in the output value as they were in
    the input value.
    """

    def __init__(
        self,
        mean,
        variance,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        axis=-1,
        mask_value: float = None,
        **kwargs,
    ) -> None:
        """
        Intialise the StandardScaleLayer layer.
        :param mean: The mean value(s) to use during normalization. The passed value(s)
        will be broadcast to the shape of the kept axes above; if the value(s)
        cannot be broadcast, an error will be raised when this layer's
        `build()` method is called.
        :param variance: The variance value(s) to use during normalization. The passed
        value(s) will be broadcast to the shape of the kept axes above; if the
        value(s) cannot be broadcast, an error will be raised when this
        layer's `build()` method is called.
        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param axis: Integer, tuple of integers, or None. The axis or axes that should
        have a separate mean and variance for each index in the shape. For
        example, if shape is `(None, 5)` and `axis=1`, the layer will track 5
        separate mean and variance values for the last axis. If `axis` is set
        to `None`, the layer will normalize all elements in the input by a
        scalar mean and variance. Defaults to -1, where the last axis of the
        input is assumed to be a feature dimension and is normalized per
        index. Note that in the specific case of batched scalar inputs where
        the only axis is the batch axis, the default will normalize each index
        in the batch separately. In this case, consider passing `axis=None`.
        :param mask_value: Value which should be ignored in the standard scaling
        process and left unchanged.
        """
        super().__init__(
            name=name,
            mean=mean,
            variance=variance,
            axis=axis,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            **kwargs,
        )
        self.mask_value = mask_value

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> Tensor:
        """
        Performs normalization on the input tensor(s) by calling the keras
        StandardScaleLayer layer. It ignores values which are equal to the
        mask_value.
        Decorated with `@enforce_single_tensor_input` to ensure that
        the input is a single tensor. Raises an error if multiple tensors are passed
        in as an iterable.
        :param inputs: Input tensor to perform the normalization on.
        :returns: The input tensor with the normalization applied.
        """
        # Ensure mean and variance match input dtype.
        mean = self._cast(self.mean, inputs.dtype.name)
        variance = self._cast(self.variance, inputs.dtype.name)
        normalized_outputs = tf.math.divide_no_nan(
            tf.math.subtract(inputs, mean),
            tf.math.maximum(tf.sqrt(variance), tf.constant(1e-8, dtype=inputs.dtype)),
        )
        if self.mask_value is not None:
            mask = tf.equal(inputs, self.mask_value)
            normalized_outputs = tf.where(
                mask, inputs, self._cast(normalized_outputs, inputs.dtype.name)
            )
        return normalized_outputs

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the StandardScaleLayer layer.
        Used for saving and loading from a model.
        Specifically adds additional parameters to the base configuration.
        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        # Ensure mean and variance are lists for serialization.
        config.update(
            {
                "mask_value": self.mask_value,
            }
        )
        return config
