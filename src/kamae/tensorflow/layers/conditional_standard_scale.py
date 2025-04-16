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
class ConditionalStandardScaleLayer(NormalizeLayer):
    """
    Performs the standard scaling of the input with a masking condition.
    This layer will shift and scale inputs into a distribution centered around
    0 with standard deviation 1. It accomplishes this by precomputing the mean
    and variance of the data, and calling `(input - mean) / sqrt(var)` at
    runtime.
    The skip_zeros parameter allows to apply the standard scaling process
    only when input is not equal to zero. If equal to zero, it will remain zero in
    the output value as it was in the input value.
    """

    def __init__(
        self,
        mean,
        variance,
        name: str = None,
        axis=-1,
        input_dtype: str = None,
        output_dtype: str = None,
        skip_zeros: bool = False,
        epsilon: float = 0,
        **kwargs,
    ) -> None:
        """
        Intialise the ConditionalStandardScaleLayer layer.
        :param mean: The mean value(s) to use during normalization. The passed value(s)
        will be broadcast to the shape of the kept axes above; if the value(s)
        cannot be broadcast, an error will be raised when this layer's
        `build()` method is called.
        :param variance: The variance value(s) to use during normalization. The passed
        value(s) will be broadcast to the shape of the kept axes above; if the
        value(s) cannot be broadcast, an error will be raised when this
        layer's `build()` method is called.
        :param name: The name of the layer. Defaults to `None`.
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
        :param skip_zeros: If True, in addition to the masking operation,
        do not apply the scaling when the values to scale are equal to zero.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param epsilon: Small value to add to conditional check of zeros. Valid only
        when skipZeros is True. Defaults to 1e-4.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        """
        super().__init__(
            name=name,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            mean=mean,
            variance=variance,
            axis=axis,
            **kwargs,
        )
        self.skip_zeros = skip_zeros
        self.epsilon = epsilon

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> Tensor:
        """
        Performs normalization on the input tensor(s) by calling the keras
        ConditionalStandardScaleLayer layer.
        It applies the scaling only to values matching the mask condition, if set.
        It applies the scaling only to values not equal to zero, if skip_zeros is set.
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
            tf.math.maximum(
                tf.sqrt(variance), tf.constant(self.epsilon, dtype=inputs.dtype)
            ),
        )
        # output is 0 if variance is 0
        normalized_outputs = tf.where(
            tf.equal(variance, 0),
            tf.zeros_like(normalized_outputs),
            normalized_outputs,
        )
        if self.skip_zeros:
            eps = tf.constant(self.epsilon, dtype=inputs.dtype)
            normalized_outputs = tf.where(
                tf.abs(inputs) <= eps,  # x = (0 +- eps)
                tf.zeros_like(normalized_outputs),
                normalized_outputs,
            )
        return normalized_outputs

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the ConditionalStandardScaleLayer layer.
        Used for saving and loading from a model.
        Specifically adds additional parameters to the base configuration.
        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "skip_zeros": self.skip_zeros,
                "epsilon": self.epsilon,
            }
        )
        return config
