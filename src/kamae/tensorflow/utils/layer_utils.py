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

import numpy as np
import tensorflow as tf

from kamae.tensorflow.layers.base import BaseLayer
from kamae.tensorflow.utils import listify_tensors


class NormalizeLayer(BaseLayer):
    """
    Intermediate layer for normalization layers.

    Reduces code duplication by providing a common interface for normalization layers.
    """

    def __init__(
        self,
        mean,
        variance,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        axis: int = -1,
        **kwargs,
    ) -> None:
        """
        Initializes the NormalizeLayer

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
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        # Standardize `axis` to a tuple.
        if axis is None:
            axis = ()
        elif isinstance(axis, int):
            axis = (axis,)
        else:
            axis = tuple(axis)

        self.axis = axis
        self.input_mean = mean
        self.input_variance = variance
        self.epsilon = 1e-8

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [tf.bfloat16, tf.float16, tf.float32, tf.float64]

    def build(self, input_shape) -> None:
        """
        Builds shapes for the mean and variance tensors.

        Specifically, understands which axis to compute the normalization across
        and broadcasts the mean and variance tensors to match the input shape.

        :param input_shape: The shape of the input tensor.
        :returns: None - layer is built.
        """
        super().build(input_shape)

        if isinstance(input_shape, (list, tuple)) and all(
            isinstance(shape, (tf.TensorShape, list, tuple)) for shape in input_shape
        ):
            # This seems to be needed to handle sending in multiple inputs as a list.
            # Although this layer should only have one input, so this is a bit of a
            # hack. We catch this nicely in call method with a decorator. Maybe we
            # should do the same here?
            input_shape = input_shape[0]

        input_shape = tf.TensorShape(input_shape).as_list()
        ndim = len(input_shape)
        self._build_input_shape = input_shape

        if any(a < -ndim or a >= ndim for a in self.axis):
            raise ValueError(
                f"""All `axis` values must be in the range [-ndim, ndim). "
                Found ndim: `{ndim}`, axis: {self.axis}"""
            )

        # Axes to be kept, replacing negative values with positive equivalents.
        # Sorted to avoid transposing axes.
        keep_axis = sorted([d if d >= 0 else d + ndim for d in self.axis])
        # All axes to be kept should have known shape.
        for d in keep_axis:
            if input_shape[d] is None:
                raise ValueError(
                    f"""All `axis` values to be kept must have known shape. "
                    Got axis: {self.axis},
                    input shape: {input_shape}, with unknown axis at index: {d}"""
                )
        # Broadcast any reduced axes.
        broadcast_shape = [input_shape[d] if d in keep_axis else 1 for d in range(ndim)]
        mean_and_var_shape = tuple(input_shape[d] for d in keep_axis)
        mean = self.input_mean * np.ones(mean_and_var_shape)
        variance = self.input_variance * np.ones(mean_and_var_shape)
        self.mean = tf.reshape(mean, broadcast_shape)
        self.variance = tf.reshape(variance, broadcast_shape)

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
                "mean": listify_tensors(self.input_mean),
                "variance": listify_tensors(self.input_variance),
                "axis": self.axis,
            }
        )
        return config

    def get_build_config(self):
        if self._build_input_shape:
            return {"input_shape": self._build_input_shape}

    def build_from_config(self, config):
        if config:
            self.build(config["input_shape"])
