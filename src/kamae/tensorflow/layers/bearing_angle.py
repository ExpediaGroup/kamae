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

import math
from typing import Any, Dict, Iterable, List, Optional

import tensorflow as tf
from tensorflow.math import atan2, cos, mod, sin

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import enforce_multiple_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class BearingAngleLayer(BaseLayer):
    """
    Computes the Bearing angle operation on a given input tensor.
    If lat_lon_constant is not set, inputs must be a list of 4 tensors,
    in the order of lat1, lon1, lat2, lon2.
    If lat_lon_constant is set, inputs must be a tensor of 2 tensors,
    in the order of lat1, lon1.

    We DO NOT check if the lat/lon values are out of bounds.
    For lat, this is [-90, 90] and for lon, this is [-180, 180].
    """

    def __init__(
        self,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        lat_lon_constant: List[float] = None,
        **kwargs,
    ) -> None:
        """
        Initializes the BearingAngleLayer layer

        :param name: Name of the layer, defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param lat_lon_constant: The lat/lons to use in the bearing angle
        calculation. Defaults to `None`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        if lat_lon_constant is not None and len(lat_lon_constant) != 2:
            raise ValueError("If set, lat_lon_constant must be a list of 2 floats")
        self.lat_lon_constant = lat_lon_constant

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [tf.bfloat16, tf.float16, tf.float32, tf.float64]

    @staticmethod
    def get_radians(degrees: Tensor) -> Tensor:
        """
        Converts degrees tensor to radians. We need to cast to float64 otherwise
        pi / 180 will lose precision.

        :param degrees: Tensor of degrees.
        :returns: Tensor of radians.
        """
        return tf.cast(degrees, dtype=tf.float64) * tf.constant(
            math.pi / 180, dtype=tf.float64
        )

    @staticmethod
    def get_degrees(radians: Tensor) -> Tensor:
        """
        Converts radians tensor to degrees.

        :param radians: Tensor of degrees.
        :returns: Tensor of degrees.
        """
        return tf.cast(radians, dtype=tf.float64) * tf.constant(
            180 / math.pi, dtype=tf.float64
        )

    def compute_bearing_angle(
        self, lat1: Tensor, lon1: Tensor, lat2: Tensor, lon2: Tensor
    ) -> Tensor:
        """
        Computes the bearing angle between two lat/lon pairs.

        :param lat1: Tensor of latitudes of the first point.
        :param lon1: Tensor of longitudes of the first point.
        :param lat2: Tensor of latitudes of the second point.
        :param lon2: Tensor of longitudes of the second point.
        :returns: Tensor of bearing angles.
        """
        lat1_radians = self.get_radians(lat1)
        lon1_radians = self.get_radians(lon1)
        lat2_radians = self.get_radians(lat2)
        lon2_radians = self.get_radians(lon2)

        lon_difference = lon2_radians - lon1_radians
        # Bearing formula calculation
        y = sin(lon_difference) * cos(lat2_radians)

        x = cos(lat1_radians) * sin(lat2_radians)
        x -= sin(lat1_radians) * cos(lat2_radians) * cos(lon_difference)

        # Calculate bearing in degrees
        bearing = atan2(y, x)
        bearing_deg = mod(self.get_degrees(bearing) + 360, 360)
        return bearing_deg

    @enforce_multiple_tensor_input
    def _call(self, inputs: Iterable[Tensor], **kwargs) -> Tensor:
        """
        Computes the bearing angle between two lat/lon pairs.

        Decorated with @enforce_multiple_tensor_input to ensure that the input
        is an iterable of tensors. Raises an error if a single tensor is passed.

        After decoration, we check the length of the inputs to ensure we have the right
        number of lat/lon tensors.

        :param inputs: Iterable of tensors.
        :returns: Tensor of bearing angles.
        """
        if self.lat_lon_constant is not None:
            if not isinstance(inputs, list) or len(inputs) != 2:
                raise ValueError(
                    """If lat_lon_constant is set,
                inputs must be a list of 2 tensors"""
                )
            return self.compute_bearing_angle(
                inputs[0],
                inputs[1],
                tf.constant(self.lat_lon_constant[0]),
                tf.constant(self.lat_lon_constant[1]),
            )
        else:
            if not isinstance(inputs, list) or len(inputs) != 4:
                raise ValueError(
                    """If lat_lon_constant is not set,
                inputs must be a list of 4 tensors"""
                )
            return self.compute_bearing_angle(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
            )

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the Bearing Angle layer.
        Used for saving and loading from a model.

        Specifically, we add the `lat_lon_constant` and `unit` to the config.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update({"lat_lon_constant": self.lat_lon_constant})
        return config
