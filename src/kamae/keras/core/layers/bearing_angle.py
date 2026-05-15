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

from typing import Any, Iterable

from keras import ops

from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_multiple_tensor_input
from kamae.keras.core.utils.ops_utils import get_degrees, get_radians
from kamae.params import ParamSpec


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

    jit_compatible = True

    _compatible_dtypes = ["bfloat16", "float16", "float32", "float64"]
    _params = {
        "lat_lon_constant": ParamSpec(
            default=None,
            doc="The lat/lons to use in the bearing angle calculation",
        ),
    }

    def _post_init(self):
        if self.lat_lon_constant is not None and len(self.lat_lon_constant) != 2:
            raise ValueError("If set, lat_lon_constant must be a list of 2 floats")

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
        lat1_radians = get_radians(lat1)
        lon1_radians = get_radians(lon1)
        lat2_radians = get_radians(lat2)
        lon2_radians = get_radians(lon2)

        lon_difference = lon2_radians - lon1_radians
        # Bearing formula calculation
        y = ops.sin(lon_difference) * ops.cos(lat2_radians)

        x = ops.cos(lat1_radians) * ops.sin(lat2_radians)
        x -= ops.sin(lat1_radians) * ops.cos(lat2_radians) * ops.cos(lon_difference)

        # Calculate bearing in degrees
        bearing = ops.arctan2(y, x)
        bearing_deg = ops.mod(get_degrees(bearing) + 360, 360)
        return bearing_deg

    @enforce_multiple_tensor_input
    def _call(self, inputs: Iterable[Tensor], **kwargs: Any) -> Tensor:
        """
        Computes the bearing angle between two lat/lon pairs.



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
                ops.convert_to_tensor(self.lat_lon_constant[0]),
                ops.convert_to_tensor(self.lat_lon_constant[1]),
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
