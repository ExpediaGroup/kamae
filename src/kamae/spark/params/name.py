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

# pylint: disable=unused-argument
# pylint: disable=invalid-name
# pylint: disable=too-many-ancestors
# pylint: disable=no-member
from pyspark.ml.param import Param, Params, TypeConverters


class HasLayerName(Params):
    """
    Mixin class for a layer name.
    """

    layerName = Param(
        Params._dummy(),
        "layerName",
        "Name of the layer",
        typeConverter=TypeConverters.toString,
    )

    def getLayerName(self) -> str:
        """
        Gets the value of the layerName parameter.

        :returns: Layer name.
        """
        return self.getOrDefault(self.layerName)
