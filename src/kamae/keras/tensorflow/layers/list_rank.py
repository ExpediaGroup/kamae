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

import tensorflow as tf

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input
from kamae.params import ParamSpec
from kamae.params.shared_specs import LISTWISE_FILTER_PARAMS, LISTWISE_PARAMS


class ListRankLayer(BaseLayer):
    """
    Calculate the rank across the axis dimension.

    Example: calculate the rank of items within a query, given the score.
    """

    supported_backends = TENSORFLOW_ONLY
    jit_compatible = True

    _compatible_dtypes = [
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "uint8",
        "int8",
        "uint16",
        "int16",
        "int32",
        "int64",
    ]
    _params = {
        **{k: v for k, v in LISTWISE_PARAMS.items() if k != "queryIdCol"},
        **LISTWISE_FILTER_PARAMS,
        "sort_order": ParamSpec(
            default="desc",
            doc="The order to sort the input tensor by. Defaults to 'desc'.",
        ),
    }

    @enforce_single_tensor_input
    def _call(self, inputs: Iterable[Tensor], **kwargs: Any) -> Tensor:
        """
        Calculate the rank.

        :param inputs: The iterable tensor for the feature.
        :returns: The new tensor result column.
        """
        return tf.math.add(
            tf.argsort(
                tf.argsort(
                    inputs,
                    axis=self.axis,
                    direction="ASCENDING" if self.sort_order == "asc" else "DESCENDING",
                    stable=True,
                ),
                axis=self.axis,
                stable=True,
            ),
            1,
        )
