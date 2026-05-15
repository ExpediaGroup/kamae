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

from typing import Any

import tensorflow as tf

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input
from kamae.params import ParamSpec


def _validate_string_case_type(value):
    if value not in ["upper", "lower"]:
        raise ValueError(f"string_case_type must be 'upper' or 'lower'. Got {value}")
    return value


class StringCaseLayer(BaseLayer):
    """
    Performs a string case transform on the input tensor.
    Supported string case types are 'upper' and 'lower'.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = ["string"]
    _params = {
        "string_case_type": ParamSpec(
            default="lower",
            doc="The type of string case transform to perform. Supported types are 'upper' and 'lower'.",
            validator=_validate_string_case_type,
        ),
    }

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Performs the string case transform on the input tensor.

        :param inputs: Input tensor to perform the string case transform on.
        :returns: The input tensor with the string case transform applied.
        """
        if self.string_case_type == "upper":
            return tf.strings.upper(inputs)
        elif self.string_case_type == "lower":
            return tf.strings.lower(inputs)
        else:
            raise ValueError(
                f"""stringCaseType must be one of 'upper' or 'lower'.
                Got {self.string_case_type}"""
            )
