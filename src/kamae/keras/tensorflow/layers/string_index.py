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
from tensorflow.keras.layers import StringLookup

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input
from kamae.params import _REQUIRED, ParamSpec
from kamae.params.shared_specs import STRING_INDEX_PARAMS


class StringIndexLayer(BaseLayer):
    """
    Wrapper around the Keras StringLookup layer.

    This layer translates a set of arbitrary strings into integer output via a
    table-based vocabulary lookup. This layer will perform no splitting or
    transformation of input strings.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = ["string"]
    _params = {
        **{
            k: v
            for k, v in STRING_INDEX_PARAMS.items()
            if k in ("maskToken", "numOOVIndices")
        },
        "encoding": ParamSpec(
            default="utf-8",
            doc="The text encoding to use to interpret the input strings.",
        ),
        "vocabulary": ParamSpec(
            default=_REQUIRED,
            doc="Either an array of strings or a string path to a text file containing the vocabulary.",
        ),
    }

    @staticmethod
    def _post_init(self):
        self.indexer = StringLookup(
            vocabulary=self.vocabulary,
            num_oov_indices=self.num_oov_indices,
            mask_token=self.mask_token,
            encoding=self.encoding,
        )

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Performs string indexing by calling the StringLookup layer.


        :param inputs: Input string tensor to index.
        :returns: Indexed tensor.
        """
        return self.indexer(inputs)
