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

import warnings
from typing import Any

import tensorflow as tf

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input
from kamae.params import _REQUIRED, ParamSpec
from kamae.params.shared_specs import DROP_UNSEEN_PARAMS, STRING_INDEX_PARAMS


class OneHotEncodeLayer(BaseLayer):
    """
    Performs a one-hot encoding of a string input tensor.

    Encodes each individual element in the input into an
    array the same size as the vocabulary, containing a 1 at the element
    index. If the last dimension is size 1, will encode on that
    dimension. If the last dimension is not size 1, will append a new
    dimension for the encoded output.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = ["int16", "int32", "int64", "string"]
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
        **DROP_UNSEEN_PARAMS,
    }

    @staticmethod
    def _post_init(self):
        self.lookup_layer = tf.keras.layers.StringLookup(
            vocabulary=self.vocabulary,
            output_mode="int",
            num_oov_indices=self.num_oov_indices,
            mask_token=self.mask_token,
            encoding=self.encoding,
        )

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Performs the one-hot encoding on the input tensor.


        :param inputs: Input tensor to one-hot encode.
        :returns: One-hot encoded input tensor.
        """
        casted_inputs = (
            tf.strings.as_string(inputs, scientific=False)
            if inputs.dtype != tf.string
            else inputs
        )
        indexed_inputs = self.lookup_layer(casted_inputs)
        mask_offset = 1 if self.mask_token is not None else 0

        # If last dimension to encode is 1,
        # remove it after one-hot encoding.
        # E.g. (None, None, 1) -> (None, None, 1, N) -> (None, None, N)
        # But (None, None, M) -> (None, None, M, N)
        ohe_depth = len(self.vocabulary) + self.num_oov_indices + mask_offset
        encoded_inputs = (
            tf.squeeze(tf.one_hot(indexed_inputs, ohe_depth), axis=-2)
            if indexed_inputs.get_shape()[-1] == 1
            else tf.one_hot(indexed_inputs, ohe_depth)
        )

        # If drop unseen, slice off the first num_oov_indices + mask_offset columns
        if self.drop_unseen:
            encoded_inputs = encoded_inputs[..., (self.num_oov_indices + mask_offset) :]

        return encoded_inputs


# TODO: Remove this alias in next breaking change,
#  it is maintained for backwards compatibility
class OneHotLayer(OneHotEncodeLayer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "OneHotLayer is deprecated and will be removed in a future release. "
            "Use OneHotEncodeLayer instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        super().__init__(*args, **kwargs)
