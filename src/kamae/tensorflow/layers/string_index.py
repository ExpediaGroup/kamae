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

from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
from tensorflow.keras.layers import StringLookup

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import enforce_single_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class StringIndexLayer(BaseLayer):
    """
    Wrapper around the Keras StringLookup layer.

    This layer translates a set of arbitrary strings into integer output via a
    table-based vocabulary lookup. This layer will perform no splitting or
    transformation of input strings.
    """

    def __init__(
        self,
        vocabulary: Union[str, List[str]],
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        num_oov_indices: int = 1,
        mask_token: str = None,
        encoding: str = "utf-8",
        **kwargs,
    ) -> None:
        """
        Intialise the StringIndexLayer layer.

        :param vocabulary: Either an array of strings or a string path to a
        text file. If passing an array, can pass a tuple, list, 1D numpy array,
        or 1D tensor containing the string vocbulary terms. If passing a file
        path, the file should contain one line per term in the vocabulary.
        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param num_oov_indices: The number of out-of-vocabulary tokens to use. If this
        value is more than 1, OOV inputs are hashed to determine their OOV
        value. If this value is 0, OOV inputs will cause an error when calling
        the layer.  Defaults to 1.
        :param mask_token: A token that represents masked inputs. The token is included
        in vocabulary and mapped to index 0. If set to None, no mask term will be added.
        Defaults to `None`.
        :param encoding: Optional. The text encoding to use to interpret the input
        strings. Defaults to `"utf-8"`.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.vocabulary = vocabulary
        self.num_oov_indices = num_oov_indices
        self.mask_token = mask_token
        self.encoding = encoding
        self.indexer = StringLookup(
            vocabulary=vocabulary,
            num_oov_indices=num_oov_indices,
            mask_token=mask_token,
            encoding=encoding,
        )

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        Returns the compatible dtypes of the layer.

        :returns: The compatible dtypes of the layer.
        """
        return [tf.string]

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs) -> Tensor:
        """
        Performs string indexing by calling the StringLookup layer.

        Decorated with `@enforce_single_tensor_input` to ensure that the input
        is a single tensor. Raises an error if multiple tensors are passed
        in as an iterable.

        :param inputs: Input string tensor to index.
        :returns: Indexed tensor.
        """
        return self.indexer(inputs)

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the StringIndexer layer.
        Used for saving and loading from a model.

        Specifically adds the `vocabulary`, `num_oov_indices`, `mask_token`, and
        `encoding` to the config.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "vocabulary": self.vocabulary,
                "num_oov_indices": self.num_oov_indices,
                "mask_token": self.mask_token,
                "encoding": self.encoding,
            }
        )
        return config
