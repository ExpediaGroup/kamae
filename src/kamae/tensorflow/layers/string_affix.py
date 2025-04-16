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

import tensorflow as tf

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import enforce_single_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(kamae.__name__)
class StringAffixLayer(BaseLayer):
    """
    Performs a prefixing and suffing on the input tensor.
    """

    def __init__(
        self,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        prefix: str = None,
        suffix: str = None,
        **kwargs,
    ) -> None:
        """
        Initialises the String Affix layer.
        :param name: The name of the layer. Defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        :param prefix: The prefix to apply to tensor.
        :param suffix: The suffix to apply to tensor.
        """
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.prefix = prefix
        self.suffix = suffix
        self.validate_params()

    def validate_params(self) -> None:
        """
        Validates the parameters of the layer.
        :raises ValueError: If both prefix and suffix are not set.
        """
        if (self.prefix is None or self.prefix == "") and (
            self.suffix is None or self.suffix == ""
        ):
            raise ValueError(
                "Either prefix or suffix must be set. Otherwise nothing to affix."
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
        Prefixes and suffixes a given input tensor.

        Decorated with `@enforce_single_tensor_input` to ensure that the input
        is a single tensor. Raises an error if multiple tensors are passed
        in as an iterable.

        :param inputs: Input tensor to affix. Must be string tensors.
        :returns: A tensor with affixed values - same shape as input.
        """
        x = inputs
        if self.prefix:
            x = tf.strings.join([self.prefix, x])
        if self.suffix:
            x = tf.strings.join([x, self.suffix])
        return x

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the StringAffix layer.
        Used for saving and loading from a model.

        Specifically adds the `prefix` and `suffix` values to the configuration.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update({"prefix": self.prefix, "suffix": self.suffix})
        return config
