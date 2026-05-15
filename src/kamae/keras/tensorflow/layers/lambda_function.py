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

from typing import Any, Callable, Iterable, List, Optional, Union

import tensorflow as tf

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import allow_single_or_multiple_tensor_input
from kamae.params import _REQUIRED, ParamSpec


class LambdaFunctionLayer(BaseLayer, tf.keras.layers.Lambda):
    """
    Performs the lambda function operation on a given input tensor

    WARNING: This layer relies on a `tf.keras.layers.Lambda` layer which have
    (de)serialization limitations!

    `Lambda` layers are saved by serializing the Python bytecode, which is fundamentally
    non-portable. They should only be loaded in the same environment where
    they were saved.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = None
    _params = {
        "function": ParamSpec(
            default=_REQUIRED,
            doc="The lambda function to apply to the input tensor(s)",
        ),
    }

    def __init__(
        self,
        function: Callable[[Union[Tensor, List[Tensor]]], Union[Tensor, List[Tensor]]],
        name: Optional[str] = None,
        input_dtype: Optional[str] = None,
        output_dtype: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the LambdaFunction layer

        :param function: The lambda function to apply to the input tensor(s).
        :param name: Name of the layer, defaults to `None`.
        :param input_dtype: The dtype to cast the input to. Defaults to `None`.
        :param output_dtype: The dtype to cast the output to. Defaults to `None`.
        """
        super().__init__(
            name=name,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            function=function,
            **kwargs,
        )

    @allow_single_or_multiple_tensor_input
    def _call(
        self, inputs: Union[Tensor, Iterable[Tensor]], **kwargs: Any
    ) -> Union[Tensor, Iterable[Tensor]]:
        """
        Transforms the input tensor(s) by applying the lambda function.


        :param inputs: Tensor(s) to apply the lambda function to.
        :returns: The transformed tensor(s).
        """
        if len(inputs) == 1:
            return tf.keras.layers.Lambda.call(self, inputs[0], **kwargs)
        return tf.keras.layers.Lambda.call(self, inputs, **kwargs)
