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

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from keras import ops

from kamae.keras.core.base import BaseLayer
from kamae.keras.core.utils.tensor_utils import listify_tensors
from kamae.params import _REQUIRED, ParamSpec


class NormalizeLayer(BaseLayer):
    """
    Intermediate layer for normalization layers.

    Reduces code duplication by providing a common interface for normalization layers.

    This is a backend-agnostic layer that works with TensorFlow, JAX, and PyTorch.
    """

    _compatible_dtypes = ["bfloat16", "float16", "float32", "float64"]
    _params = {
        "mean": ParamSpec(
            default=_REQUIRED,
            doc="Mean of the feature values.",
        ),
        "variance": ParamSpec(
            default=_REQUIRED,
            doc="Variance of the feature values.",
        ),
        "axis": ParamSpec(
            default=-1,
            doc="The axis that should have a separate mean and variance",
        ),
    }

    def _post_init(self):
        """Standardize axis to tuple format and store input mean/variance."""
        if self.axis is None:
            self.axis = ()
        elif isinstance(self.axis, int):
            self.axis = (self.axis,)
        else:
            self.axis = tuple(self.axis)

        self.input_mean = self.mean
        self.input_variance = self.variance

    def build(self, input_shape: Tuple[int]) -> None:
        """
        Builds shapes for the mean and variance tensors.

        Specifically, understands which axis to compute the normalization across
        and broadcasts the mean and variance tensors to match the input shape.

        :param input_shape: The shape of the input tensor.
        :returns: None - layer is built.
        """
        super().build(input_shape)

        if isinstance(input_shape, (list, tuple)):
            self._build_input_shape = tuple(input_shape)
        else:
            self._build_input_shape = input_shape

        if not isinstance(input_shape, list):
            input_shape = list(input_shape)

        # Handle Keras serialization quirk: when a tuple like (100, 10, 5) is saved
        # and deserialized, Keras may wrap it as [(100, 10, 5)]
        if len(input_shape) == 1 and isinstance(input_shape[0], (list, tuple)):
            input_shape = list(input_shape[0])

        ndim = len(input_shape)

        if any(a < -ndim or a >= ndim for a in self.axis):
            raise ValueError(
                f"""All `axis` values must be in the range [-ndim, ndim). "
                Found ndim: `{ndim}`, axis: {self.axis}"""
            )

        keep_axis = sorted([d if d >= 0 else d + ndim for d in self.axis])
        for d in keep_axis:
            if input_shape[d] is None:
                raise ValueError(
                    f"""All `axis` values to be kept must have known shape. "
                    Got axis: {self.axis},
                    input shape: {input_shape}, with unknown axis at index: {d}"""
                )
        broadcast_shape = [input_shape[d] if d in keep_axis else 1 for d in range(ndim)]
        mean_and_var_shape = tuple(
            int(input_shape[d][0])
            if isinstance(input_shape[d], tuple)
            else int(input_shape[d])
            for d in keep_axis
        )
        mean = self.input_mean * np.ones(mean_and_var_shape)
        variance = self.input_variance * np.ones(mean_and_var_shape)
        self.mean = ops.reshape(mean, broadcast_shape)
        self.variance = ops.reshape(variance, broadcast_shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "mean": listify_tensors(self.input_mean),
                "variance": listify_tensors(self.input_variance),
                "axis": list(self.axis) if self.axis else None,
            }
        )
        return config

    def get_build_config(self) -> Optional[Dict[str, Any]]:
        if self._build_input_shape:
            return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        if config:
            self.build(config["input_shape"])
