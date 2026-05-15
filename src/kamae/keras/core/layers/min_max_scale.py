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

import keras
import numpy as np
from keras import ops

from kamae.keras.core.base import BaseLayer
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input
from kamae.keras.core.utils.ops_utils import divide_no_nan
from kamae.keras.core.utils.tensor_utils import listify_tensors
from kamae.params import _REQUIRED, ParamSpec
from kamae.params.shared_specs import MASK_VALUE_PARAMS


class MinMaxScaleLayer(BaseLayer):
    """
    Performs a min-max scaling operation on the input tensor(s).

    This is used to standardize/transform the input tensor
    to the range [0, 1] using the minimum and maximum values.

    Formula: (x - min)/(max - min)
    """

    jit_compatible = True

    _compatible_dtypes = ["bfloat16", "float16", "float32", "float64"]
    _params = {
        "min": ParamSpec(
            default=_REQUIRED,
            doc="The min value(s) to use during scaling",
        ),
        "max": ParamSpec(
            default=_REQUIRED,
            doc="The max value(s) to use during scaling",
        ),
        "axis": ParamSpec(
            default=-1,
            doc="The axis that should have a separate min and max",
        ),
        **MASK_VALUE_PARAMS,
    }

    def _post_init(self):
        """Standardize axis to tuple format and store input min/max."""
        # Standardize `axis` to a tuple.
        if self.axis is None:
            self.axis = ()
        elif isinstance(self.axis, int):
            self.axis = (self.axis,)
        else:
            self.axis = tuple(self.axis)

        # Store original min/max for serialization
        self.input_min = self.min
        self.input_max = self.max

    def build(self, input_shape: Tuple[int]) -> None:
        """
        Builds shapes for the min and max tensors.

        Specifically, understands which axis to compute the scaling across
        and broadcasts the min and max tensors to match the input shape.

        :param input_shape: The shape of the input tensor.
        :returns: None - layer is built.
        """
        super().build(input_shape)

        # Save the original input_shape for serialization
        # Store as tuple to ensure consistent format
        if isinstance(input_shape, (list, tuple)):
            self._build_input_shape = tuple(input_shape)
        else:
            self._build_input_shape = input_shape

        # Ensure input_shape is a list for easier manipulation
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

        # Axes to be kept, replacing negative values with positive equivalents.
        # Sorted to avoid transposing axes.
        keep_axis = sorted([d if d >= 0 else d + ndim for d in self.axis])
        # All axes to be kept should have known shape.
        for d in keep_axis:
            if input_shape[d] is None:
                raise ValueError(
                    f"""All `axis` values to be kept must have known shape. "
                    Got axis: {self.axis},
                    input shape: {input_shape}, with unknown axis at index: {d}"""
                )
        # Broadcast any reduced axes.
        broadcast_shape = [input_shape[d] if d in keep_axis else 1 for d in range(ndim)]
        # Extract shape dimensions - handle both int and tuple (e.g., 5 or (5,))
        min_and_max_shape = tuple(
            int(input_shape[d][0])
            if isinstance(input_shape[d], tuple)
            else int(input_shape[d])
            for d in keep_axis
        )
        min_tensor = self.input_min * np.ones(min_and_max_shape)
        max_tensor = self.input_max * np.ones(min_and_max_shape)
        self.min = ops.reshape(min_tensor, broadcast_shape)
        self.max = ops.reshape(max_tensor, broadcast_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the MinMaxScaleLayer layer.
        Used for saving and loading from a model.
        Specifically adds additional parameters to the base configuration.
        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        # Ensure min and max are lists for serialization.
        config["min"] = listify_tensors(self.input_min)
        config["max"] = listify_tensors(self.input_max)
        return config

    def get_build_config(self) -> Optional[Dict[str, Any]]:
        """
        Gets the build configuration of the MinMaxScaleLayer layer.

        Used for saving and loading from a model.

        :returns: Dictionary of the build configuration of the layer.
        """
        if self._build_input_shape:
            return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """
        Builds the min/max tensor shapes from the provided configuration.

        Specifically it calls the `build` method with the input shape in order to
        construct the min and max tensors with the correct shape.

        :param config: Configuration dictionary containing the input shape.
        :returns: None - layer is built.
        """
        if config:
            self.build(config["input_shape"])

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Performs normalization on the input tensor(s) to scale it to the range [0, 1]


        :param inputs: Input tensor to perform the normalization on.
        :returns: The input tensor with the normalization applied.
        """
        # Ensure min and max match input dtype.
        input_dtype_str = keras.backend.standardize_dtype(inputs.dtype)
        min_tensor = self._cast(self.min, input_dtype_str)
        max_tensor = self._cast(self.max, input_dtype_str)

        # Compute (input - min) / (max - min) using safe division
        numerator = ops.subtract(inputs, min_tensor)
        denominator = ops.subtract(max_tensor, min_tensor)
        normalized_outputs = divide_no_nan(numerator, denominator)

        if self.mask_value is not None:
            mask = ops.equal(inputs, self.mask_value)
            normalized_outputs = ops.where(
                mask, inputs, self._cast(normalized_outputs, input_dtype_str)
            )
        return normalized_outputs
