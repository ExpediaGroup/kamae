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

"""
Multi-backend base layer with string support on TensorFlow backend.

This base layer provides casting and dtype validation for layers that work across
TensorFlow, JAX, and PyTorch backends.

String operations (input_dtype="string" or output_dtype="string") are supported
only when running on TensorFlow backend. Multi-backend numeric operations work
on all backends.
"""

from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import keras
import tensorflow as tf
from keras import ops

import kamae
from kamae.keras.core.backend import ALL_BACKENDS, current_backend, require_tensorflow
from kamae.keras.core.typing import Tensor
from kamae.keras.core.utils.input_utils import allow_single_or_multiple_tensor_input


@keras.saving.register_keras_serializable(package=kamae.__name__)
class BaseLayer(keras.layers.Layer, ABC):
    """
    Abstract base layer for multi-backend layers with TensorFlow string support.

    Provides:
    - Multi-backend numeric dtype casting (works on TensorFlow, JAX, PyTorch)
    - String dtype casting (TensorFlow backend only)
    - Dtype compatibility validation
    - Numeric constant type coercion
    - Boolean string parsing (TensorFlow backend only)

    String operations automatically work when running on TensorFlow backend.
    Attempting to use string dtypes on JAX or PyTorch backends raises an error.
    """

    supported_backends: frozenset = ALL_BACKENDS
    jit_compatible: bool = False

    def __init__(
        self,
        name: Optional[str] = None,
        input_dtype: Optional[str] = None,
        output_dtype: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the BaseLayer.

        :param name: Name of the layer, defaults to `None`.
        :param input_dtype: Input data type of the layer. If specified, inputs will be
        cast to this data type before any computation is performed. Defaults to `None`.
        :param output_dtype: Output data type of the layer. Defaults to `None`. If
        specified, the output will be cast to this data type before being returned.
        """
        backend = current_backend()
        if backend not in self.supported_backends:
            raise RuntimeError(
                f"{self.__class__.__name__} requires one of {sorted(self.supported_backends)} backends. "
                f"Current backend: '{backend}'. "
                f"Set KERAS_BACKEND=tensorflow before importing keras."
            )
        super().__init__(name=name, **kwargs)
        # Disable Keras automatic casting to prevent float32 coercion
        # This is critical for layers that require 64-bit precision (e.g., timestamps)
        self._autocast = False
        self._convert_input_args = False
        self._input_dtype = input_dtype
        self._output_dtype = output_dtype
        self.true_bool_strings = ["true", "t", "yes", "y", "1"]
        self.false_bool_strings = ["false", "f", "no", "n", "0"]

    @property
    @abstractmethod
    def compatible_dtypes(self) -> Optional[List[str]]:
        """
        List of compatible data type names for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible dtype names (e.g., ['float32', 'float64'])
        or None if any dtype is compatible.
        """
        raise NotImplementedError

    def _string_to_bool_cast(self, inputs: Tensor) -> Tensor:
        """
        Casts a string tensor to a bool tensor.

        :param inputs: Input string tensor
        :returns: Bool tensor.
        """
        if keras.backend.standardize_dtype(inputs.dtype) != "string":
            raise TypeError(
                f"Expected a string tensor, but got a "
                f"{keras.backend.standardize_dtype(inputs.dtype)} tensor."
            )

        # Replace true strings with "1" and false strings with "0"
        is_bool_true_string_tensor = [
            tf.strings.lower(inputs) == bool_string
            for bool_string in self.true_bool_strings
        ]
        is_bool_false_string_tensor = [
            tf.strings.lower(inputs) == bool_string
            for bool_string in self.false_bool_strings
        ]

        string_bool_tensor = tf.where(
            reduce(tf.math.logical_or, is_bool_true_string_tensor),
            tf.constant("1"),
            inputs,
        )
        string_bool_tensor = tf.where(
            reduce(tf.math.logical_or, is_bool_false_string_tensor),
            tf.constant("0"),
            string_bool_tensor,
        )

        # If we have other strings that are not "1" or "0", these are invalid.
        # We insert these as "NULL" values so that the casting will fail.
        string_bool_tensor_with_invalid = tf.where(
            tf.math.logical_or(string_bool_tensor == "1", string_bool_tensor == "0"),
            string_bool_tensor,
            tf.constant("NULL"),
        )

        bool_float_tensor = tf.strings.to_number(
            string_bool_tensor_with_invalid, out_type=tf.float32
        )
        return tf.cast(bool_float_tensor, tf.bool)

    @staticmethod
    def _float_to_string_cast(inputs: Tensor) -> Tensor:
        """
        Casts a float tensor to a string tensor. Ensures that the precision of the float
        does not impact the string representation. Specifically, we want the string
        to be the shortest possible representation of the float,
        i.e. 1.145000 -> "1.145".

        However, we also want to ensure that the string representation of the float
        has a decimal point, i.e. 2.00000 -> "2.0" and not "2".

        :param inputs: Input string tensor
        :returns: Float tensor.
        """
        # This gives 1.145000 -> "1.145" and 2.00000 -> "2".
        # We need to add a decimal point to the second example.
        shortest_float_string = tf.strings.as_string(inputs, shortest=True)

        # Find strings without decimal points
        no_decimal = tf.logical_not(
            tf.strings.regex_full_match(
                shortest_float_string, "-?\d*\.\d*"  # noqa W605
            )
        )
        # Create decimal point constant string
        decimal_string = tf.constant(".0")

        # Add decimal point to string without decimal points
        return tf.where(
            no_decimal,
            tf.strings.join([shortest_float_string, decimal_string]),
            shortest_float_string,
        )

    def _to_string_cast(self, inputs: Tensor) -> Tensor:
        """
        Casts inputs to string tensor.

        :param inputs: Input tensor.
        :returns: String tensor.
        """
        if "float" in keras.backend.standardize_dtype(inputs.dtype):
            return self._float_to_string_cast(inputs)
        return tf.strings.as_string(inputs)

    def _from_string_cast(self, inputs: Tensor, cast_dtype: str) -> Tensor:
        """
        Casts inputs to the desired dtype when inputs are a string tensor.

        :param inputs: String tensor
        :param cast_dtype: Dtype to cast to.
        :returns: Tensor cast to the desired dtype.
        """
        if keras.backend.standardize_dtype(inputs.dtype) != "string":
            raise TypeError("inputs is not a string Tensor.")
        if cast_dtype in ["float32", "float64", "int32", "int64"]:
            return tf.strings.to_number(inputs, out_type=cast_dtype)
        elif "int" in cast_dtype:
            intermediate_cast = tf.strings.to_number(inputs, out_type="int64")
            return ops.cast(intermediate_cast, cast_dtype)
        elif "float" in cast_dtype:
            intermediate_cast = tf.strings.to_number(inputs, out_type="float64")
            return ops.cast(intermediate_cast, cast_dtype)
        elif cast_dtype == "bool":
            return self._string_to_bool_cast(inputs)
        else:
            raise TypeError(f"Casting string to dtype {cast_dtype} is not supported.")

    def _string_cast(self, inputs: Tensor, cast_dtype: str) -> Tensor:
        """
        Casts from and to string tensors.

        Either inputs is a string tensor, and we want to cast it to the desired dtype,
        or inputs is not a string tensor, and we want to cast it to a string tensor.

        Requires TensorFlow backend.

        :param inputs: Input tensor.
        :param cast_dtype: Dtype to cast to.
        :returns: Tensor cast to the desired dtype.
        """
        require_tensorflow()

        if (
            keras.backend.standardize_dtype(inputs.dtype) == "string"
            and cast_dtype == "string"
        ):
            return inputs
        if cast_dtype == "string":
            return self._to_string_cast(inputs)
        return self._from_string_cast(inputs, cast_dtype)

    @staticmethod
    def _check_string_dtype_backend_compatibility(dtype_str: str) -> None:
        """
        Check if string dtype is used on a non-TensorFlow backend.

        String operations are only supported on TensorFlow backend. JAX and PyTorch
        do not support string tensors.

        :param dtype_str: Dtype string to check (e.g., 'float32', 'string')
        :raises RuntimeError: If string dtype is used on JAX or PyTorch backend.
        """
        if dtype_str == "string":
            backend = keras.backend.backend()
            if backend != "tensorflow":
                raise RuntimeError(
                    f"String dtype is not supported on '{backend}' backend. "
                    f"String operations require TensorFlow backend. "
                    f"Set KERAS_BACKEND=tensorflow before importing keras."
                )

    @staticmethod
    def _numeric_cast(inputs: Tensor, cast_dtype: str) -> Tensor:
        """
        Casts a numeric tensor to the desired dtype using keras.ops.

        :param inputs: Input numeric tensor
        :param cast_dtype: Dtype to cast to (e.g., 'float32', 'int64')
        :returns: Tensor cast to the desired dtype.
        """
        # keras.ops.cast doesn't support string dtype, even on TF backend
        # Check if we're on TF backend and dealing with strings
        if cast_dtype == "string" or (
            hasattr(inputs, "dtype")
            and keras.backend.standardize_dtype(inputs.dtype) == "string"
        ):
            if keras.backend.backend() == "tensorflow":
                return (
                    tf.strings.as_string(inputs)
                    if cast_dtype == "string"
                    else tf.cast(inputs, cast_dtype)
                )
            else:
                # String operations not supported on JAX/PyTorch backends
                raise ValueError(
                    f"String dtype casting not supported on {keras.backend.backend()} backend. "
                    "String operations require TensorFlow backend."
                )
        return ops.cast(inputs, cast_dtype)

    def _cast(self, inputs: Tensor, cast_dtype: str) -> Tensor:
        """
        Casts inputs to the desired dtype.

        Routes to string casting when string dtype is involved (TensorFlow backend only),
        otherwise uses numeric casting for multi-backend compatibility.

        :param inputs: Input tensor.
        :param cast_dtype: Dtype to cast to.
        :returns: Tensor cast to the desired dtype.
        """
        # Check if string dtype is involved
        if (
            keras.backend.standardize_dtype(inputs.dtype) == "string"
            or cast_dtype == "string"
        ):
            return self._string_cast(inputs, cast_dtype)
        return self._numeric_cast(inputs, cast_dtype)

    def _force_cast_to_compatible_numeric_type(
        self, inputs: Tensor, constant: Union[float, int]
    ) -> Tuple[Tensor, Tensor]:
        """
        Casts an input tensor and a single constant to compatible numeric tensors.

        This ensures operations between tensors and constants work correctly:
        - If input is float, constant becomes float of same precision
        - If input is int and constant is int, keep as int of same precision
        - If input is int but constant is float, cast input to float

        :param inputs: Input numeric tensor
        :param constant: The constant to cast to the compatible dtype.
        :returns: Tuple of (cast_input, cast_constant) with compatible types
        """
        input_dtype = keras.backend.standardize_dtype(inputs.dtype)

        # Check if dtype is floating point
        if "float" in input_dtype:
            # Input is float - cast constant to same precision
            if isinstance(constant, float):
                return inputs, ops.convert_to_tensor(constant, dtype=input_dtype)
            return inputs, ops.convert_to_tensor(float(constant), dtype=input_dtype)

        # Check if dtype is integer
        if "int" in input_dtype:
            # Input is integer
            if isinstance(constant, int):
                # Constant is also int - keep as int
                return inputs, ops.convert_to_tensor(constant, dtype=input_dtype)

            if isinstance(constant, float) and constant.is_integer():
                # Constant is float but represents an integer
                return inputs, ops.convert_to_tensor(int(constant), dtype=input_dtype)

            if isinstance(constant, float):
                # Constant is truly float - need to cast input to float
                # Extract precision (e.g., int32 -> 32 bits)
                if "64" in input_dtype:
                    float_dtype = "float64"
                else:
                    float_dtype = "float32"
                return (
                    self._cast(inputs, float_dtype),
                    ops.convert_to_tensor(constant, dtype=float_dtype),
                )

        raise TypeError(
            f"inputs must be a numeric tensor (got {input_dtype}) "
            f"and constant must be a numeric value (got {type(constant)})."
        )

    def _cast_input_output_tensors(
        self, tensors: Union[Tensor, List[Tensor]], ingress: bool
    ) -> Union[Tensor, List[Tensor]]:
        """
        Casts either the input or output tensors to the given input/output dtype, if
        specified. Ingress is a boolean that indicates whether we are casting the
        input (True) or output (False) tensors.

        :param tensors: The input or output tensor(s) to the layer to be cast.
        :param ingress: Boolean indicating whether we are casting the input (True) or
        output (False) tensors.
        :returns: The input or output tensor(s) cast to the desired input/output_dtype.
        """
        if ingress:
            cast_dtype = self._input_dtype
            # Validate input_dtype is compatible
            if (
                cast_dtype is not None
                and self.compatible_dtypes is not None
                and cast_dtype not in self.compatible_dtypes
            ):
                raise ValueError(
                    f"input_dtype {cast_dtype} is not a compatible dtype for "
                    f"this layer. Compatible dtypes are {self.compatible_dtypes}."
                )
        else:
            cast_dtype = self._output_dtype

        if cast_dtype is not None:
            # Check if string dtype is being used on non-TF backend
            self._check_string_dtype_backend_compatibility(cast_dtype)
            # Check if tensors is a single tensor
            if not isinstance(tensors, list):
                current_dtype = keras.backend.standardize_dtype(tensors.dtype)
                return (
                    self._cast(tensors, cast_dtype)
                    if current_dtype != cast_dtype
                    else tensors
                )
            # Handle list of tensors
            return [
                self._cast(inp, cast_dtype)
                if keras.backend.standardize_dtype(inp.dtype) != cast_dtype
                else inp
                for inp in tensors
            ]
        return tensors

    def cast_input_tensors(
        self, inputs: Union[Tensor, List[Tensor]]
    ) -> Union[Tensor, List[Tensor]]:
        """
        Casts the input tensors to the given input dtype, if specified. All tensors are
        cast to this. Subclasses can override for more complex casting behavior.

        :param inputs: The input tensor(s) to the layer.
        :returns: The input tensor(s) cast to the desired input_dtype.
        """
        return self._cast_input_output_tensors(tensors=inputs, ingress=True)

    def cast_output_tensors(
        self, outputs: Union[Tensor, List[Tensor]]
    ) -> Union[Tensor, List[Tensor]]:
        """
        Casts the output tensors to the given output dtype, if specified. All tensors
        are cast to this. Subclasses can override for more complex casting behavior.

        :param outputs: The output tensor(s) of the layer.
        :returns: The output tensor(s) cast to the desired output_dtype.
        """
        return self._cast_input_output_tensors(tensors=outputs, ingress=False)

    def _check_input_dtypes_compatible(self, inputs: List[Tensor]) -> None:
        """
        Checks if the input tensors are compatible with the compatible_dtypes of the
        layer.

        :param inputs: The input tensor(s) to the layer.
        :raises ValueError: If the input tensors are not compatible with the
        compatible_dtypes of the layer.
        :returns: None
        """
        if self.compatible_dtypes is None:
            # Any dtype is compatible, but check for string dtype on non-TF backends
            for inp in inputs:
                inp_dtype = keras.backend.standardize_dtype(inp.dtype)
                self._check_string_dtype_backend_compatibility(inp_dtype)
            return

        for inp in inputs:
            inp_dtype = keras.backend.standardize_dtype(inp.dtype)
            if inp_dtype not in self.compatible_dtypes:
                raise TypeError(
                    f"Input tensor with dtype {inp_dtype} "
                    f"is not a compatible dtype for this layer. "
                    f"Compatible dtypes are {self.compatible_dtypes}."
                )

    @allow_single_or_multiple_tensor_input
    def call(
        self, inputs: Iterable[Tensor], **kwargs: Any
    ) -> Union[Tensor, List[Tensor]]:
        """
        Casts inputs to the given `input_dtype`, calls the internal `_call` method, and
        casts the outputs to the given `output_dtype`.

        :param inputs: The input tensor(s) to the layer.
        :returns: The output tensor(s) of the layer.
        """
        # Cast inputs to a compatible dtype for the layer
        casted_inputs = self.cast_input_tensors(inputs=inputs)
        # Check if the input tensors are now compatible with the layer
        self._check_input_dtypes_compatible(inputs=casted_inputs)
        # Call the internal _call method
        outputs = self._call(inputs=casted_inputs, **kwargs)
        # Cast outputs to the desired output_dtype
        casted_outputs = self.cast_output_tensors(outputs=outputs)
        return casted_outputs

    @abstractmethod
    def _call(
        self, inputs: Union[Tensor, List[Tensor]], **kwargs: Any
    ) -> Union[Tensor, List[Tensor]]:
        """
        The internal call method that should be implemented by the layer.

        Subclasses implement this method to define the layer's computation.
        Input and output casting is handled by the base class `call()` method.

        :param inputs: The input tensor(s) to the layer (after input casting).
        :returns: The output tensor(s) of the layer (before output casting).
        """
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of the BaseLayer.
        Used for saving and loading from a model.

        :returns: Dictionary of the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "name": self.name,
                "input_dtype": self._input_dtype,
                "output_dtype": self._output_dtype,
            }
        )
        return config
