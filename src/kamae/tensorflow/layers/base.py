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

from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import tensorflow as tf

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import allow_single_or_multiple_tensor_input


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class BaseLayer(tf.keras.layers.Layer, ABC):
    """
    Abstract base layer that performs casting of inputs and outputs to specified
    data types. All layers should inherit from this class.
    """

    def __init__(
        self,
        name: str = None,
        input_dtype: str = None,
        output_dtype: str = None,
        **kwargs,
    ) -> None:
        """
        Initialises the BaseLayer.

        :param name: Name of the layer, defaults to `None`.
        :param input_dtype: Input data type of the layer. If specified, inputs will be
        cast to this data type before any computation is performed. Defaults to `None`.
        :param output_dtype: Output data type of the layer. Defaults to `None`. If
        specified, the output will be cast to this data type before being returned.
        """
        super().__init__(name=name, **kwargs)
        # We handle casting of inputs and outputs in the call method
        # Allowing keras to also autocast causes issues in some layers that require
        # 64 bit precision. Such as timestamp layers after the year 2038.
        self._autocast = False
        # Needed to ensure keras 3 does not autocast inputs to float32
        self._convert_input_args = False
        self._input_dtype = input_dtype
        self._output_dtype = output_dtype
        self.true_bool_strings = ["true", "t", "yes", "y", "1"]
        self.false_bool_strings = ["false", "f", "no", "n", "0"]

    @property
    @abstractmethod
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        raise NotImplementedError

    def _string_to_bool_cast(self, inputs: Tensor) -> Tensor:
        """
        Casts a string tensor to a bool tensor.

        :param inputs: Input string tensor
        :returns: Bool tensor.
        """
        if inputs.dtype.name != "string":
            raise TypeError(
                f"Expected a string tensor, but got a {inputs.dtype.name} tensor."
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
        if inputs.dtype.is_floating:
            return self._float_to_string_cast(inputs)
        return tf.strings.as_string(inputs)

    def _from_string_cast(self, inputs: Tensor, cast_dtype: str) -> Tensor:
        """
        Casts inputs to the desired dtype when inputs are a string tensor.

        :param inputs: String tensor
        :param cast_dtype: Dtype to cast to.
        :returns: Tensor cast to the desired dtype.
        """
        if inputs.dtype.name != "string":
            raise TypeError("inputs is not a string Tensor.")
        if cast_dtype in ["float32", "float64", "int32", "int64"]:
            # If the casting dtype is supported by tf.strings.to_number, we use that.
            return tf.strings.to_number(inputs, out_type=cast_dtype)
        elif tf.as_dtype(cast_dtype).is_integer:
            # If the casting dtype is an integer, we need to cast to int64 first
            intermediate_cast = tf.strings.to_number(inputs, out_type="int64")
            return tf.cast(intermediate_cast, cast_dtype)
        elif tf.as_dtype(cast_dtype).is_floating:
            # If the casting dtype is a float, we need to cast to float64 first
            intermediate_cast = tf.strings.to_number(inputs, out_type="float64")
            return tf.cast(intermediate_cast, cast_dtype)
        elif tf.as_dtype(cast_dtype).is_bool:
            # If the casting dtype is a boolean, we need to use a custom function
            # to cast the string to boolean.
            return self._string_to_bool_cast(inputs)
        else:
            raise TypeError(f"Casting string to dtype {cast_dtype} is not supported.")

    def _string_cast(self, inputs: Tensor, cast_dtype: str) -> Tensor:
        """
        Casts from and to string tensors.

        Either inputs is a string tensor, and we want to cast it to the desired dtype,
        or inputs is not a string tensor, and we want to cast it to a string tensor.

        :param inputs: Input tensor.
        :param cast_dtype: Dtype to cast to.
        :returns: Tensor cast to the desired dtype.
        """
        if inputs.dtype.name == "string" and cast_dtype == "string":
            return inputs
        if cast_dtype == "string":
            return self._to_string_cast(inputs)
        return self._from_string_cast(inputs, cast_dtype)

    @staticmethod
    def _numeric_cast(inputs: Tensor, cast_dtype: str) -> Tensor:
        """
        Casts a numeric tensor to the desired (non-string) dtype.

        :param inputs: Input numeric tensor
        :param cast_dtype: Dtype to cast to.
        :returns: Tensor cast to the desired dtype.
        """
        return tf.cast(inputs, cast_dtype)

    def _cast(self, inputs: Tensor, cast_dtype: str) -> Tensor:
        """
        Casts inputs to the desired dtype.

        :param inputs: Input tensor.
        :param cast_dtype: Dtype to cast to.
        :returns: Tensor cast to the desired dtype.
        """
        if inputs.dtype.name == "string" or cast_dtype == "string":
            # If input tensor is a string tensor, or we are casting to a string,
            # we need to use the string_cast function.
            return self._string_cast(inputs, cast_dtype)
        else:
            return self._numeric_cast(inputs, cast_dtype)

    def _force_cast_to_compatible_numeric_type(
        self, inputs: Tensor, constant: Union[float, int]
    ) -> Tuple[Tensor, Tensor]:
        """
        Casts an input tensor and a single constant to compatible tensors.

        If the provided input is a float, create the constant tensor as a float of the
        same precision. If the provided input is an integer, check if the constant is
        non-floating, and if so, create the constant tensor as an integer of the same
        precision. If the constant is floating, cast the input to a float with the same
        precision as its integer dtype and create the constant tensor likewise.

        :param inputs: Input numeric tensor
        :param constant: The constant to cast to the compatible dtype.
        :returns: Tuple of tensors cast to compatible types
        """
        if inputs.dtype.is_floating:
            if isinstance(constant, float):
                return inputs, tf.constant(constant, dtype=inputs.dtype)
            return inputs, tf.constant(float(constant), dtype=inputs.dtype)
        if inputs.dtype.is_integer:
            if isinstance(constant, int):
                return inputs, tf.constant(constant, dtype=inputs.dtype)
            if isinstance(constant, float) and constant.is_integer():
                return inputs, tf.constant(int(constant), dtype=inputs.dtype)
            if isinstance(constant, float):
                precision = inputs.dtype.size * 8
                return (
                    self._cast(inputs, f"float{precision}"),
                    tf.constant(constant, dtype=f"float{precision}"),
                )
        raise TypeError(
            "inputs must be a numeric tensor and constant must be a numeric value."
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
            if (
                cast_dtype is not None
                and self.compatible_dtypes is not None
                and cast_dtype not in [dtype.name for dtype in self.compatible_dtypes]
            ):
                raise ValueError(
                    f"""input_dtype {cast_dtype} is not a compatible dtype for
                    this layer. Compatible dtypes are {[
                        dtype.name for dtype in self.compatible_dtypes
                    ]}."""
                )
        else:
            cast_dtype = self._output_dtype

        if cast_dtype is not None:
            if tf.is_tensor(tensors):
                return (
                    self._cast(tensors, cast_dtype)
                    if tensors.dtype.name != cast_dtype
                    else tensors
                )
            return [
                self._cast(inp, cast_dtype) if inp.dtype.name != cast_dtype else inp
                for inp in tensors
            ]
        return tensors

    def cast_input_tensors(
        self, inputs: Union[Tensor, List[Tensor]]
    ) -> Union[Tensor, List[Tensor]]:
        """
        Casts the input tensors to the given input dtype, if specified. All tensors are
        cast to this. This might not be ideal, there may be layers where some inputs are
        expected to be different types. In these cases, the subclass should
        implement the cast_input_tensors method.

        :param inputs: The input tensor(s) to the layer.
        :returns: The input tensor(s) cast to the desired input_dtype.
        """
        return self._cast_input_output_tensors(tensors=inputs, ingress=True)

    def cast_output_tensors(
        self, outputs: Union[Tensor, List[Tensor]]
    ) -> Union[Tensor, List[Tensor]]:
        """
        Casts the output tensors to the given output dtype, if specified. All tensors
        are cast to this. This might not be ideal, there may be layers where some
        outputs are expected to be different types. In these cases, the subclass should
        implement the cast_output_tensors method.

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
        for inp in inputs:
            if (
                self.compatible_dtypes is not None
                and inp.dtype not in self.compatible_dtypes
            ):
                raise TypeError(
                    f"""Input tensor with dtype {inp.dtype.name}
                    is not a compatible dtype for this layer.
                    Compatible dtypes are {[
                        dtype.name for dtype in self.compatible_dtypes
                    ]}."""
                )

    @allow_single_or_multiple_tensor_input
    def call(self, inputs: Iterable[Tensor], **kwargs) -> Union[Tensor, List[Tensor]]:
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
        self, inputs: Union[Tensor, List[Tensor]], **kwargs
    ) -> Union[Tensor, List[Tensor]]:
        """
        The internal call method that should be implemented by the layer.

        :param inputs: The input tensor(s) to the layer.
        :returns: The output tensor(s) of the layer.
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
