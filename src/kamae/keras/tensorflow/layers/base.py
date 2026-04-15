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
TensorFlow-specific base layer that extends BaseLayer with string operations.

This base layer requires the TensorFlow backend and provides string casting in addition
to the numeric operations from BaseLayer.
"""

from abc import abstractmethod
from functools import reduce
from typing import Any, List, Optional, Union

import tensorflow as tf

from kamae.keras.core.backend import require_tensorflow
from kamae.keras.core.layers.base import BaseLayer
from kamae.tensorflow.typing import Tensor


class TfBaseLayer(BaseLayer):
    """
    TensorFlow-specific base layer with string casting support.

    Inherits numeric operations from BaseLayer and adds:
    - String to/from numeric casting
    - Boolean string parsing
    - TensorFlow dtype compatibility checking
    """

    def __init__(
        self,
        name: Optional[str] = None,
        input_dtype: Optional[str] = None,
        output_dtype: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the TfBaseLayer.

        :param name: Name of the layer, defaults to `None`.
        :param input_dtype: Input data type of the layer. If specified, inputs will be
        cast to this data type before any computation is performed. Defaults to `None`.
        :param output_dtype: Output data type of the layer. Defaults to `None`. If
        specified, the output will be cast to this data type before being returned.
        """
        # Fail fast if not on TensorFlow backend
        require_tensorflow()

        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )

        # Boolean string parsing configuration
        self.true_bool_strings = ["true", "t", "yes", "y", "1"]
        self.false_bool_strings = ["false", "f", "no", "n", "0"]

    @property
    @abstractmethod
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        """
        List of compatible TensorFlow data types for the layer.
        If the computation can be performed on any data type, return None.

        Note: This overrides BaseLayer to return TensorFlow dtype objects
        instead of strings, for compatibility with existing TF layers.

        :returns: List of compatible tf.dtypes.DType objects or None.
        """
        raise NotImplementedError

    def _string_to_bool_cast(self, inputs: Tensor) -> Tensor:
        """
        Casts a string tensor to a bool tensor.

        Recognizes common boolean string representations:
        - True: "true", "t", "yes", "y", "1"
        - False: "false", "f", "no", "n", "0"

        :param inputs: Input string tensor
        :returns: Bool tensor.
        :raises TypeError: If inputs is not a string tensor
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

        :param inputs: Input float tensor
        :returns: String tensor.
        """
        # This gives 1.145000 -> "1.145" and 2.00000 -> "2".
        # We need to add a decimal point to the second example.
        shortest_float_string = tf.strings.as_string(inputs, shortest=True)

        # Find strings without decimal points
        no_decimal = tf.logical_not(
            tf.strings.regex_full_match(
                shortest_float_string, "-?\\d*\\.\\d*"  # noqa W605
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
        :raises TypeError: If inputs is not a string tensor or cast_dtype is unsupported
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

    def _cast(self, inputs: Tensor, cast_dtype: str) -> Tensor:
        """
        Casts inputs to the desired dtype.

        Overrides BaseLayer._cast to add string support.

        :param inputs: Input tensor.
        :param cast_dtype: Dtype to cast to.
        :returns: Tensor cast to the desired dtype.
        """
        if inputs.dtype.name == "string" or cast_dtype == "string":
            # If input tensor is a string tensor, or we are casting to a string,
            # we need to use the string_cast function.
            return self._string_cast(inputs, cast_dtype)
        else:
            # Use parent class numeric casting
            return super()._cast(inputs, cast_dtype)

    def _check_input_dtypes_compatible(self, inputs: List[Tensor]) -> None:
        """
        Checks if the input tensors are compatible with the compatible_dtypes of the
        layer.

        Overrides BaseLayer to work with tf.dtypes.DType objects.

        :param inputs: The input tensor(s) to the layer.
        :raises ValueError: If the input tensors are not compatible with the
        compatible_dtypes of the layer.
        :returns: None
        """
        if self.compatible_dtypes is None:
            # Any dtype is compatible
            return

        for inp in inputs:
            if inp.dtype not in self.compatible_dtypes:
                raise TypeError(
                    f"Input tensor with dtype {inp.dtype.name} "
                    f"is not a compatible dtype for this layer. "
                    f"Compatible dtypes are {[dt.name for dt in self.compatible_dtypes]}."
                )
