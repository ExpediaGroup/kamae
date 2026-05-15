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

# pylint: disable=unused-argument
# pylint: disable=invalid-name
# pylint: disable=too-many-ancestors
# pylint: disable=no-member
import base64
from typing import Any, Callable, List, Optional, Union

import dill
import pyspark.sql.functions as F
import tensorflow as tf
from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, DataType, StructField, StructType

from kamae.keras.tensorflow.layers import LambdaFunctionLayer
from kamae.keras.tensorflow.utils.typing import Tensor
from kamae.spark.params import (
    MultiInputMultiOutputParams,
    MultiInputSingleOutputParams,
    SingleInputMultiOutputParams,
    SingleInputSingleOutputParams,
)

from .base import BaseTransformer


class LambdaFunctionParams(Params):
    """
    Mixin class containing parameters needed for the LambdaFunctionTransformer.
    """

    function = Param(
        Params._dummy(),
        "function",
        "Function to apply to the input column. Serialized by dill.",
        typeConverter=TypeConverters.toString,
    )

    functionReturnTypes = Param(
        Params._dummy(),
        "functionReturnTypes",
        """List of return types of the lambda function. Should be length 1 for
        single output. Used to understand the UDF return type.""",
        typeConverter=TypeConverters.toListString,
    )

    def setFunction(self, value: Callable[[Any], Any]) -> "LambdaFunctionParams":
        """
        Sets the lambda function to apply to the input column.

        :param value: Lambda function to apply to the input column.
        :returns: Class instance.
        """
        dill_bytes = dill.dumps(value)
        return self._set(function=base64.b64encode(dill_bytes).decode("utf-8"))

    def setFunctionReturnTypes(self, value: List[DataType]) -> "LambdaFunctionParams":
        """
        Sets the return type of the lambda function.

        :param value: Return type of the lambda function.
        :returns: Class instance.
        """
        serialised_dtypes = []
        for return_type in value:
            dill_bytes = dill.dumps(return_type)
            serialised_dtypes.append(base64.b64encode(dill_bytes).decode("utf-8"))

        return self._set(functionReturnTypes=serialised_dtypes)

    def getFunction(self) -> Callable[[Any], Any]:
        """
        Gets the lambda function to apply to the input column.

        :returns: Lambda function to apply to the input column.
        """
        utf_str = self.getOrDefault(self.function)
        return dill.loads(base64.b64decode(utf_str))

    def getFunctionReturnTypes(self) -> List[DataType]:
        """
        Gets the return type of the lambda function.

        :returns: Return type of the lambda function.
        """
        utf_str_list = self.getOrDefault(self.functionReturnTypes)
        return [dill.loads(base64.b64decode(utf_str)) for utf_str in utf_str_list]


class LambdaFunctionTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    SingleInputMultiOutputParams,
    MultiInputSingleOutputParams,
    MultiInputMultiOutputParams,
    LambdaFunctionParams,
):
    """
    Spark Transformer that applies tensorflow lambda functions to the input column(s).

    The provided function must either take a single tensor as input or take a list of
    tensors as input. Same for the output, either single tensor or list of tensors.
    In the case of multiple input/output, it is assumed that the order matches the
    order of the input/output columns.

    It can only use tensorflow methods, and reference them with `tf.`.

    An example of a valid lambda function is:

    ```python
    def my_tf_fn(x):
        return tf.square(x) - tf.math.log(x)
    ```

    Note: No validation is done on the lambda function. It is up to the user to
    ensure the lambda function is valid and works as expected. The lambda function
    must be serializable by dill: https://github.com/uqfoundation/dill

    The function will be used within a User Defined Function (UDF) in Spark.
    You need to provide the return type for the UDF computation as a string.
    E.g. "float", "array<string>" etc. If you require the operation to be quicker
    (in Spark), consider composing it from other transformers, that are written in
    native Spark functions.
    """

    _compatible_dtypes = None
    _keras_layer_class = None

    def get_keras_layer(self):
        return LambdaFunctionLayer(
            function=self.getFunction(),
            name=self.getLayerName(),
            input_dtype=self.getInputKerasDtype(),
            output_dtype=self.getOutputKerasDtype(),
        )

    def setInputCol(self, value: str) -> "LambdaFunctionTransformer":
        """
        Sets the parameter inputCol to the given string value.
        Only allows setting if inputCols is not set.

        :param value: String to set the inputCol parameter to.
        :returns: Instance of class mixed in.
        """
        if self.isSet("inputCols"):
            raise ValueError("inputCols is already set. Cannot set inputCol as well.")
        return self._set(inputCol=value)

    def setInputCols(self, value: List[str]) -> "LambdaFunctionTransformer":
        """
        Sets the parameter inputCols to the given list of strings.
        Only allows setting if inputCol is not set. Only allows multiple inputs.

        :param value: List of strings to set the inputCols parameter to.
        :returns: Instance of class mixed in.
        """
        if len(value) < 2:
            raise ValueError(
                """inputCols must be set with more than one column.
                Use inputCol for single input."""
            )
        if self.isSet("inputCol"):
            raise ValueError("inputCol is already set. Cannot set inputCols as well.")
        return self._set(inputCols=value)

    def setOutputCol(self, value: str) -> "LambdaFunctionTransformer":
        """
        Sets the parameter outputCol to the given string value.
        Only allows setting if outputCols is not set.

        :param value: String to set the outputCol parameter to.
        :returns: Instance of class mixed in.
        """
        if self.isSet("outputCols"):
            raise ValueError("outputCols is already set. Cannot set outputCol as well.")
        return self._set(outputCol=value)

    def setOutputCols(self, value: List[str]) -> "LambdaFunctionTransformer":
        """
        Sets the parameter outputCols to the given list of strings.
        Only allows setting if outputCol is not set. Only allows multiple outputs.

        :param value: List of strings to set the outputCols parameter to.
        :returns: Instance of class mixed in.
        """
        if len(value) < 2:
            raise ValueError(
                """outputCols must be set with more than one column.
                Use outputCol for single output."""
            )
        if self.isSet("outputCol"):
            raise ValueError("outputCol is already set. Cannot set outputCols as well.")
        return self._set(outputCols=value)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return None

    def _validate_params(self) -> None:
        """
        Validates the parameters of the LambdaFunctionTransformer.
        """
        if not (self.isDefined("function") and self.isDefined("functionReturnTypes")):
            raise ValueError("function and functionReturnTypes must be set.")

        if not (self.isDefined("inputCol") or self.isDefined("inputCols")):
            raise ValueError("inputCol or inputCols must be set.")

        if not (self.isDefined("outputCol") or self.isDefined("outputCols")):
            raise ValueError("outputCol or outputCols must be set.")

        output_col_names = self._get_output_cols()
        if len(output_col_names) != len(self.getFunctionReturnTypes()):
            raise ValueError(
                "Number of output columns must match number of function return types."
            )

    def _get_input_cols(self) -> List[str]:
        """
        Gets the input columns for the transformer.

        :returns: List of input column names.
        """
        return (
            self.getInputCols() if self.isDefined("inputCols") else [self.getInputCol()]
        )

    def _get_output_cols(self) -> List[str]:
        """
        Gets the output columns for the transformer.

        :returns: List of output column names.
        """
        return (
            self.getOutputCols()
            if self.isDefined("outputCols")
            else [self.getOutputCol()]
        )

    @staticmethod
    def _apply_udf_func_to_dataset(
        dataset: DataFrame,
        func: Callable[[Union[Tensor, List[Tensor]]], Any],
        input_col_names: List[str],
        output_col_names: List[str],
        function_return_types: List[DataType],
    ) -> DataFrame:
        """
        Gets and applies the udf to the dataset. If the output is a single column, the
        output is directly applied to the dataset. If the output is multiple columns,
        a struct column is created and then the columns are extracted.

        :param dataset: Pyspark dataframe to transform.
        :param func: Keras function.
        :param input_col_names: List of input column names.
        :param output_col_names: List of output column names.
        :param function_return_types: List of return types of the lambda function.
        :returns: Transformed pyspark dataframe.
        """
        if len(output_col_names) == 1:
            # Single output scenario
            output_col = F.udf(func, function_return_types[0])(
                *[F.col(c) for c in input_col_names]
            )
            return dataset.withColumn(output_col_names[0], output_col)
        else:
            # Multi-output scenario. UDF cannot return multiple columns, so
            # return a struct type and then extract the columns.
            udf_return_type = StructType(
                [
                    StructField(name=output_col, dataType=function_return_types[i])
                    for i, output_col in enumerate(output_col_names)
                ]
            )
            output_struct_col = F.udf(func, udf_return_type)(
                *[F.col(c) for c in input_col_names]
            )
            for output_col in output_col_names:
                dataset = dataset.withColumn(output_col, output_struct_col[output_col])
            return dataset

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which applies `function` to the input column(s).

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        self._validate_params()

        input_col_names = self._get_input_cols()
        output_col_names = self._get_output_cols()
        function_return_types = self.getFunctionReturnTypes()

        def tf_function_wrapper(
            fn: Callable[[Union[Tensor, List[Tensor]]], Tensor],
        ) -> Callable[[Union[Tensor, List[Tensor]]], Any]:
            """
            Wraps the tensorflow function, so we can use it within a Spark UDF.

            Specifically:

            - Handles the fact that Spark UDFs require multiple inputs to be passed as
            arguments, not in a list. However, the tensorflow function expects multiple
            inputs to be passed as a list.
            - Wraps scalar inputs in a list before creating the tensor. The tensor also
            has a single batch dimension. Therefore, a scalar value has shape (1, 1)
            and an array has shape (1, N).
            - Converts the output tensor to a numpy value and extracts the single batch.
            If value is a list of size 1, return the single value.
            - If the output tensor is a string, decodes the bytes to a string.

            :param fn: Keras function.
            :returns: Function that can be used within a Spark UDF.
            """

            def wrapper(*args: Any) -> Union[Any, tuple[Any, ...]]:
                # Wrap args in a list if they are not already a list. This ensures
                # that scalar inputs have shape (1, 1) in tensorflow.
                args = [a if isinstance(a, list) else [a] for a in args]
                # Call the function with the input tensors.
                if len(args) == 1:
                    tf_val: Union[Tensor, list[Tensor]] = fn(tf.constant([args[0]]))
                else:
                    tf_val: Union[Tensor, list[Tensor]] = fn(
                        [tf.constant([a]) for a in args]
                    )

                # Single output scenario
                if len(function_return_types) == 1:
                    if "string" in function_return_types[0].simpleString():
                        # If string output, decode the bytes to a string.
                        output = tf_val.numpy().astype(str).tolist()[0]
                    else:
                        output = tf_val.numpy().tolist()[0]
                    if len(output) == 1 and not isinstance(
                        function_return_types[0], ArrayType
                    ):
                        return output[0]
                    return output
                # Multi output scenario
                else:
                    tf_outputs = []
                    for i, tensor in enumerate(tf_val):
                        if "string" in function_return_types[i].simpleString():
                            # If string output, decode the bytes to a string.
                            output = tensor.numpy().astype(str).tolist()[0]
                        else:
                            output = tensor.numpy().tolist()[0]
                        if len(output) == 1 and not isinstance(
                            function_return_types[i], ArrayType
                        ):
                            # If the output is a scalar value, extract it from the list.
                            tf_outputs.append(output[0])
                        else:
                            tf_outputs.append(output)
                    # Return a tuple, which will be unpacked into multiple columns.
                    return tuple(tf_outputs)

            return wrapper

        func = tf_function_wrapper(self.getFunction())

        return self._apply_udf_func_to_dataset(
            dataset=dataset,
            func=func,
            input_col_names=input_col_names,
            output_col_names=output_col_names,
            function_return_types=function_return_types,
        )
