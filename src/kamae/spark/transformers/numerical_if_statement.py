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
from typing import List, Optional

import pyspark.sql.functions as F
from pyspark import keyword_only
from pyspark.ml.param import TypeConverters
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import DoubleType, FloatType

from kamae.keras.core.layers import NumericalIfStatementLayer
from kamae.params import ParamSpec
from kamae.spark.params import (
    MultiInputSingleOutputParams,
    SingleInputSingleOutputParams,
)
from kamae.spark.utils import multi_input_single_output_scalar_transform
from kamae.utils import get_condition_operator

from .base import BaseTransformer


def _validate_condition_operator(value: str) -> str:
    """Validates conditionOperator parameter."""
    allowed_operators = ["eq", "neq", "lt", "gt", "leq", "geq"]
    if value not in allowed_operators:
        raise ValueError(
            f"conditionOperator must be one of {allowed_operators}, but got {value}"
        )
    return value


# TODO: Deprecate this in favor of IfStatementTransformer in next major release.
class NumericalIfStatementTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    MultiInputSingleOutputParams,
):
    """
    NumericalIfStatement Spark Transformer for use in Spark pipelines.
    This transformer computes an if statement between a set of numerical constants
    and columns.
    """

    jit_compatible = True

    _compatible_dtypes = [FloatType(), DoubleType()]
    _keras_layer_class = NumericalIfStatementLayer
    _params = {
        "conditionOperator": ParamSpec(
            spark_typeconverter=TypeConverters.toString,
            default=None,
            doc="Operator to use in condition: eq, neq, lt, gt, leq, geq",
            validator=_validate_condition_operator,
        ),
        "valueToCompare": ParamSpec(
            spark_typeconverter=TypeConverters.toFloat,
            default=None,
            doc="Float value to compare to input column",
        ),
        "resultIfTrue": ParamSpec(
            spark_typeconverter=TypeConverters.toFloat,
            default=None,
            doc="Float value to return if condition is true",
        ),
        "resultIfFalse": ParamSpec(
            spark_typeconverter=TypeConverters.toFloat,
            default=None,
            doc="Float value to return if condition is false",
        ),
    }

    def setInputCols(self, value: List[str]) -> "NumericalIfStatementTransformer":
        """
        Sets the inputCols parameter, throwing an error if the total length of the
        inputCols and the constants are more than 4.

        :param value: List of input column names.
        :returns: Instance of class mixed in.
        """
        optional_constant_cols = ["valueToCompare", "resultIfTrue", "resultIfFalse"]
        num_defined_constants = len(
            [c for c in optional_constant_cols if self.getOrDefault(c) is not None]
        )
        if len(value) + num_defined_constants > 4:
            raise ValueError(
                f"""Total number of input columns and constants cannot be more than 4,
                but got {len(value)} input columns and
                {num_defined_constants} constants."""
            )

        return self._set(inputCols=value)

    def _construct_input_cols(self) -> List[Column]:
        """
        Constructs the input columns for the transformer. This is fairly complex
        since the user can set any of the following:
        - valueToCompare
        - resultIfTrue
        - resultIfFalse

        If all of these are set, then the user should have provided a single `inputCol`.
        Otherwise, if any of these are not set, then the user should have provided
        `inputCols` containing the missing values. The `inputCols` should be in the
        order [valueToCompare, resultIfTrue, resultIfFalse]. But if the user has
        specified, for example, `resultIfTrue` as a constant then the
        `inputCols` should be in the order [valueToCompare, resultIfFalse].

        :returns: Tuple of 4 pyspark columns.
        """
        optional_constant_cols = ["valueToCompare", "resultIfTrue", "resultIfFalse"]
        optional_constants_defined = {
            const: self.getOrDefault(const) is not None
            for const in optional_constant_cols
        }

        if self.isDefined("inputCols"):
            # If the user has set inputCols, then some or all of the optional constants
            # are defined as input column variables
            input_cols = self.getInputCols()
            if len(input_cols) + sum(optional_constants_defined.values()) != 4:
                raise ValueError(
                    f"""Total number of input columns and constants must be 4,
                    but got {len(input_cols)} input columns and
                    {sum(optional_constants_defined.values())} constants."""
                )
            # The first input column is always the value to compare
            input_col_list = [
                F.col(input_cols[0]),
            ]
            input_col_counter = 1
            for const_col in optional_constant_cols:
                # Loop through the optional constant names.
                # If the constant is not defined then it must be an input column.
                # Otherwise, it is a literal value.
                if not optional_constants_defined[const_col]:
                    input_col_list.append(F.col(input_cols[input_col_counter]))
                    input_col_counter += 1
                else:
                    input_col_list.append(F.lit(self.getOrDefault(const_col)))
            return [
                input_col_list[0].alias(self.uid + "_inputCol"),
                input_col_list[1].alias(self.uid + "_valueToCompare"),
                input_col_list[2].alias(self.uid + "_resultIfTrue"),
                input_col_list[3].alias(self.uid + "_resultIfFalse"),
            ]
        elif self.isDefined("inputCol"):
            # If the user has set inputCol, then all the optional constants
            # must be defined.
            if not all(
                [
                    self.getOrDefault(const) is not None
                    for const in optional_constant_cols
                ]
            ):
                raise ValueError(
                    f"""Must specify all of {optional_constant_cols}"
                    if inputCol is specified."""
                )
            return [
                F.col(self.getInputCol()).alias(self.uid + "_inputCol"),
                F.lit(self.getValueToCompare()).alias(self.uid + "_valueToCompare"),
                F.lit(self.getResultIfTrue()).alias(self.uid + "_resultIfTrue"),
                F.lit(self.getResultIfFalse()).alias(self.uid + "_resultIfFalse"),
            ]
        else:
            raise ValueError("Must specify either inputCol or inputCols.")

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which is the output of an if statement with condition:

        IF `inputCol` `conditionOperator` `valueToCompare`
        THEN `resultIfTrue`
        ELSE `resultIfFalse`

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        input_cols = self._construct_input_cols()
        # input_cols can contain either actual columns or lit(constants). In order to
        # determine the datatype of the input columns, we select them from the dataset
        # first.
        input_col_names = dataset.select(input_cols).columns
        input_col_datatypes = [
            self.get_column_datatype(dataset=dataset.select(input_cols), column_name=c)
            for c in input_col_names
        ]
        condition_operator = get_condition_operator(self.getConditionOperator())

        output_col = multi_input_single_output_scalar_transform(
            input_cols=input_cols,
            input_col_names=input_col_names,
            input_col_datatypes=input_col_datatypes,
            func=lambda x: F.when(
                condition_operator(x[input_col_names[0]], x[input_col_names[1]]),
                x[input_col_names[2]],
            ).otherwise(x[input_col_names[3]]),
        )

        return dataset.withColumn(self.getOutputCol(), output_col)
