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
from typing import Any, List, Optional, Union

import pyspark.sql.functions as F
from pyspark.ml.param import TypeConverters
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import (
    ByteType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
)

from kamae.keras.core.layers import BinLayer
from kamae.params import ParamSpec
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_scalar_transform
from kamae.utils import get_condition_operator

from .base import BaseTransformer


def _validate_condition_operators(self: Any, value: List[str]) -> List[str]:
    """
    Validates that all condition operators are allowed values.
    """
    if value is None:
        return value
    allowed_operators = ["eq", "neq", "lt", "gt", "leq", "geq"]
    if any([v not in allowed_operators for v in value]):
        raise ValueError(
            f"""All conditionOperators must be one of {allowed_operators},
            but got {value}"""
        )
    _check_params_size(self, "conditionOperators", value)
    return value


def _validate_bin_values(self: Any, value: List[float]) -> List[float]:
    """
    Validates that bin values have consistent length with other params.
    """
    if value is None:
        return value
    _check_params_size(self, "binValues", value)
    return value


def _validate_bin_labels(
    self: Any, value: List[Union[float, int, str]]
) -> List[Union[float, int, str]]:
    """
    Validates that bin labels have consistent length with other params.
    """
    if value is None:
        return value
    _check_params_size(self, "binLabels", value)
    return value


def _check_params_size(self: Any, param_name: str, param_value: List[Any]) -> None:
    """
    Checks that the length of the given parameter is the same as the length of
    the other parameters.

    Used to ensure that the parameters are consistent with each other.

    :param self: Transformer instance.
    :param param_name: Name of the parameter to check.
    :param param_value: Value of the parameter to check.
    :returns: None
    :raises ValueError: If the length of the given parameter is not the same as
    the length of the other parameters.
    """
    names_to_check = ["conditionOperators", "binValues", "binLabels"]
    names_to_check.remove(param_name)
    for name in names_to_check:
        if self.isDefined(name):
            other_value = self.getOrDefault(name)
            if other_value is not None and len(param_value) != len(other_value):
                raise ValueError(
                    f"""{param_name} must have the same length as {name} but got
                    {len(param_value)} and {len(other_value)}"""
                )


class BinTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    Bin Spark Transformer for use in Spark pipelines.
    This transformer performs a binning operation on a column in a Spark dataframe.

    The binning operation is performed by comparing the input column to a list of
    values using a list of operators. The bin label corresponding to the first
    condition that evaluates to True is returned.

    If no conditions evaluate to True, the default label is returned.
    """

    jit_compatible = True

    _compatible_dtypes = [
        FloatType(),
        DoubleType(),
        ByteType(),
        ShortType(),
        IntegerType(),
        LongType(),
    ]
    _keras_layer_class = BinLayer
    _params = {
        "conditionOperators": ParamSpec(
            spark_typeconverter=TypeConverters.toListString,
            default=None,
            doc="Operators to use in condition: eq, neq, lt, gt, leq, geq",
            validator=_validate_condition_operators,
        ),
        "binValues": ParamSpec(
            spark_typeconverter=TypeConverters.toListFloat,
            default=None,
            doc="Float values to compare to input column",
            validator=_validate_bin_values,
        ),
        "binLabels": ParamSpec(
            spark_typeconverter=TypeConverters.toList,
            default=None,
            doc="Bin labels to use when binning",
            validator=_validate_bin_labels,
        ),
        "defaultLabel": ParamSpec(
            spark_typeconverter=TypeConverters.identity,
            default=None,
            doc="Default label to use when binning",
        ),
    }

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which uses the binValues and binLabels parameters to bin the input column
        according to the conditionOperators parameter.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        condition_operators = [
            get_condition_operator(c) for c in self.getConditionOperators()
        ]
        bin_values = self.getBinValues()
        bin_labels = self.getBinLabels()

        def bin_func(x: Column) -> Column:
            """
            Perfoms the binning of a given column x.
            :param x: Column to bin.
            :returns: Binned column.
            """
            bin_output = F.lit(self.getDefaultLabel())
            # Loop through the conditions.
            # Reverse the list of conditions so that we start from the last condition
            # and work backwards. This ensures that the first condition that is met
            # is the one that is used.
            conds = zip(condition_operators[::-1], bin_values[::-1], bin_labels[::-1])

            for cond_op, value, label in conds:
                bin_output = F.when(cond_op(x, value), F.lit(label)).otherwise(
                    bin_output,
                )
            return bin_output

        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: bin_func(x),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)
