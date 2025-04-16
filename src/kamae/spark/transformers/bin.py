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
import tensorflow as tf
from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import (
    ByteType,
    DataType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
)

from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_scalar_transform
from kamae.tensorflow.layers import BinLayer
from kamae.utils import get_condition_operator

from .base import BaseTransformer


class BinParams(Params):
    """
    Mixin class containing parameters needed for Bin transform layers.
    """

    conditionOperators = Param(
        Params._dummy(),
        "conditionOperators",
        "Operators to use in condition: eq, neq, lt, gt, leq, geq",
        typeConverter=TypeConverters.toListString,
    )

    binValues = Param(
        Params._dummy(),
        "binValues",
        "Float values to compare to input column",
        typeConverter=TypeConverters.toListFloat,
    )

    binLabels = Param(
        Params._dummy(),
        "binLabels",
        "Bin labels to use when binning",
        typeConverter=TypeConverters.toList,
    )

    defaultLabel = Param(
        Params._dummy(),
        "defaultLabel",
        "Default label to use when binning",
        typeConverter=TypeConverters.identity,
    )

    def _check_params_size(self, param_name: str, param_value: List[Any]) -> None:
        """
        Checks that the length of the given parameter is the same as the length of
        the other parameters.

        Used to ensure that the parameters are consistent with each other.

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
                if len(param_value) != len(self.getOrDefault(name)):
                    raise ValueError(
                        f"""{param_name} must have the same length as {name} but got
                        {len(param_value)} and {len(self.getOrDefault(name))}"""
                    )

    def setConditionOperators(self, value: List[str]) -> "BinParams":
        """
        Sets the conditionOperators parameter.

        :param value: List of string values describing the operator to use in condition:
        - eq
        - neq
        - lt
        - gt
        - leq
        - geq
        :returns: Instance of class mixed in.
        """
        allowed_operators = ["eq", "neq", "lt", "gt", "leq", "geq"]
        if any([v not in allowed_operators for v in value]):
            raise ValueError(
                f"""All conditionOperators must be one of {allowed_operators},
                but got {value}"""
            )
        self._check_params_size("conditionOperators", value)
        return self._set(conditionOperators=value)

    def setBinValues(self, value: List[float]) -> "BinParams":
        """
        Sets the binValues parameter.

        :param value: List of float values to compare to input column
        :returns: Instance of class mixed in.
        """
        self._check_params_size("binValues", value)
        return self._set(binValues=value)

    def setBinLabels(self, value: List[Union[float, int, str]]) -> "BinParams":
        """
        Sets the binLabels parameter.

        :param value: List of string values use when binning.
        :returns: Instance of class mixed in.
        """
        self._check_params_size("binLabels", value)
        return self._set(binLabels=value)

    def setDefaultLabel(self, value: Union[float, int, str]) -> "BinParams":
        """
        Sets the defaultLabel parameter.

        :param value: Default label to use when binning.
        :returns: Instance of class mixed in.
        """
        return self._set(defaultLabel=value)

    def getConditionOperators(self) -> List[str]:
        """
        Gets the conditionOperators parameter.

        :returns: List of string values describing the operator to use in condition:
        - eq
        - neq
        - lt
        - gt
        - leq
        - geq
        """
        return self.getOrDefault(self.conditionOperators)

    def getBinValues(self) -> List[float]:
        """
        Gets the binValues parameter.

        :returns: List of float values to compare to input column
        """
        return self.getOrDefault(self.binValues)

    def getBinLabels(self) -> List[Union[float, int, str]]:
        """
        Gets the binLabels parameter.

        :returns: List of string values use when binning.
        """
        return self.getOrDefault(self.binLabels)

    def getDefaultLabel(self) -> Union[float, int, str]:
        """
        Gets the defaultLabel parameter.

        :returns: Default label to use when binning.
        """
        return self.getOrDefault(self.defaultLabel)


class BinTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    BinParams,
):
    """
    Bin Spark Transformer for use in Spark pipelines.
    This transformer performs a binning operation on a column in a Spark dataframe.

    The binning operation is performed by comparing the input column to a list of
    values using a list of operators. The bin label corresponding to the first
    condition that evaluates to True is returned.

    If no conditions evaluate to True, the default label is returned.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        conditionOperators: Optional[List[str]] = None,
        binValues: Optional[List[float]] = None,
        binLabels: Optional[List[Union[float, int, str]]] = None,
        defaultLabel: Optional[Union[float, int, str]] = None,
        layerName: Optional[str] = None,
    ) -> None:
        """
        Initializes a BinTransformer transformer.

        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param conditionOperators: List of string values describing the operator to
        use in condition:
        - eq
        - neq
        - lt
        - gt
        - leq
        - geq
        :param binValues: Float values to compare to input column.
        :param binLabels: Bin labels to use when binning.
        :param defaultLabel: Default label to use when binning.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :returns: None - class instantiated.
        """
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return [
            FloatType(),
            DoubleType(),
            ByteType(),
            ShortType(),
            IntegerType(),
            LongType(),
        ]

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

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the bin transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
         performs the binning operation.
        """
        return BinLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            condition_operators=self.getConditionOperators(),
            bin_values=self.getBinValues(),
            bin_labels=self.getBinLabels(),
            default_label=self.getDefaultLabel(),
        )
