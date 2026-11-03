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
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyspark.sql.functions as F
from numpy.typing import NDArray
from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import ArrayType, DataType, DoubleType, FloatType

from kamae.keras.core.backend import ALL_BACKENDS
from kamae.spark.params import (
    NanFillValueParams,
    SampleFractionParams,
    SingleInputSingleOutputParams,
    StandardScaleSkipZerosParams,
)
from kamae.spark.transformers import ConditionalStandardScaleTransformer
from kamae.spark.utils import (
    construct_nested_elements_for_scaling,
    get_array_nesting_level_and_element_dtype,
)
from kamae.utils import get_condition_operator

from .base import BaseEstimator


class ConditionalStandardScaleEstimatorParams(Params):
    """
    Mixin class containing conditional standard scale parameters,
    needed for single feature array scaler layers.
    """

    scalingFunction = Param(
        Params._dummy(),
        "scalingFunction",
        """
        The name of the scaling function to use during spark fit function,
        to estimate the mean and standard deviation. Defaults to 'standard'.
        """,
        typeConverter=TypeConverters.toString,
    )

    maskCols = Param(
        Params._dummy(),
        "maskCols",
        """
        Columns on which to apply the mask condition and the mask value
        during moments calculation. Defaults to None.
        """,
        typeConverter=TypeConverters.toListString,
    )

    maskOperators = Param(
        Params._dummy(),
        "maskOperators",
        "Operators to use in masking conditions: eq, neq, lt, gt, leq, geq",
        typeConverter=TypeConverters.toListString,
    )

    maskValues = Param(
        Params._dummy(),
        "maskValues",
        """
        Values applied to the respective maskCol with the respective maskOperator
        to make the value ignored during moments calculation.
        """,
        typeConverter=TypeConverters.toListFloat,
    )

    relevanceCol = Param(
        Params._dummy(),
        "relevanceCol",
        """
        The name of the relevance column to use during the calculation of the moments.
        """,
        typeConverter=TypeConverters.toString,
    )

    def setScalingFunction(
        self,
        value: str,
    ) -> "ConditionalStandardScaleEstimatorParams":
        """
        Sets the scalingFunction parameter.

        :param value: String value to indicate which scaling function to use.
        :returns: Instance of class mixed in.
        """
        if value.lower() == "standard":
            return self._set(scalingFunction=value.lower())
        elif value.lower() == "binary":
            return self._set(scalingFunction=value.lower())
        else:
            raise ValueError(f"Unknown scaling function: {value}.")

    def getScalingFunction(self) -> str:
        """
        Gets the scalingFunction parameter.

        :returns: Boolean value of the scalingFunction value.
        """
        return self.getOrDefault(self.scalingFunction)

    def setMaskCols(
        self, value: List[str]
    ) -> "ConditionalStandardScaleEstimatorParams":
        """
        Sets the maskCols parameter.

        :param value: Columns to use as the mask columns.
        :returns: Instance of class mixed in.
        """
        return self._set(maskCols=value)

    def getMaskCols(self) -> List[str]:
        """
        Gets the maskCols parameter.
        :returns: List of string values of the mask column.
        """
        return self.getOrDefault(self.maskCols)

    def setMaskOperators(
        self, value: List[str]
    ) -> "ConditionalStandardScaleEstimatorParams":
        """
        Sets the maskOperators parameter.

        :param value: String value describing the operator to use in condition:
        - eq
        - neq
        - lt
        - gt
        - leq
        - geq
        :returns: Instance of class mixed in.
        """
        allowed_operators = ["eq", "neq", "lt", "gt", "leq", "geq"]
        for v in value:
            if v not in allowed_operators:
                raise ValueError(
                    f"conditionOperator must be one of {allowed_operators}, "
                    f"but got {value}"
                )
        return self._set(maskOperators=value)

    def getMaskOperators(self) -> List[str]:
        """
        Gets the maskOperators parameter.

        :returns: List of string values describing the operators to use in conditions:
        - eq
        - neq
        - lt
        - gt
        - leq
        - geq
        """
        return self.getOrDefault(self.maskOperators)

    def setMaskValues(
        self, value: List[float]
    ) -> "ConditionalStandardScaleEstimatorParams":
        """
        Sets the maskValues parameter.

        :param value: List of float values to use as the mask value.
        :returns: Instance of class mixed in.
        """
        return self._set(maskValues=value)

    def getMaskValues(self) -> List[float]:
        """
        Gets the maskValues parameter.

        :returns: List of float values of the mask values.
        """
        return self.getOrDefault(self.maskValues)

    def setRelevanceCol(self, value: str) -> "ConditionalStandardScaleEstimatorParams":
        """
        Sets the relevanceCol parameter.

        :param value: String value to use as the relevance column.
        :returns: Instance of class mixed in.
        """
        return self._set(relevanceCol=value)

    def getRelevanceCol(self) -> str:
        """
        Gets the relevanceCol parameter.

        :returns: String value of the relevance column.
        """
        return self.getOrDefault(self.relevanceCol)


class ConditionalStandardScaleEstimator(
    BaseEstimator,
    SampleFractionParams,
    SingleInputSingleOutputParams,
    ConditionalStandardScaleEstimatorParams,
    StandardScaleSkipZerosParams,
    NanFillValueParams,
):
    """
    Conditional standard scaler estimator for use in Spark pipelines.
    This is used to calculate the mean and standard deviation with masking,
    and then to standardize/transform the input column using the mean and
    standard deviation, optionally skipping the standardization of inputs
    equal to zero.
    The mask columns, mask values and mask conditions are used to calculate
    the moments only when all the masking conditions are satisfied.
    The skip_zeros parameter allows to apply both the scaling and the transformation
    only when input is not equal to zero. If equal to zero, it will remain zero in
    the output value as it was in the input value.
    It is also possible to specify a non-standard scaling function using the
    scalingFunction parameter.
    When fit is called it returns a ConditionalStandardScaleTransformer
    which can be used to standardize/transform the input data.

    WARNING: If the input is an array, we assume that the array has a constant
    shape across all rows.
    """

    supported_backends = ALL_BACKENDS
    jit_compatible = True

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        layerName: Optional[str] = None,
        scalingFunction: str = "standard",
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        maskCols: Optional[List[str]] = None,
        maskOperators: Optional[List[str]] = None,
        maskValues: Optional[List[float]] = None,
        relevanceCol: Optional[str] = None,
        skipZeros: bool = False,
        epsilon: float = 0,
        nanFillValue: Optional[float] = None,
        sampleFraction: Optional[float] = None,
        useFitSample: bool = False,
    ) -> None:
        """
        Initializes a ConditionalStandardScaleEstimator estimator.
        Sets all parameters to given inputs.

        :param inputCol: Input column name to standardize.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param scalingFunction: The scaling function to use: 'standard', 'binary'.
        :param maskCols: Columns on which to apply the mask values. Defaults to None.
        :param maskOperators: Operators to use in each masking condition:
        eq, neq, lt, gt, leq, geq. Defaults to 'neq'.
        :param maskValues: Values applied to the maskCols which makes the value
        ignored in the calculation of the moments. Defaults to None.
        :param relevanceCol: The name of the relevance column to use during spark
        fit function.
        :param skipZeros: If True, during spark transform and keras inference,
        do not apply the scaling when the values to scale are equal to zero.
        :param epsilon: Small value to add to conditional check of zeros. Valid only
        when skipZeros is True. Defaults to 0.
        :param nanFillValue: Value to fill NaNs with after scaling. It is important
        to use it if epsilon filters out all the values. Defaults to None.
        :param sampleFraction: Fraction of data to sample for statistics
        estimation (exclusive 0.0-1.0). Default None (no sampling).
        :param useFitSample: If True, fit on the enclosing pipeline's shared sample
        when fitSampleFraction is set. Default False.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(
            scalingFunction="standard",
            maskCols=None,
            maskOperators=None,
            maskValues=None,
            relevanceCol=None,
            skipZeros=False,
            epsilon=0,
            nanFillValue=None,
            sampleFraction=None,
            useFitSample=False,
        )
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        Returns the compatible data types for transformer.
        :returns: List of compatible data types.
        """
        return [DoubleType(), FloatType()]

    def _get_binary_moments(self, f: NDArray, n: NDArray) -> (NDArray, NDArray):
        """
        Calculates the moments for a binary variable.

        :param f: Number of samples where feature is one/true.
        :param n: Number of samples.
        :returns:
            - mean - The mean of the distribution.
            - stddev - The stddev of the distribution.
        """
        true_ratio = np.where(n <= 0, 0, f / n)
        variance = np.where(
            (n - 1) <= 0,
            0,
            (f * pow(1 - true_ratio, 2) + (n - f) * pow(0 - true_ratio, 2)) / (n - 1),
        )
        mean = 1 - true_ratio
        stddev = np.sqrt(variance)
        return mean, stddev

    def _validate_mask_ops(self) -> None:
        """
        Validates the mask operators.
        Mask columns, operators and values must be set together
        and must have the same length.
        """
        arr1 = self.getMaskCols()
        arr2 = self.getMaskOperators()
        arr3 = self.getMaskValues()
        if any(arr is not None for arr in [arr1, arr2, arr3]):
            if any(arr is None or len(arr) != len(arr1) for arr in [arr1, arr2, arr3]):
                raise ValueError(
                    "Mask columns, operators and values must be set together "
                    "and must have the same length."
                )
        return

    def _fit(self, dataset: DataFrame) -> "ConditionalStandardScaleTransformer":
        """
        Fits the ConditionalStandardScaleEstimator estimator to the given dataset.
        Calculates the moments of the input feature column (see `fit_aggregations`)
        and returns a ConditionalStandardScaleTransformer with them set.

        :param dataset: Pyspark dataframe to fit the estimator to.
        :returns: ConditionalStandardScaleTransformer instance with
        mean & standard deviation set.
        """
        # Masks are applied to the dataset before calculating the moments. The mask
        # condition inside the aggregations (needed when fitting together with other
        # estimators) is then true on every remaining row.
        mask = self._mask_condition(dataset)
        if mask is not None:
            dataset = dataset.filter(mask)
        return self._fit_by_aggregation(dataset)

    def can_fit_by_aggregation(self, dataset: DataFrame) -> bool:
        """
        Both scaling functions fit as one aggregation, except for nested
        array inputs: their elements are exploded into rows (see
        `construct_nested_elements_for_scaling`), which changes the rows, so they
        keep their own fit.

        :param dataset: The input dataset.
        :returns: True unless the input column is a nested array.
        """
        nested_lvl, _ = get_array_nesting_level_and_element_dtype(
            self.get_column_datatype(dataset, self.getInputCol())
        )
        return nested_lvl <= 1

    def fit_aggregations(
        self, dataset: DataFrame, prefix: str
    ) -> Tuple[Dict[str, Column], List[Column]]:
        """
        Builds the aggregations of the moments of the input feature column. Masks
        are applied as a condition inside the aggregations, so only the rows meeting
        them are counted.

        :param dataset: Pyspark dataframe to fit the estimator to.
        :param prefix: Prefix of the helper column name and aggregation aliases.
        :returns: Tuple of the helper columns and the aggregations.
        """
        input_column_dtype = self.get_column_datatype(dataset, self.getInputCol())
        if not isinstance(input_column_dtype, ArrayType):
            input_col = F.array(F.col(self.getInputCol()))
            input_column_dtype = ArrayType(input_column_dtype)
        else:
            input_col = F.col(self.getInputCol())

        mask = self._mask_condition(dataset)
        scaling_function = self.getScalingFunction().lower()
        if scaling_function not in ["standard", "binary"]:
            raise ValueError(f"Unknown scaling function: {self.getScalingFunction()}.")
        if scaling_function == "binary" and self.getRelevanceCol() is None:
            raise ValueError("Relevance column must be set for binary scaling.")

        # Collect a single row (meeting the masks) to driver and get the length.
        # We assume all subsequent rows have the same length.
        row = (
            (dataset if mask is None else dataset.filter(mask))
            .select(input_col)
            .first()
        )
        if row is None:
            raise ValueError("No data left after application of mask conditions.")
        array_size = np.array((row[0])).shape[-1]

        # Construct the elements to calculate the moments
        struct_col_name = f"{prefix}element_struct"
        element_struct = construct_nested_elements_for_scaling(
            column=input_col,
            column_datatype=input_column_dtype,
            array_dim=array_size,
        )
        elements = [
            F.col(f"{struct_col_name}.element_{i}") for i in range(1, array_size + 1)
        ]
        if scaling_function == "standard":
            aggregations = self._standard_aggregations(elements, prefix, mask)
        else:
            aggregations = self._binary_aggregations(elements, prefix, mask)
        # Carry the array size to `fit_from_row`.
        aggregations.append(F.lit(array_size).alias(f"{prefix}array_size"))
        return {struct_col_name: element_struct}, aggregations

    def fit_from_row(
        self, row: Dict[str, Any], prefix: str
    ) -> "ConditionalStandardScaleTransformer":
        """
        Builds the fitted transformer from the moments aggregated by
        `fit_aggregations`.

        :param row: The aggregation result row, as a dict of alias -> value.
        :param prefix: The prefix given to `fit_aggregations`.
        :returns: ConditionalStandardScaleTransformer instance with
        mean & standard deviation set.
        """
        array_size = row[f"{prefix}array_size"]
        fill_val = self.getNanFillValue()
        if self.getScalingFunction().lower() == "standard":
            mean = [row[f"{prefix}mean_{i}"] for i in range(1, array_size + 1)]
            stddev = [row[f"{prefix}stddev_{i}"] for i in range(1, array_size + 1)]
            if fill_val is not None:
                mean = [fill_val if m is None else m for m in mean]
                stddev = [fill_val if s is None else s for s in stddev]
        else:
            count = [row[f"{prefix}count_{i}"] for i in range(1, array_size + 1)]
            count_ones = [
                row[f"{prefix}count_ones_{i}"] for i in range(1, array_size + 1)
            ]
            if fill_val is not None:
                count = [fill_val if c is None else c for c in count]
                count_ones = [fill_val if c is None else c for c in count_ones]
            mean, stddev = self._get_binary_moments(
                np.array(count_ones), np.array(count)
            )
            mean, stddev = mean.tolist(), stddev.tolist()
        return ConditionalStandardScaleTransformer(
            inputCol=self.getInputCol(),
            outputCol=self.getOutputCol(),
            layerName=self.getLayerName(),
            inputDtype=self.getInputDtype(),
            outputDtype=self.getOutputDtype(),
            mean=mean,
            stddev=stddev,
            skipZeros=self.getSkipZeros(),
            epsilon=self.getEpsilon(),
        )

    def _mask_condition(self, dataset: DataFrame) -> Optional[Column]:
        """
        The combined mask condition, or None without masks.

        :param dataset: Pyspark dataframe holding the mask columns.
        :returns: The mask condition.
        """
        self._validate_mask_ops()
        if self.getMaskCols() is None:
            return None
        mask_conditions = []
        for mask_col, mask_op, mask_val in zip(
            self.getMaskCols(), self.getMaskOperators(), self.getMaskValues()
        ):
            if mask_col not in dataset.columns:
                raise ValueError(f"Mask column {mask_col} not found in dataset.")
            mask_conditions.append(
                get_condition_operator(mask_op)(F.col(mask_col), mask_val)
            )
        return reduce(lambda a, b: a & b, mask_conditions)

    def _standard_aggregations(
        self, elements: List[Column], prefix: str, mask: Optional[Column]
    ) -> List[Column]:
        """
        The mean and stddev aggregations of the standard scaling function. Elements
        are only counted on rows meeting the masks, where the relevance column is
        positive (if set) and, with skipZeros, where they are not zero (+- epsilon).

        :param elements: The element columns to standardize.
        :param prefix: Prefix of the aggregation aliases.
        :param mask: The mask condition, or None.
        :returns: The mean aggregations followed by the stddev aggregations.
        """
        values = []
        for element in elements:
            conditions = []
            if self.getSkipZeros():
                # x != (0 +- eps)
                conditions.append(F.abs(element) > F.lit(self.getEpsilon()))
            if self.getRelevanceCol() is not None:
                conditions.append(F.col(self.getRelevanceCol()) > 0)
            if mask is not None:
                conditions.append(mask)
            values.append(
                F.when(reduce(lambda a, b: a & b, conditions), element)
                if conditions
                else element
            )
        mean_cols = [
            F.mean(value).alias(f"{prefix}mean_{i}")
            for i, value in enumerate(values, start=1)
        ]
        stddev_cols = [
            F.stddev_pop(value).alias(f"{prefix}stddev_{i}")
            for i, value in enumerate(values, start=1)
        ]
        return mean_cols + stddev_cols

    def _binary_aggregations(
        self, elements: List[Column], prefix: str, mask: Optional[Column]
    ) -> List[Column]:
        """
        The count aggregations of the binary scaling function, on rows meeting the
        masks. This should be used only when input is a binary variable and the
        label can be transformed into a classification task (if relevance > 0 then
        1 else 0). With f and n the counts below, the mean and stddev are:
            mean = 1 - (f/n)
            stddev = sqrt((f * pow(1-(f/n), 2) + (n-f) * pow(0-(f/n), 2)) / (n-1))
        where:
            n = sum(when(x==1, 1, 0))
            f = sum(when(x==1 && relevance > 0, 1, 0))

        :param elements: The element columns to scale.
        :param prefix: Prefix of the aggregation aliases.
        :param mask: The mask condition, or None.
        :returns: The count aggregations followed by the count-of-ones aggregations.
        """

        def where(condition: Column) -> Column:
            return condition if mask is None else (condition & mask)

        count_cols = [
            F.sum(F.when(where(element == F.lit(1)), 1).otherwise(0)).alias(
                f"{prefix}count_{i}"
            )
            for i, element in enumerate(elements, start=1)
        ]
        count_ones_cols = [
            F.sum(
                F.when(
                    where((element == F.lit(1)) & (F.col(self.getRelevanceCol()) > 0)),
                    1,
                ).otherwise(0)
            ).alias(f"{prefix}count_ones_{i}")
            for i, element in enumerate(elements, start=1)
        ]
        return count_cols + count_ones_cols
