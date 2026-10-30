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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyspark.sql.functions as F
from pyspark import keyword_only
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import ArrayType, DataType, DoubleType, FloatType

from kamae.keras.core.backend import ALL_BACKENDS
from kamae.spark.params import (
    MaskValueParams,
    SampleFractionParams,
    SingleInputSingleOutputParams,
)
from kamae.spark.transformers import MinMaxScaleTransformer
from kamae.spark.utils import (
    construct_nested_elements_for_scaling,
    get_array_nesting_level_and_element_dtype,
)

from .base import BaseEstimator


class MinMaxScaleEstimator(
    BaseEstimator,
    SampleFractionParams,
    SingleInputSingleOutputParams,
    MaskValueParams,
):
    """
    Min max estimator for use in Spark pipelines.
    This estimator is used to calculate the min and max of the input
    feature column. When fit is called it returns a MinMaxScaleTransformer
    which can be used to standardize/transform additional features.

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
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        maskValue: Optional[float] = None,
        sampleFraction: Optional[float] = None,
        useFitSample: bool = False,
    ) -> None:
        """
        Initializes a MinMaxScaleEstimator estimator.
        Sets all parameters to given inputs.

        :param inputCol: Input column name to standardize.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
         in the keras model. If not set, we use the uid of the Spark transformer.
        :param maskValue: Value to use for masking. If set, these values will be ignored
        during the computation of the min and max values.
        :param sampleFraction: Fraction of data to sample for statistics
        estimation (exclusive 0.0-1.0). Default None (no sampling).
        :param useFitSample: If True, fit on the enclosing pipeline's shared sample
        when fitSampleFraction is set. Default False.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(maskValue=None, sampleFraction=None, useFitSample=False)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return [FloatType(), DoubleType()]

    def _fit(self, dataset: DataFrame) -> "MinMaxScaleTransformer":
        """
        Fits the MinMaxScaleEstimator estimator to the given dataset.
        Calculates the min and max of the input feature column and
        returns a MinMaxScaleTransformer with the min and max set.

        :param dataset: Pyspark dataframe to fit the estimator to.
        :returns: MinMaxScaleTransformer instance with min & max set.
        """
        return self._fit_by_aggregation(dataset, persist=False)

    def can_fit_by_aggregation(self, dataset: DataFrame) -> bool:
        """
        The min and max are one aggregation, except for nested
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
        Builds the min and max aggregations of each element of the input feature
        column, ignoring elements equal to the mask value.

        :param dataset: Pyspark dataframe to fit the estimator to.
        :param prefix: Prefix of the helper column name and aggregation aliases.
        :returns: Tuple of the helper columns and the aggregations.
        """
        input_column_type = self.get_column_datatype(dataset, self.getInputCol())
        if not isinstance(input_column_type, ArrayType):
            input_col = F.array(F.col(self.getInputCol()))
            input_column_type = ArrayType(input_column_type)
        else:
            input_col = F.col(self.getInputCol())

        # Collect a single row to driver and get the length.
        # We assume all subsequent rows have the same length.
        array_size = np.array((dataset.select(input_col).first()[0])).shape[-1]
        struct_col_name = f"{prefix}element_struct"
        element_struct = construct_nested_elements_for_scaling(
            column=input_col,
            column_datatype=input_column_type,
            array_dim=array_size,
        )
        elements = [
            F.when(
                F.col(f"{struct_col_name}.element_{i}") == F.lit(self.getMaskValue()),
                F.lit(None),
            ).otherwise(F.col(f"{struct_col_name}.element_{i}"))
            for i in range(1, array_size + 1)
        ]
        min_cols = [
            F.min(element).alias(f"{prefix}min_{i}")
            for i, element in enumerate(elements, start=1)
        ]
        max_cols = [
            F.max(element).alias(f"{prefix}max_{i}")
            for i, element in enumerate(elements, start=1)
        ]
        # Carry the array size to `fit_from_row`.
        array_size_col = F.lit(array_size).alias(f"{prefix}array_size")
        return {struct_col_name: element_struct}, min_cols + max_cols + [array_size_col]

    def fit_from_row(
        self, row: Dict[str, Any], prefix: str
    ) -> "MinMaxScaleTransformer":
        """
        Builds the fitted transformer from the min and max aggregated by
        `fit_aggregations`.

        :param row: The aggregation result row, as a dict of alias -> value.
        :param prefix: The prefix given to `fit_aggregations`.
        :returns: MinMaxScaleTransformer instance with min & max set.
        """
        array_size = row[f"{prefix}array_size"]
        min_vals = [row[f"{prefix}min_{i}"] for i in range(1, array_size + 1)]
        max_vals = [row[f"{prefix}max_{i}"] for i in range(1, array_size + 1)]
        return MinMaxScaleTransformer(
            inputCol=self.getInputCol(),
            outputCol=self.getOutputCol(),
            layerName=self.getLayerName(),
            inputDtype=self.getInputDtype(),
            outputDtype=self.getOutputDtype(),
            min=min_vals,
            max=max_vals,
            maskValue=self.getMaskValue(),
        )
