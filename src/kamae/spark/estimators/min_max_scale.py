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
from typing import Optional

import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, DoubleType, FloatType

from kamae.params.shared_specs import MASK_VALUE_PARAMS, SAMPLE_FRACTION_PARAMS
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.transformers import MinMaxScaleTransformer
from kamae.spark.utils import construct_nested_elements_for_scaling

from .base import BaseEstimator


class MinMaxScaleEstimator(
    BaseEstimator,
    SingleInputSingleOutputParams,
):
    """
    Min max estimator for use in Spark pipelines.
    This estimator is used to calculate the min and max of the input
    feature column. When fit is called it returns a MinMaxScaleTransformer
    which can be used to standardize/transform additional features.

    WARNING: If the input is an array, we assume that the array has a constant
    shape across all rows.
    """

    jit_compatible = True

    _compatible_dtypes = [FloatType(), DoubleType()]
    _params = {**MASK_VALUE_PARAMS, **SAMPLE_FRACTION_PARAMS}

    def _fit(self, dataset: DataFrame) -> "MinMaxScaleTransformer":
        """
        Fits the MinMaxScaleEstimator estimator to the given dataset.
        Calculates the min and max of the input feature column and
        returns a MinMaxScaleTransformer with the min and max set.

        :param dataset: Pyspark dataframe to fit the estimator to.
        :returns: MinMaxScaleTransformer instance with min & max set.
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

        element_struct = construct_nested_elements_for_scaling(
            column=input_col,
            column_datatype=input_column_type,
            array_dim=array_size,
        )

        min_cols = [
            F.min(
                F.when(
                    F.col(f"element_struct.element_{i}") == F.lit(self.getMaskValue()),
                    F.lit(None),
                ).otherwise(F.col(f"element_struct.element_{i}"))
            ).alias(f"min_{i}")
            for i in range(1, array_size + 1)
        ]

        max_cols = [
            F.max(
                F.when(
                    F.col(f"element_struct.element_{i}") == F.lit(self.getMaskValue()),
                    F.lit(None),
                ).otherwise(F.col(f"element_struct.element_{i}"))
            ).alias(f"max_{i}")
            for i in range(1, array_size + 1)
        ]

        metric_cols = min_cols + max_cols

        min_and_max_dict = (
            dataset.select(element_struct).agg(*metric_cols).first().asDict()
        )
        min_vals = [min_and_max_dict[f"min_{i}"] for i in range(1, array_size + 1)]
        max_vals = [min_and_max_dict[f"max_{i}"] for i in range(1, array_size + 1)]

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
