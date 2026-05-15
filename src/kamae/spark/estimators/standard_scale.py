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
from kamae.spark.transformers import StandardScaleTransformer
from kamae.spark.utils import construct_nested_elements_for_scaling

from .base import BaseEstimator


class StandardScaleEstimator(
    BaseEstimator,
    SingleInputSingleOutputParams,
):
    """
    Standard scaler estimator for use in Spark pipelines.
    This estimator is used to calculate the mean and standard deviation of the input
    feature column. When fit is called it returns a StandardScaleTransformer
    which can be used to standardize/transform additional features.

    WARNING: If the input is an array, we assume that the array has a constant
    shape across all rows.
    """

    jit_compatible = True

    _compatible_dtypes = [FloatType(), DoubleType()]
    _params = {**MASK_VALUE_PARAMS, **SAMPLE_FRACTION_PARAMS}

    def _fit(self, dataset: DataFrame) -> "StandardScaleTransformer":
        """
        Fits the StandardScaleEstimator estimator to the given dataset.
        Calculates the mean and standard deviation of the input feature column and
        returns a StandardScaleTransformer with the mean and standard deviation set.

        :param dataset: Pyspark dataframe to fit the estimator to.
        :returns: StandardScaleTransformer instance with mean & standard deviation set.
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

        mean_cols = [
            F.mean(
                F.when(
                    F.col(f"element_struct.element_{i}") == F.lit(self.getMaskValue()),
                    F.lit(None),
                ).otherwise(F.col(f"element_struct.element_{i}"))
            ).alias(f"mean_{i}")
            for i in range(1, array_size + 1)
        ]

        stddev_cols = [
            F.stddev_pop(
                F.when(
                    F.col(f"element_struct.element_{i}") == F.lit(self.getMaskValue()),
                    F.lit(None),
                ).otherwise(F.col(f"element_struct.element_{i}"))
            ).alias(f"stddev_{i}")
            for i in range(1, array_size + 1)
        ]

        metric_cols = mean_cols + stddev_cols

        mean_and_stddev_dict = (
            dataset.select(element_struct).agg(*metric_cols).first().asDict()
        )
        mean = [mean_and_stddev_dict[f"mean_{i}"] for i in range(1, array_size + 1)]
        stddev = [mean_and_stddev_dict[f"stddev_{i}"] for i in range(1, array_size + 1)]

        return StandardScaleTransformer(
            inputCol=self.getInputCol(),
            outputCol=self.getOutputCol(),
            layerName=self.getLayerName(),
            inputDtype=self.getInputDtype(),
            outputDtype=self.getOutputDtype(),
            mean=mean,
            stddev=stddev,
            maskValue=self.getMaskValue(),
        )
