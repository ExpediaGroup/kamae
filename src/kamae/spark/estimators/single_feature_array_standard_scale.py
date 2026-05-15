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

from typing import Dict, List, Optional

import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, DataType, DoubleType, FloatType

from kamae.params.shared_specs import MASK_VALUE_PARAMS, SAMPLE_FRACTION_PARAMS
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.transformers import StandardScaleTransformer
from kamae.spark.utils import flatten_nested_arrays

from .base import BaseEstimator


class SingleFeatureArrayStandardScaleEstimator(
    BaseEstimator,
    SingleInputSingleOutputParams,
):
    """
    Single feature array standard scaler estimator for use in Spark pipelines.
    This estimator is used to calculate the mean and standard deviation of the input
    feature column when it is an array where all the elements represent the same
    feature. An example would be a sequence of trip durations or booking windows in
    a traveller's session. When fit is called it returns a StandardScaleTransformer
    which can be used to standardize/transform additional features, where the mean
    and standard deviation are calculated across all elements in all the arrays.
    """

    jit_compatible = True

    _compatible_dtypes = [FloatType(), DoubleType()]
    _params = {**MASK_VALUE_PARAMS, **SAMPLE_FRACTION_PARAMS}

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return [FloatType(), DoubleType()]

    def _fit(self, dataset: DataFrame) -> "StandardScaleTransformer":
        """
        Fits the SingleFeatureArrayStandardScaleEstimator estimator to the given
        dataset. Calculates the mean and standard deviation of the input feature column
        and returns a StandardScaleTransformer with the mean and standard deviation set.

        All rows are assumed to be of the same length. The mask value which is set in
        the estimator is used to ignore certain values in the process of calculating
        the mean and stddev.
        :param dataset: Pyspark dataframe to fit the estimator to.
        :returns: StandardScaleTransformer instance with mean & standard deviation set.
        """

        input_column_type = self.get_column_datatype(dataset, self.getInputCol())
        if not isinstance(input_column_type, ArrayType):
            raise ValueError(
                f"""Input column {self.getInputCol()} must be of ArrayType.
                        Got {input_column_type} instead."""
            )

        # Collect a single row to driver and get the length.
        # We assume all subsequent rows have the same length.
        array_size = np.array((dataset.select(self.getInputCol()).first()[0])).shape[-1]

        # Flatten the array to a single array.
        # Will do nothing if the array is not nested.
        flattened_array_col = flatten_nested_arrays(
            column=F.col(self.getInputCol()), column_data_type=input_column_type
        )

        mean_and_stddev_dict: Dict[str, float] = (
            dataset.select(F.explode(flattened_array_col).alias(self.getInputCol()))
            .withColumn(
                "mask",
                F.when(
                    F.col(self.getInputCol()) == F.lit(self.getMaskValue()), 1
                ).otherwise(0),
            )
            .filter(F.col("mask") == F.lit(0))
            .agg(
                F.mean(self.getInputCol()).alias("mean"),
                F.stddev_pop(self.getInputCol()).alias("stddev"),
            )
            .first()
            .asDict()
        )
        mean: List[float] = [mean_and_stddev_dict["mean"] for _ in range(array_size)]
        stddev: List[float] = [
            mean_and_stddev_dict["stddev"] for _ in range(array_size)
        ]

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
