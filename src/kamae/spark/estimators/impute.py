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
from typing import List, Optional, Union

import pyspark.sql.functions as F
from pyspark import keyword_only
from pyspark.sql.types import ArrayType, DataType, DoubleType, FloatType

from kamae.spark.params import (
    ImputeMethodParams,
    MaskValueParams,
    SingleInputSingleOutputParams,
)
from kamae.spark.transformers import ImputeTransformer
from kamae.spark.utils import flatten_nested_arrays

from .base import BaseEstimator


class ImputeEstimator(
    BaseEstimator,
    SingleInputSingleOutputParams,
    MaskValueParams,
    ImputeMethodParams,
):
    """
    Imputation estimator for use in Spark pipelines.
    This estimator is used to calculate the chosen statistic of the input
    feature column. When fit is called it returns a ImputeTransformer
    which can be used to impute either the mean or median of a column.
    Rows are not included in the calculation of the statistic when they are
    either null or equal to the supplied mask value.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        maskValue: Optional[Union[float, int, str]] = None,
        imputeMethod: Optional[str] = None,
    ) -> None:
        """
        Initializes a ImputeEstimator estimator.
        Sets all parameters to given inputs.

        :param inputCol: Input column name to standardize.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
         in the keras model. If not set, we use the uid of the Spark transformer.
        :param maskValue: Value which to ignore, in addition to nulls, when
        computing imputation statistic.
        This is also the value that is imputed over in TF at inference.
        :param imputeMethod: Method by which to compute the value to be imputed.
        Valid values are "mean" or "median".
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(imputeMethod="mean")
        self.valid_impute_methods = ["mean", "median"]
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

    def _fit(self, dataset) -> "ImputeTransformer":
        """
        Fits the ImputeEstimator estimator to the given dataset.
        Calculates the imputation statistic of the input feature column and
        returns an ImputeTransformer with the statistic set.

        :param dataset: Pyspark dataframe to fit the estimator to.
        :returns: ImputeTransformer instance with impute value set.
        """
        imputeMethod = self.getImputeMethod()

        if imputeMethod == "mean":
            estimator_fn = F.mean
        elif imputeMethod == "median":
            estimator_fn = F.median

        input_column_type = self.get_column_datatype(dataset, self.getInputCol())
        input_col_an_array = isinstance(input_column_type, ArrayType)

        # If the column input is an array then we need to flatten and explode it to
        # calculate the impute value
        if input_col_an_array:
            # Flatten the array to a single array
            flattened_array_col = flatten_nested_arrays(
                column=F.col(self.getInputCol()), column_data_type=input_column_type
            )
            processed_input_col = F.explode(flattened_array_col).alias(
                self.uid + "_input_col"
            )
        else:
            processed_input_col = F.col(self.getInputCol()).alias(
                self.uid + "_input_col"
            )

        imputeValue = (
            dataset.select(processed_input_col)
            .select(
                F.when(
                    (F.col(self.uid + "_input_col") == F.lit(self.getMaskValue()))
                    | (F.col(self.uid + "_input_col").isNull()),
                    None,
                )
                .otherwise(F.col(self.uid + "_input_col"))
                .alias("input_col")
            )
            .agg(estimator_fn("input_col"))
            .collect()[0][0]
        )

        return ImputeTransformer(
            inputCol=self.getInputCol(),
            outputCol=self.getOutputCol(),
            layerName=self.getLayerName(),
            inputDtype=self.getInputDtype(),
            outputDtype=self.getOutputDtype(),
            imputeValue=imputeValue,
            maskValue=self.getMaskValue(),
        )
