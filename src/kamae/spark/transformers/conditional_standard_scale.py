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

import numpy as np
import pyspark.sql.functions as F
import tensorflow as tf
from pyspark import keyword_only
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, DataType, DoubleType, FloatType

from kamae.spark.params import (
    SingleInputSingleOutputParams,
    StandardScaleSkipZerosParams,
)
from kamae.spark.transformers.standard_scale import StandardScaleParams
from kamae.spark.utils.transform_utils import single_input_single_output_array_transform
from kamae.tensorflow.layers import ConditionalStandardScaleLayer

from .base import BaseTransformer


class ConditionalStandardScaleTransformer(
    BaseTransformer,
    StandardScaleParams,
    StandardScaleSkipZerosParams,
    SingleInputSingleOutputParams,
):
    """
    Conditional standard scaler transformer for use in Spark pipelines.
    This is used to standardize/transform the input column using the mean and
    the standard deviation.
    The skip_zeros parameter allows to apply the standard scaling process
    only when input is not equal to zero. If equal to zero, it will remain zero in
    the output value as it was in the input value.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        layerName: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        mean: Optional[List[float]] = None,
        stddev: Optional[List[float]] = None,
        skipZeros: bool = False,
        epsilon: float = 0,
    ) -> None:
        """
        Initializes a ConditionalStandardScaleParams transformer.
        It differs from the default StandardScaleParams in that it gives
        more control over the standard scaling process by allowing the user
        to specify a mask to be used during the fit and transform process, and
        to specify whether to skip zeros during the transform process.

        :param inputCol: Input column name to standardize.
        :param outputCol: Output column name.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param mean: List of mean values corresponding to the input column.
        :param stddev: List of standard deviation values corresponding to the
        input column.
        :param skipZeros: If True, during spark transform and keras inference,
        do not apply the scaling when the values to scale are equal to zero.
        :param epsilon: Small value to add to conditional check of zeros. Valid only
        when skipZeros is True. Defaults to 0.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(skipZeros=False, epsilon=0)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.

        :returns: List of compatible data types for the layer.
        """
        return [FloatType(), DoubleType()]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset using the mean and standard deviation
        to standardize the input column. If a mask value is set, it is used
        to ignore elements in the dataset with that value, and they will remain
        unchanged in the standardization process. If skipZeros is set to True,
        it also ignores elements with value equal to zero in the standardization
        process.

        :param dataset: Pyspark dataframe to transform.
        :returns: Pyspark dataframe with the input column standardized,
         named as the output column.
        """
        original_input_datatype = self.get_column_datatype(dataset, self.getInputCol())
        if not isinstance(original_input_datatype, ArrayType):
            input_col = F.array(F.col(self.getInputCol()))
            input_datatype = ArrayType(original_input_datatype)
        else:
            input_col = F.col(self.getInputCol())
            input_datatype = original_input_datatype

        shift = F.array([F.lit(m) for m in self.getMean()])
        scale = F.array([F.lit(1.0 / s if s != 0 else 0.0) for s in self.getStddev()])
        if self.getSkipZeros():
            eps = self.getEpsilon()
            func = lambda x: F.transform(  # noqa: E731
                x,
                lambda y, i: F.when(
                    # x != (0 +- eps)
                    F.abs(y) > F.lit(eps),
                    (y - F.lit(shift)[i]) * F.lit(scale)[i],
                ).otherwise(0),
            )
        else:
            func = lambda x: F.transform(  # noqa: E731
                x,
                lambda y, i: (y - F.lit(shift)[i]) * F.lit(scale)[i],
            )
        output_col = single_input_single_output_array_transform(
            input_col=input_col,
            input_col_datatype=input_datatype,
            func=func,
        )
        if not isinstance(original_input_datatype, ArrayType):
            output_col = output_col.getItem(0)
        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the standard scaler transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter
         that performs the standardization.
        """
        np_mean = np.array(self.getMean())
        np_variance = np.array(self.getStddev()) ** 2
        return ConditionalStandardScaleLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            mean=np_mean,
            variance=np_variance,
            skip_zeros=self.getSkipZeros(),
            epsilon=self.getEpsilon(),
        )
