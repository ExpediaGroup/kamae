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
from typing import List

import keras
import numpy as np
import pyspark.sql.functions as F
from pyspark.ml.param import TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, DoubleType, FloatType

from kamae.keras.core.layers import ConditionalStandardScaleLayer
from kamae.params import ParamSpec
from kamae.params.shared_specs import STANDARD_SCALE_PARAMS
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils.transform_utils import single_input_single_output_array_transform

from .base import BaseTransformer


class ConditionalStandardScaleTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    Conditional standard scaler transformer for use in Spark pipelines.
    This is used to standardize/transform the input column using the mean and
    the standard deviation.
    The skip_zeros parameter allows to apply the standard scaling process
    only when input is not equal to zero. If equal to zero, it will remain zero in
    the output value as it was in the input value.

    WARNING: If the input is an array, we assume that the array has a constant
    shape across all rows.
    """

    jit_compatible = True

    _compatible_dtypes = [FloatType(), DoubleType()]
    _keras_layer_class = ConditionalStandardScaleLayer
    _params = {
        **STANDARD_SCALE_PARAMS,
        "skipZeros": ParamSpec(
            spark_typeconverter=TypeConverters.toBoolean,
            default=False,
            doc="If True, during spark transform and keras inference, do not apply "
            "the scaling when the values to scale are equal to zero.",
        ),
        "epsilon": ParamSpec(
            spark_typeconverter=TypeConverters.toFloat,
            default=0.0,
            doc="Small value to add to conditional check of zeros. Valid only when "
            "skipZeros is True.",
        ),
    }

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
        # Compute scale from variance: scale = 1/sqrt(variance)
        variance = self.getVariance()
        scale = F.array([F.lit(1.0 / (v**0.5) if v != 0 else 0.0) for v in variance])
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
