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
from operator import add
from typing import List, Optional

import tensorflow as tf
from pyspark import keyword_only
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    ByteType,
    DataType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
)

from kamae.spark.params import (
    MathFloatConstantParams,
    MultiInputSingleOutputParams,
    SingleInputSingleOutputParams,
)
from kamae.spark.utils import multi_input_single_output_scalar_transform
from kamae.tensorflow.layers import SumLayer

from .base import BaseTransformer


class SumTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    MultiInputSingleOutputParams,
    MathFloatConstantParams,
):
    """
    SumLayer Spark Transformer for use in Spark pipelines.
    This transformer sums a column with a constant or another column.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        inputCols: Optional[List[str]] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        mathFloatConstant: Optional[float] = None,
    ) -> None:
        """
        Initializes an SumTransformer transformer.

        :param inputCol: Input column name. Only used if inputCols is not specified.
        If specified, we sum this column by the mathFloatConstant.
        :param inputCols: Input column names.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column(s) to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param mathFloatConstant: Optional constant to sum. If not provided,
        then two input columns are required.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(mathFloatConstant=None)
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
        which is the same as the column with name `inputCol`.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        input_cols = self.get_multiple_input_cols(
            constant_param_name="mathFloatConstant",
        )
        # input_cols can contain either actual columns or lit(constants). In order to
        # determine the datatype of the input columns, we select them from the dataset
        # first.
        input_col_names = dataset.select(input_cols).columns
        input_col_datatypes = [
            self.get_column_datatype(dataset=dataset.select(input_cols), column_name=c)
            for c in input_col_names
        ]
        output_col = multi_input_single_output_scalar_transform(
            input_cols=input_cols,
            input_col_names=input_col_names,
            input_col_datatypes=input_col_datatypes,
            func=lambda x: reduce(add, [x[c] for c in input_col_names]),
        )

        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the sum transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
         performs a sum operation.
        """
        return SumLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            addend=self.getMathFloatConstant(),
        )
