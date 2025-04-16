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
from operator import and_
from typing import List, Optional

import pyspark.sql.functions as F
import tensorflow as tf
from pyspark import keyword_only
from pyspark.sql import DataFrame
from pyspark.sql.types import BooleanType, DataType

from kamae.spark.params import MultiInputSingleOutputParams
from kamae.spark.utils import multi_input_single_output_scalar_transform
from kamae.tensorflow.layers import LogicalAndLayer

from .base import BaseTransformer


class LogicalAndTransformer(
    BaseTransformer,
    MultiInputSingleOutputParams,
):
    """
    Logical And Spark Transformer for use in Spark pipelines.
    This transformer performs an element-wise logical and operation on multiple columns.
    """

    @keyword_only
    def __init__(
        self,
        inputCols: Optional[List[str]] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
    ) -> None:
        """
        Initializes a LogicalAndTransformer transformer.

        :param inputCols: Input column names.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column(s) to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
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
        return [BooleanType()]

    def setInputCols(self, value: List[str]) -> "LogicalAndTransformer":
        """
        Sets the inputCols parameter. Raises an error if the value is a list of
        length 1.

        :param value: List of input column names.
        :returns: Instance of class with inputCols parameter set.
        """
        if len(value) == 1:
            raise ValueError("inputCols must be a list of length > 1")
        return self._set(inputCols=value)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which is the logical and of the input columns.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        input_col_names = self.getInputCols()
        input_cols = [F.col(c) for c in input_col_names]
        input_col_datatypes = [
            self.get_column_datatype(dataset=dataset, column_name=c)
            for c in input_col_names
        ]
        output_col = multi_input_single_output_scalar_transform(
            input_cols=input_cols,
            input_col_datatypes=input_col_datatypes,
            input_col_names=input_col_names,
            func=lambda x: reduce(and_, [x[c] for c in input_col_names]),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the logical and transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
         performs a logical and operation.
        """
        return LogicalAndLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
        )
