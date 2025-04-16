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

from typing import List, Optional

import pyspark.sql.functions as F
import tensorflow as tf
from pyspark import keyword_only
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, DataType, IntegerType, StringType

from kamae.spark.params import PadValueParams, SingleInputSingleOutputParams
from kamae.spark.utils import (
    ordinal_array_encode_udf,
    single_input_single_output_array_udf_transform,
)
from kamae.tensorflow.layers import OrdinalArrayEncodeLayer

from .base import BaseTransformer


class OrdinalArrayEncodeTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    PadValueParams,
):
    """
    Transformer that encodes an array of strings into an array of integers.

    The transformer will map each unique string in the array to an integer,
    according to the order in which they appear in the array. It will also
    ignore the pad value if specified.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        padValue: Optional[str] = None,
    ) -> None:
        """
        Initialises the OrdinalArrayEncodeTransformer
        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column(s) to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        :param padValue: The value to be considered as padding. Defaults to `None`.
        :returns: None
        """
        super().__init__()
        self._setDefault(padValue=None)
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
            StringType(),
        ]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Performs the ordinal encoding on the input dataset.
        Example:
         dataset = spark.Dataframe(
            [
                ['a', 'a', 'a', 'b', 'c', '-1', '-1', '-1'],
                ['x', 'x', 'x', 'x', 'y', 'z', '-1', '-1'],
            ],
            'input_col'
         )
         Output: spark.Dataframe(
            [
                ['a', 'a', 'a', 'b', 'c', '-1', '-1', '-1'],
                ['x', 'x', 'x', 'x', 'y', 'z', '-1', '-1'],
            ],
            [
                [0, 0, 0, 1, 2, -1, -1, -1],
                [0, 0, 0, 0, 1, 2, -1, -1],
            ],
            'input_col', 'output_col'
        )
        :param dataset: The input dataframe.
        :returns: Transformed pyspark dataframe.
        """
        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        if not isinstance(input_datatype, ArrayType):
            raise ValueError(
                f"""Input column {self.getInputCol()} must be of ArrayType.
                        Got {input_datatype} instead."""
            )

        output_col = single_input_single_output_array_udf_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: ordinal_array_encode_udf(x, self.getPadValue()),
            udf_return_element_datatype=IntegerType(),
        )

        return dataset.withColumn(
            self.getOutputCol(),
            output_col,
        )

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer that performs the ordinal array encoding.

        :returns: Tensorflow keras layer with name equal to the layerName parameter
        that performs the ordinal array encoding operation.
        """
        return OrdinalArrayEncodeLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            pad_value=self.getPadValue(),
            axis=-1,
        )
