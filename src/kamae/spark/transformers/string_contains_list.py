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
from typing import List, Optional

import pyspark.sql.functions as F
import tensorflow as tf
from pyspark import keyword_only
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import DataType, StringType

from kamae.spark.params import (
    ConstantStringArrayParams,
    NegationParams,
    SingleInputSingleOutputParams,
)
from kamae.spark.transformers.base import BaseTransformer
from kamae.spark.utils import single_input_single_output_scalar_transform
from kamae.tensorflow.layers import StringContainsListLayer


class StringContainsListTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    NegationParams,
    ConstantStringArrayParams,
):
    """
    String contains list Spark Transformer for use in Spark pipelines.
    This transformer performs a string contains operation on the input column over all
    constants in the passed constantStringArray.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        constantStringArray: Optional[List[str]] = None,
        negation: bool = False,
        layerName: Optional[str] = None,
    ) -> None:
        """
        Intializes a StringContainsListTransformer transformer.

        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param constantStringArray: String constant array to use in string contains list
        operation.
        :param negation: Whether to negate the string contains list operation.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(negation=False)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return [StringType()]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which contains the result of the string contains operation.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )

        if not self.isDefined("constantStringArray"):
            raise ValueError("constantStringArray must be defined.")

        def string_contains_list(
            x: Column, string_list: List[str], negation: bool
        ) -> Column:
            contains_cols = [
                x.contains(string_constant) for string_constant in string_list
            ]
            col_expr = reduce(lambda y, z: y | z, contains_cols)
            return col_expr if not negation else ~col_expr

        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: string_contains_list(
                x=x,
                string_list=self.getConstantStringArray(),
                negation=self.getNegation(),
            ),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the StringContainsLayer transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
         performs a string contains operation.
        """

        if not self.isDefined("constantStringArray"):
            raise ValueError("constantStringArray must be defined.")

        return StringContainsListLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            negation=self.getNegation(),
            string_constant_list=self.getConstantStringArray(),
        )
