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

import pyspark.sql.functions as F
import tensorflow as tf
from pyspark import keyword_only
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import ArrayType, DataType, DoubleType, FloatType

from kamae.spark.params import MultiInputSingleOutputParams
from kamae.spark.utils import multi_input_single_output_array_transform
from kamae.tensorflow.layers import CosineSimilarityLayer

from .base import BaseTransformer


class CosineSimilarityTransformer(
    BaseTransformer,
    MultiInputSingleOutputParams,
):
    """
    Cosine Similarity Spark Transformer for use in Spark pipelines.
    This transformer computes the cosine similarity between two array columns.
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
        Initializes a CosineSimilarityTransformer transformer.

        :param inputCols: Input column names. Must be two columns.
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
        return [FloatType(), DoubleType()]

    def setInputCols(self, value: List[str]) -> "CosineSimilarityTransformer":
        """
        Sets the inputCols parameter. Ensures that there are only two input columns.

        :param value: List of input column names.
        :returns: Class instance.
        """
        if len(value) != 2:
            raise ValueError(
                f"""Expected 2 input columns, received {len(value)}
                input columns instead."""
            )
        return self._set(inputCols=value)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which is the cosine similarity between the two input columns.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        input_col_names = self.getInputCols()
        input_cols = [F.col(c) for c in input_col_names]

        # Check both columns are arrays.
        datatypes = [
            self.get_column_datatype(dataset=dataset, column_name=input_col_name)
            for input_col_name in input_col_names
        ]
        if not all([isinstance(datatype, ArrayType) for datatype in datatypes]):
            raise TypeError(
                f"""Expected input columns to be of type ArrayType,
                received {datatypes} instead."""
            )

        # Compute dot product and the norms of the two arrays. The arrays are
        # represented by the "input_0" and "input_1" elements in the zipped array.
        def dot_product(x: Column) -> Column:
            return F.aggregate(
                x,
                F.lit(0.0),
                lambda acc, y: acc + y[input_col_names[0]] * y[input_col_names[1]],
            )

        def norm(x: Column, col_name: str) -> Column:
            return F.sqrt(
                F.aggregate(
                    x, F.lit(0.0), lambda acc, y: acc + y[col_name] * y[col_name]
                )
            )

        # If the norms are zero, then we match the tensorflow behavior and return 0.0.
        output_col = multi_input_single_output_array_transform(
            input_cols=input_cols,
            input_col_datatypes=datatypes,
            input_col_names=input_col_names,
            func=lambda x: F.coalesce(
                dot_product(x)
                / (norm(x, input_col_names[0]) * norm(x, input_col_names[1])),
                F.lit(0.0),
            ),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the cosine similarity transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
         computes the cosine similarity between two arrays.
        """
        return CosineSimilarityLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            axis=-1,
            keepdims=True,
        )
