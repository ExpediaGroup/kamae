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
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType

from kamae.spark.params import SingleInputMultiOutputParams
from kamae.spark.utils import single_input_single_output_array_transform
from kamae.tensorflow.layers import ArraySplitLayer

from .base import BaseTransformer


class ArraySplitTransformer(
    BaseTransformer,
    SingleInputMultiOutputParams,
):
    """
    ArraySplit Spark Transformer for use in Spark pipelines.
    This transformer splits an array column into multiple columns.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCols: Optional[List[str]] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
    ) -> None:
        """
        Initialize a ArraySplitTransformer transformer.

        :param inputCol: Input column name.
        :param outputCols: Output column names.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column(s) to after
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
        return None

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column for each output column equal
        to the value of the input column at the given index.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        output_cols = []
        for i, column in enumerate(self.getOutputCols()):
            output_col = single_input_single_output_array_transform(
                input_col=F.col(self.getInputCol()),
                input_col_datatype=input_datatype,
                func=lambda x: F.element_at(x, i + 1),
            )
            output_cols.append(output_col.alias(self.getOutputCols()[i]))
        original_columns = [F.col(c) for c in dataset.columns]
        select_cols = original_columns + output_cols
        return dataset.select(select_cols)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for that unstacks the input tensor and reshapes
        to the original shape.

        :returns: Tensorflow keras layer with name equal to the layerName parameter
        that slices the input tensors.
        """
        return ArraySplitLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            axis=-1,
        )
