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
from pyspark.sql.types import ArrayType, DataType

from kamae.spark.params import AutoBroadcastParams, MultiInputSingleOutputParams
from kamae.spark.utils import (
    broadcast_scalar_column_to_array_with_inner_singleton_array,
    get_array_nesting_level,
    nested_arrays_zip,
    nested_transform,
)
from kamae.tensorflow.layers import ArrayConcatenateLayer

from .base import BaseTransformer


class ArrayConcatenateTransformer(
    BaseTransformer,
    MultiInputSingleOutputParams,
    AutoBroadcastParams,
):
    """
    ArrayConcatenate Spark Transformer for use in Spark pipelines.
    This transformer assembles multiple columns into a single array column.
    """

    @keyword_only
    def __init__(
        self,
        inputCols: Optional[List[str]] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        autoBroadcast: Optional[bool] = False,
    ) -> None:
        """
        Initialize a ArrayConcatenateTransformer transformer.

        :param inputCols: List of input column names.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column(s) to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param autoBroadcast: If True, the Keras transformer will broadcast scalar
        inputs to the biggest rank. Default is False.
        WARNING: This modifies only the Keras layer, not the Spark transformer!
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(autoBroadcast=False)
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

    @staticmethod
    def array_concatenate_transform(
        input_cols: List[Column],
        input_col_datatypes: List[DataType],
    ) -> Column:
        """
        Similar to the transform helpers in the utils module,but for the
        ArrayConcatenateTransformer. This function is specific to this transformer
        because whilst it still broadcasts scalars to arrays, we do not want to repeat
        the innermost array elements. Thus, each array can have different innermost size
        for concatenation. Other implementations perform elementwise computations across
        nested arrays and so require that all arrays are exactly the same shape.

        Caters for the case, where the input columns are:

        1. All scalars.
        2. All (possibly nested) arrays.
        3. A mix of scalars and (possibly nested) arrays.

        If all inputs are scalars, applies concat(array()) directly to the input
        columns. If inputs are a mixture of scalars and arrays, broadcasts any scalars
        to the size of the arrays in all N-1 dimensions, and creates an array of size 1
        in the final Nth dimension. Then zips the arrays into a single nested array
        column, stopping before the innermost (Nth) array dimension. Finally, concats
        along this Nth dimension.

        Example:

        Input 1: 1.0 (scalar)
        Input 2: [
                    [[2.0, 3.0], [4.0, 5.0]],
                    [[6.0, 7.0], [8.0, 9.0]]
                ] (nested 3D array)
        Input 3: 10.0 (scalar)
        Input 4: [
                    [[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]],
                    [[17.0, 18.0, 19.0], [20.0, 21.0, 22.0]]
                ] (nested 3D array)

        Input 1 & 3 are broadcast to the size of the arrays up to N-1th dim, giving:
        Input 1: [
                    [[1.0], [1.0]],
                    [[1.0], [1.0]]
                ]
        Input 3: [
                    [[10.0], [10.0]],
                    [[10.0], [10.0]]
                ]
        Then zipped together to give:
        [
            [
                {
                    "input1": [1.0],
                    "input2": [2.0, 3.0],
                    "input3": [10.0],
                    "input4": [11.0, 12.0, 13.0]
                },
                {
                    "input1": [1.0],
                    "input2": [4.0, 5.0],
                    "input3": [10.0],
                    "input4": [14.0, 15.0, 16.0]
                }
            ],
            [
                {
                    "input1": [1.0],
                    "input2": [6.0, 7.0],
                    "input3": [10.0],
                    "input4": [17.0, 18.0, 19.0]
                },
                {
                    "input1": [1.0],
                    "input2": [8.0, 9.0],
                    "input3": [10.0],
                    "input4": [20.0, 21.0, 22.0]
                }
            ]
        ]
        Then concats along the Nth dimension to give final output:
        [
            [
                [1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0],
                [1.0, 4.0, 5.0, 10.0, 14.0, 15.0, 16.0]
            ],
            [
                [1.0, 6.0, 7.0, 10.0, 17.0, 18.0, 19.0],
                [1.0, 8.0, 9.0, 10.0, 20.0, 21.0, 22.0]
            ]
        ]

        :param input_cols: List of input columns.
        :param input_col_datatypes: List of input column datatypes.
        :returns: Output column.
        """
        input_col_names = [f"input{i}" for i in range(len(input_cols))]
        if all(
            [not isinstance(datatype, ArrayType) for datatype in input_col_datatypes]
        ):
            # All inputs are scalars.
            # Apply the concat, wrapping each scalar in an array.
            return F.concat(*[F.array(x) for x in input_cols])

        if all([isinstance(datatype, ArrayType) for datatype in input_col_datatypes]):
            # All inputs are arrays. Zip the arrays together into a single column.
            nesting_level = get_array_nesting_level(column_dtype=input_col_datatypes[0])
            zipped_array_column = nested_arrays_zip(
                columns=input_cols,
                nest_level=nesting_level - 1,
                column_names=input_col_names,
            )
        else:
            # Inputs are a mix of scalars and arrays.
            # Broadcast the scalars to the size of the arrays.
            scalar_columns = [
                (idx, col_w_datatype[0])
                for idx, col_w_datatype in enumerate(
                    zip(input_cols, input_col_datatypes)
                )
                if not isinstance(col_w_datatype[1], ArrayType)
            ]
            array_columns_w_types = [
                (idx, col_w_datatype[0], col_w_datatype[1])
                for idx, col_w_datatype in enumerate(
                    zip(input_cols, input_col_datatypes)
                )
                if isinstance(col_w_datatype[1], ArrayType)
            ]
            # Broadcast the scalar to the size of the arrays. Use the first array column
            # to determine the size of the broadcasted scalar. Assumes all arrays are
            # of the same size in the N-1 dimensions
            broadcasted_scalars = [
                (
                    idx,
                    broadcast_scalar_column_to_array_with_inner_singleton_array(
                        scalar_column=scalar_column,
                        array_column=array_columns_w_types[0][1],
                        array_column_datatype=array_columns_w_types[0][2],
                    ),
                )
                for idx, scalar_column in scalar_columns
            ]
            # Resort the array columns and the broadcasted scalars,
            # so they match the order of the input columns.
            columns_w_idx = [
                (idx, column) for idx, column, _ in array_columns_w_types
            ] + broadcasted_scalars
            sorted_columns = [
                column for idx, column in sorted(columns_w_idx, key=lambda x: x[0])
            ]
            nesting_level = get_array_nesting_level(
                column_dtype=array_columns_w_types[0][2]
            )
            zipped_array_column = nested_arrays_zip(
                columns=sorted_columns,
                nest_level=nesting_level - 1,
                column_names=input_col_names,
            )

        # Create the nested transform function that applies the function to the zipped
        # array column.
        nested_func = nested_transform(
            func=lambda x: F.concat(*[x[c] for c in input_col_names]),
            nest_level=nesting_level - 1,
        )
        return nested_func(zipped_array_column)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transform the input dataset. Creates a new column named outputCol which is a
        concatenated array of all input columns.

        :param dataset: Pyspark DataFrame to transform.
        :returns: Transformed pyspark dataFrame.
        """
        input_col_names = self.getInputCols()
        input_cols = [F.col(c) for c in input_col_names]
        input_col_datatypes = [
            self.get_column_datatype(dataset=dataset, column_name=c)
            for c in input_col_names
        ]

        output_col = self.array_concatenate_transform(
            input_cols=input_cols,
            input_col_datatypes=input_col_datatypes,
        )
        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer that concatneates the input tensors.

        :returns: Tensorflow keras layer with name equal to the layerName parameter
        that concatenates the input tensors.
        """
        return ArrayConcatenateLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            axis=-1,
            auto_broadcast=self.getAutoBroadcast(),
        )
