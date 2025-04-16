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

import pyspark.sql.functions as F
from pyspark.sql import Column
from pyspark.sql.types import DataType

from kamae.spark.utils import (
    build_udf_return_type,
    flatten_nested_arrays,
    get_array_nesting_level_and_element_dtype,
    single_input_single_output_array_transform,
)


def construct_nested_elements_for_scaling(
    column: Column,
    column_datatype: DataType,
    array_dim: int,
) -> Column:
    """
    Scaling nested elements in Spark is difficult and requires us to extract the `ith`
    element from the innermost array to compute the moments on.
    This function creates multiple columns, one for each dimension of the inner array
    and flattens them into a single array. It then explodes these out so that the mean
    and variance can be computed on the flattened array.

    Only intended to be used for the StandardScaleEstimator &
    ConditionalStandardScaleEstimator.

    :param column: The input column to extract the element from.
    :param column_datatype: The datatype of the input column.
    :param array_dim: The dimension of the innermost array.
    :returns: A column containing a struct of (possibly exploded) elements.
    """
    nested_lvl, element_dtype = get_array_nesting_level_and_element_dtype(
        column_datatype
    )
    elements = [
        single_input_single_output_array_transform(
            input_col=column,
            input_col_datatype=column_datatype,
            func=lambda x: F.element_at(x, idx),
        ).alias(f"element_{idx}")
        for idx in range(1, array_dim + 1)
    ]
    if nested_lvl > 1:
        # If the element is a nested array, we need to flatten it.
        flat_elements = [
            flatten_nested_arrays(
                column=e,
                column_data_type=build_udf_return_type(
                    element_dtype=element_dtype, nest_level=nested_lvl - 1
                ),
            )
            for e in elements
        ]
        # We then zip the flattened elements together and call explode.
        # Explode is needed in order to use the spark mean and stddev functions
        # over the arrays.
        return F.explode(F.arrays_zip(*flat_elements)).alias("element_struct")

    else:
        # If the element is scalar then we wrap it in a struct to mimic
        # the same structure as the nested arrays.
        return F.struct(*elements).alias("element_struct")
