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

from typing import Callable, List, Optional, Tuple

import pyspark.sql.functions as F
from pyspark.sql import Column
from pyspark.sql.types import ArrayType, DataType


def get_array_nesting_level(column_dtype: DataType, start_level: int = 0) -> int:
    """
    Calls get_array_nesting_level_and_element_dtype to determine the nesting level of a
    Spark array column.

    :param column_dtype: DataType of the column to check.
    :param start_level: Counter of the number of times it had to recurse to get the
    element type of the array column. Default is 0.
    :returns: The nesting level of the array column.
    """
    return get_array_nesting_level_and_element_dtype(column_dtype, start_level)[0]


def get_element_type(dtype: DataType) -> DataType:
    """
    Gets the element type of given datatype. If the datatype is not an array,
    it returns the datatype itself.

    :param dtype: The datatype to get the element type of.
    :returns: The element type of the datatype.
    """
    return get_array_nesting_level_and_element_dtype(dtype)[1]


def get_array_nesting_level_and_element_dtype(
    column_dtype: DataType, start_level: int = 0
) -> Tuple[int, DataType]:
    """
    Determines the nesting level of a Spark array column. Also returns the underlying
    type of the column. Recursively attempts to get the element type of the array column
    until it is no longer an array. Keeps a counter of the number of times it had to
    recurse to get the element type. This counter is the nesting level of the array
    column. Returns this counter along with the last found element type.

    :param column_dtype: DataType of the column to check.
    :param start_level: Counter of the number of times it had to recurse to get the
    element type of the array column. Default is 0.
    :returns: Tuple of the nesting level of the array column and the element type of the
    array column.
    """
    try:
        child_dtype = column_dtype.elementType
    except AttributeError:
        return start_level, column_dtype
    else:
        return get_array_nesting_level_and_element_dtype(
            column_dtype=child_dtype, start_level=start_level + 1
        )


def broadcast_scalar_column_to_array(
    scalar_column: Column, array_column: Column, array_column_datatype: DataType
) -> Column:
    """
    Broadcasts a scalar column to a (possibly nested) array column. Repeats the scalar
    value to match the shape of the given array column.

    :param scalar_column: Spark column to broadcast.
    :param array_column: Spark array column to broadcast to.
    :param array_column_datatype: DataType of the array column. Used to understand how
    nested the array column is.
    :returns: Spark array column with the scalar column broadcasted to it.
    """
    nested_level = get_array_nesting_level(array_column_datatype)
    nested_transform_func = nested_transform(
        func=lambda x: scalar_column, nest_level=nested_level
    )
    return nested_transform_func(array_column)


def broadcast_scalar_column_to_array_with_inner_singleton_array(
    scalar_column: Column, array_column: Column, array_column_datatype: DataType
) -> Column:
    """
    Similar to broadcast_scalar_column_to_array, but does not repeat the scalar value in
    the final innermost array.

    Broadcasts a scalar column to a (possibly nested) array column. Repeats the scalar
    value to match the shape of the first N-1 of an N-dim array. The final innermost
    array is left as a singleton array. Used for the array concatenate operation where
    we do not want to repeat the value in the final array.

    Example:
    scalar_column = 1
    array_column = [[1, 2], [3, 4]]
    array_column_datatype = ArrayType(ArrayType(IntegerType()))

    Output: [[1], [1]]

    Using `broadcast_scalar_column_to_array` with the same example would return

    Output: [[1, 1], [1, 1]]

    :param scalar_column: Spark column to broadcast.
    :param array_column: Spark array column to broadcast to.
    :param array_column_datatype: DataType of the array column. Used to understand how
    nested the array column is.
    :returns: Spark array column with the scalar column broadcasted to it, and with a
    singleton array in the innermost position.
    """
    nested_level = get_array_nesting_level(array_column_datatype)
    nested_transform_func = nested_transform(
        func=lambda x: F.array(scalar_column), nest_level=nested_level - 1
    )
    return nested_transform_func(array_column)


def nested_arrays_zip(
    columns: List[Column], nest_level: int, column_names: Optional[List[str]] = None
) -> Column:
    """
    Zips multiple columns of (possibly nested) array type.

    :param columns: List of Spark array columns to be zipped.
    :param nest_level: Nesting level of the array columns.
    :param column_names: Optional list of column names to use for the zipped array. If
    provided, the final array of structs will use these as the names of the columns.
    If not set, the columns will be named "input_0", "input_1", etc.
    If set, must be the same length as the number of columns.
    :returns: Zipped array column.
    """
    if column_names is not None:
        if len(column_names) != len(columns):
            raise ValueError(
                f"""column_names must be the same length as the number of columns.
                Received {len(column_names)} column names for {len(columns)} columns."""
            )
    else:
        column_names = [f"input_{i}" for i in range(len(columns))]

    # First collect everything into a struct, so we can operate on a single column
    arrays_zipped = F.struct(*[c.alias(n) for n, c in zip(column_names, columns)])
    # For each successive nesting level below this, zip the arrays together, each time
    # pushing the struct down one level.
    for n_lvl in range(0, nest_level):
        arrays_zipped = nested_transform(
            func=lambda x: F.arrays_zip(*[x[n].alias(n) for n in column_names]),
            nest_level=n_lvl,
        )(arrays_zipped)

    return arrays_zipped


def nested_transform(
    func: Callable[[Column], Column], nest_level: int
) -> Callable[[Column], Column]:
    """
    Creates a nested version of `pyspark.sql.functions.transform` that can be used to
    apply a given `func` elementwise on a nested array column.

    :param func: Function to apply to each element of the nested array column.
    :param nest_level: Nesting level of the array column.
    :returns: Nested transform function.
    """
    if nest_level <= 0:
        # If the nesting level is 0 or less, we have run out of arrays and are now at
        # the scalar level.
        return func
    return lambda x: F.transform(x, nested_transform(func, nest_level=nest_level - 1))


def nested_lambda(func: Callable, nest_level: int) -> Callable:
    """
    Similar to nested_transform, however in this case creates a lambda function that
    applies a function to each element of a nested array. Used within UDFs to apply a
    function to each element of a nested array.

    :param func: Function to apply to each element of the nested array column.
    :param nest_level: Nesting level of the array column.
    :returns: Nested lambda function.
    """

    def apply_func_to_list(x, function):
        return [function(y) for y in x]

    if nest_level == 0:
        return func
    elif nest_level < 0:
        raise ValueError("nest_level must be greater than or equal to 0.")
    return lambda x: apply_func_to_list(
        x, nested_lambda(func, nest_level=nest_level - 1)
    )


def build_udf_return_type(element_dtype: DataType, nest_level: int) -> DataType:
    """
    Builds the return type of a UDF that applies a function to each element of a nested
    array. This is used to specify the return type of the UDF.

    :param element_dtype: DataType of the elements of the nested array.
    :param nest_level: Nesting level of the array column.
    :returns: DataType of the return type of the UDF.
    """
    if nest_level == 0:
        return element_dtype
    elif nest_level < 0:
        raise ValueError("nest_level must be greater than or equal to 0.")
    for _ in range(nest_level):
        element_dtype = ArrayType(element_dtype)
    return element_dtype


def flatten_nested_arrays(column: Column, column_data_type: DataType) -> Column:
    """
    Flattens a nested array to a single array.

    :param column: Spark array column to be flattened
    :param column_data_type: Datatype of the Spark array column.
    :returns: Flattened array column.
    """
    nested_level = get_array_nesting_level(column_dtype=column_data_type)
    flattened_column = column
    for _ in range(1, nested_level):
        flattened_column = F.flatten(flattened_column)

    return flattened_column
