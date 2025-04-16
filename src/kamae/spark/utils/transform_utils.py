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

from typing import Callable, List

import pyspark.sql.functions as F
from pyspark.sql import Column
from pyspark.sql.types import ArrayType, DataType

from kamae.spark.utils import (
    broadcast_scalar_column_to_array,
    build_udf_return_type,
    get_array_nesting_level,
    nested_arrays_zip,
    nested_lambda,
    nested_transform,
)


def _single_input_single_output_transform(
    input_col: Column,
    input_col_datatype: DataType,
    func: Callable[[Column], Column],
    scalar: bool,
) -> Column:
    """
    Applies either a scalar or array Spark function to a single input column and returns
    a single output column. Ideally this function should not be used directly, the user
    should choose one of the below scalar vs array functions for their usecase. As every
    transform is either scalar or array, not both. This function is used to reduce code
    duplication between the scalar and array functions.

    :param input_col: Input column.
    :param input_col_datatype: Input column datatype.
    :param func: Function to apply to the input column.
    :param scalar: Whether the function to be applied is a scalar function or an
    array function.
    :returns: Output column.
    """
    if not scalar and not isinstance(input_col_datatype, ArrayType):
        raise ValueError(
            f"""Input column was expected to be an array but received datatype
            {input_col_datatype}
            """
        )
    nested_level = get_array_nesting_level(input_col_datatype)
    if not scalar:
        # If the function is not scalar then it operates on the full array, therefore
        # we reduce the nested level by 1.
        nested_level -= 1
    nested_transform_func = nested_transform(func, nested_level)
    return nested_transform_func(input_col)


def single_input_single_output_scalar_transform(
    input_col: Column, input_col_datatype: DataType, func: Callable[[Column], Column]
) -> Column:
    """
    Applies a scalar Spark function (e.g. a Spark standard library function that can be
    applied elementwise to arrays) to a single input column and returns a single output
    column. Caters for the case, where the input column is:

    1. A scalar.
    2. A (possibly nested) array.

    If the input is a scalar, apply `func` directly to the input column.
    If the input is an array, apply `func` elementwise to the input column.

    :param input_col: Input column.
    :param input_col_datatype: Input column datatype.
    :param func: Function to apply to the input column.
    :returns: Output column.
    """
    return _single_input_single_output_transform(
        input_col, input_col_datatype, func, scalar=True
    )


def single_input_single_output_array_transform(
    input_col: Column, input_col_datatype: DataType, func: Callable[[Column], Column]
) -> Column:
    """
    Applies an array Spark function (e.g. a Spark standard library function that
    operates on an array directly, as opposed to elementwise on arrays) to a single
    input column and returns a single output column. Caters for the case, where the
    input column is a nested array, in which case the function is applied to the
    innermost array.

    :param input_col: Input column.
    :param input_col_datatype: Input column datatype.
    :param func: Function to apply to the input column.
    :returns: Output column.
    """
    return _single_input_single_output_transform(
        input_col, input_col_datatype, func, scalar=False
    )


def _single_input_single_output_udf_transform(
    input_col: Column,
    input_col_datatype: DataType,
    func: Callable,
    udf_return_element_datatype: DataType,
    scalar: bool,
) -> Column:
    """
    Applies a Python function (e.g. a lambda function that we will wrap into a
    UDF) to a single input column and returns a single output column. Ideally this
    function should not be used directly, the user should choose one of the below scalar
    vs array functions for their usecase. As every transform is either scalar or array,
    not both. This function is used to reduce code duplication between the scalar and
    array functions.

    :param input_col: Input column.
    :param input_col_datatype: Input column datatype.
    :param func: Function to apply to the input column.
    :param udf_return_element_datatype: Datatype of the UDF return type. Should be the
    raw underlying type.
    :param scalar: Whether the function to be applied is a scalar function or an
    array function.
    :returns: Output column.
    """
    if not scalar and not isinstance(input_col_datatype, ArrayType):
        raise ValueError(
            f"""Input column was expected to be an array but received datatype
            {input_col_datatype}
            """
        )
    nested_level = get_array_nesting_level(input_col_datatype)
    udf_return_type = build_udf_return_type(
        element_dtype=udf_return_element_datatype, nest_level=nested_level
    )
    if not scalar:
        # If the function is not scalar then it operates on the full array, therefore
        # we reduce the nested level by 1.
        nested_level -= 1

    nested_lambda_func = nested_lambda(
        func=func,
        nest_level=nested_level,
    )
    udf_func = F.udf(nested_lambda_func, udf_return_type)
    return udf_func(input_col)


def single_input_single_output_scalar_udf_transform(
    input_col: Column,
    input_col_datatype: DataType,
    func: Callable,
    udf_return_element_datatype: DataType,
) -> Column:
    """
    Applies a scalar Python function (e.g. a lambda function that we will wrap into a
    UDF) to a single input column and returns a single output column. Caters for the
    case, where the input column is:

    1. A scalar.
    2. A (possibly nested) array.

    If the input is a scalar, apply `func` directly to the input column.
    If the input is an array, apply `func` elementwise to the input column.

    `func` should be provided as a python function, and not wrapped in the Spark UDF
    wrapper.

    :param input_col: Input column.
    :param input_col_datatype: Input column datatype.
    :param func: Function to apply to the input column.
    :param udf_return_element_datatype: Datatype of the UDF return type. Should be the
    raw underlying type.
    :returns: Output column.
    """
    return _single_input_single_output_udf_transform(
        input_col, input_col_datatype, func, udf_return_element_datatype, scalar=True
    )


def single_input_single_output_array_udf_transform(
    input_col: Column,
    input_col_datatype: DataType,
    func: Callable,
    udf_return_element_datatype: DataType,
) -> Column:
    """
    Applies an array Python function (e.g. a lambda function that operates on an array
    directly, as opposed to elementwise on arrays which is wrapped into a UDF) to a
    single input column and returns a single output column. Caters for the case, where
    the input column is a nested array, in which case the function is applied to the
    innermost array.

    `func` should be provided as a python function, and not wrapped in the Spark UDF
    wrapper.

    :param input_col: Input column.
    :param input_col_datatype: Input column datatype.
    :param func: Function to apply to the input column.
    :param udf_return_element_datatype: Datatype of the UDF return type. Should be the
    raw underlying type.
    :returns: Output column.
    """
    return _single_input_single_output_udf_transform(
        input_col, input_col_datatype, func, udf_return_element_datatype, scalar=False
    )


def _multi_input_single_output_transform(
    input_cols: List[Column],
    input_col_datatypes: List[DataType],
    input_col_names: List[str],
    func: Callable[[Column], Column],
    scalar: bool,
) -> Column:
    """
    Applies either a scalar or array Spark function to multiple input columns and
    returns a single output column.

    NOTE: This function expects a `func` that takes a single column returning a single
    column. Under the hood we will zip the input columns into a single column and apply
    `func` to the zipped column. Therefore `func` should be cognizant of the names of
    the elements in the zipped array column.

    Example adding two input columns together with names `input_0` and `input_1`:
    func = lambda x: x["input_0"] + x["input_1"]

    Ideally this function should not be used directly, the user
    should choose one of the below scalar vs array functions for their usecase. As every
    transform is either scalar or array, not both. This function is used to reduce code
    duplication between the scalar and array functions.

    :param input_cols: List of input columns.
    :param input_col_datatypes: List of input column datatypes.
    :param input_col_names: List of input column names. If any of the columns are
    arrays, the names of the elements in the zipped array column will be the same as
    these names.
    :param func: Function to apply to the zipped input column.
    :param scalar: Whether the function to be applied is a scalar function or an
    array function.
    :returns: Output column.
    """
    inputs_are_scalar = [
        not isinstance(datatype, ArrayType) for datatype in input_col_datatypes
    ]
    inputs_are_array = [
        isinstance(datatype, ArrayType) for datatype in input_col_datatypes
    ]

    if not scalar and any(inputs_are_scalar):
        raise ValueError(
            f"""Expected all input columns to be of type ArrayType,
            received {input_col_datatypes} instead."""
        )

    if not (all(inputs_are_array) or all(inputs_are_scalar)):
        # If the inputs are not either all scalar or all arrays, then we have a mix of
        # scalars and arrays. In this case we need to broadcast the scalars to the size
        # of the arrays.
        scalar_columns = [
            (idx, col_w_datatype[0])
            for idx, col_w_datatype in enumerate(zip(input_cols, input_col_datatypes))
            if not isinstance(col_w_datatype[1], ArrayType)
        ]
        array_columns_w_types = [
            (idx, col_w_datatype[0], col_w_datatype[1])
            for idx, col_w_datatype in enumerate(zip(input_cols, input_col_datatypes))
            if isinstance(col_w_datatype[1], ArrayType)
        ]
        # Broadcast the scalar to the size of the arrays. Use the first array column
        # to determine the size of the broadcasted scalar. Assumes all arrays are
        # of the same size.
        broadcasted_scalars = [
            (
                idx,
                broadcast_scalar_column_to_array(
                    scalar_column=scalar_column,
                    array_column=array_columns_w_types[0][1],
                    array_column_datatype=array_columns_w_types[0][2],
                ),
            )
            for idx, scalar_column in scalar_columns
        ]
        # Resort the array columns and the broadcasted scalars, so they match the order
        # of the input columns.
        columns_w_idx = [
            (idx, column) for idx, column, _ in array_columns_w_types
        ] + broadcasted_scalars
        sorted_columns = [
            column for idx, column in sorted(columns_w_idx, key=lambda x: x[0])
        ]
        # Check all arrays have same nesting level
        nested_levels = [
            get_array_nesting_level(column_dtype)
            for _, _, column_dtype in array_columns_w_types
        ]
        if not all(nested_level == nested_levels[0] for nested_level in nested_levels):
            raise ValueError(
                f"""Expected all input columns to have the same array nesting level,
                received {nested_levels} instead."""
            )
        nested_level = nested_levels[0]
        zipped_array_column = nested_arrays_zip(
            columns=sorted_columns,
            nest_level=nested_level,
            column_names=input_col_names,
        )
    else:
        # Otherwise, we have all scalars or all arrays and so can just zip together
        # the input columns.
        nested_level = get_array_nesting_level(column_dtype=input_col_datatypes[0])
        zipped_array_column = nested_arrays_zip(
            columns=input_cols, nest_level=nested_level, column_names=input_col_names
        )

    if not scalar:
        # If the function is not scalar then it operates on the full array, therefore
        # we reduce the nested level by 1.
        nested_level -= 1

    nested_func = nested_transform(func=func, nest_level=nested_level)
    return nested_func(zipped_array_column)


def multi_input_single_output_scalar_transform(
    input_cols: List[Column],
    input_col_datatypes: List[DataType],
    input_col_names: List[str],
    func: Callable[[Column], Column],
) -> Column:
    """
    Applies a scalar Spark function (e.g. a Spark standard library function that can be
    applied elementwise to arrays) to multiple input columns and returns a single output
    column. Caters for the case, where the input columns are:

    1. All scalars.
    2. All (possibly nested) arrays.
    3. A mix of scalars and (possibly nested) arrays.

    Zips the arrays into a single (potentially nested array) column.
    Applies `func` to the zipped column. `func` must be cognizant of the names of the
    elements in the zipped array column. If the inputs are a mix of scalars and arrays,
    broadcast the scalar to the size of the arrays before zipping the input columns.

    Example of `func` for a transformer that sums a list of input columns:

    `func = lambda x: reduce(add, [x[c] for c in input_col_names])`

    :param input_cols: List of input columns.
    :param input_col_datatypes: List of input column datatypes.
    :param input_col_names: List of input column names. If any of the columns are
    arrays, the names of the elements in the zipped array column will be the same as
    these names.
    :param func: Function to apply to the input columns in the case they are all scalar.
    :returns: Output column.
    """
    return _multi_input_single_output_transform(
        input_cols=input_cols,
        input_col_datatypes=input_col_datatypes,
        input_col_names=input_col_names,
        func=func,
        scalar=True,
    )


def multi_input_single_output_array_transform(
    input_cols: List[Column],
    input_col_datatypes: List[DataType],
    input_col_names: List[str],
    func: Callable[[Column], Column],
) -> Column:
    """
    Applies an array Spark function (e.g. a Spark standard library function that
    operates on an array directly, as opposed to elementwise on arrays) to multiple
    input columns and returns a single output column. Caters for the case, where the
    input column is a nested array, in which case the function is applied to the
    innermost array.

    NOTE: Function zips the multple inputs into a single column and so `func` should
    reference the input names of the new zipped struct column.

    Example:
    `func = lambda x: reduce(add, [x[c] for c in input_col_names])`

    :param input_cols: List of input columns.
    :param input_col_datatypes: List of input column datatypes.
    :param input_col_names: List of input column names. If any of the columns are
    arrays, the names of the elements in the zipped array column will be the same as
    these names.
    :param func: Function to apply to the input columns in the case they are all scalar.
    arrays. Main difference here is that the function must be cognizant of the names of
    the elements in the zipped array column.
    :returns: Output column.
    """
    return _multi_input_single_output_transform(
        input_cols=input_cols,
        input_col_datatypes=input_col_datatypes,
        input_col_names=input_col_names,
        func=func,
        scalar=False,
    )
