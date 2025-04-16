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
from typing import Optional

import pyspark.sql.functions as F
from pyspark.sql import Column, DataFrame, Window, WindowSpec
from pyspark.sql.types import ArrayType


def check_listwise_columns(
    dataset: DataFrame,
    query_col_name: str,
    value_col_name: str,
    sort_col_name: Optional[str] = None,
) -> None:
    """
    Check if the query and value columns are array columns in the dataset.
    If so throw an error since these do not support arrays currently.

    :param dataset: DataFrame containing the query and value columns.
    :param query_col_name: Name of the query column.
    :param value_col_name: Name of the value column.

    :raises ValueError: If the query or value column is an array column.
    :returns: None
    """
    query_col_datatype = dataset.schema[query_col_name].dataType
    value_col_datatype = dataset.schema[value_col_name].dataType

    if isinstance(query_col_datatype, ArrayType):
        raise ValueError("Query column cannot be an array column")

    if isinstance(value_col_datatype, ArrayType):
        raise ValueError("Value column cannot be an array column")

    if sort_col_name is not None:
        sort_col_datatype = dataset.schema[sort_col_name].dataType
        if isinstance(sort_col_datatype, ArrayType):
            raise ValueError("Sort column cannot be an array column")


def get_listwise_condition_and_window(
    query_col: Column,
    value_col: Column,
    sort_col: Column = None,
    sort_order: str = "asc",
    sort_top_n: int = None,
    min_filter_value: float = None,
) -> (Column, WindowSpec):
    """
    Get the condition and window operations for listwise statistics calculation.

    :param query_col: Column containing the query id.
    :param value_col: Column containing the value to calculate statistics on.
    :param sort_col: Column to sort the values by. Default is None.
    :param sort_order: Order to sort the values by. Default is "asc".
    :param sort_top_n: Number of top values to consider for statistics calculation.
    Default is None.
    :param min_filter_value: Minimum value to consider for statistics calculation.
    Default is None.
    :returns: Tuple of the condition and window operations.
    """
    condition_col = None
    window_spec = Window.partitionBy(query_col)

    # Define statistics calculation condition based on topN and sortOrder
    if sort_col is not None:
        if sort_order == "asc":
            sort_col = sort_col.asc()
        elif sort_order == "desc":
            sort_col = sort_col.desc()
        else:
            ValueError(f"Invalid sortOrder: {sort_order}")
        rank_fun = F.row_number().over(window_spec.orderBy(sort_col))
        if sort_top_n is not None:
            condition_col = rank_fun <= sort_top_n
        else:
            ValueError("sortTopN must be set if sortCol is set")
    else:
        condition_col = F.lit(True)

    # Define statistics calculation condition based on min filter value
    if min_filter_value is not None:
        condition_col = condition_col & (value_col >= F.lit(min_filter_value))

    return condition_col, window_spec
