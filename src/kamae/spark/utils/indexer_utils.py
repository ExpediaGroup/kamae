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
from farmhash import hash64
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import ArrayType, DataType, StringType

from kamae.spark.utils import flatten_nested_arrays


def safe_hash64(label: str) -> int:
    """
    Attempt to hash a string label and raise an exception if it contains a null
    character.

    :param label: String to hash.
    :raises ValueError: If the label contains a null character.
    :returns: Hashed integer value.
    """
    try:
        return hash64(label)
    except ValueError as e:
        if str(e) == "embedded null character":
            raise ValueError(
                f"""Label {label} contains a null character.
                These cause issues with hashing. You should remove these from your data.
                https://en.wikipedia.org/wiki/Null_character
                """
            ).with_traceback(e.__traceback__)
        else:
            raise e


def collect_labels_array(
    dataset: DataFrame,
    column: Column,
    column_datatype: DataType,
    string_order_type: str = "frequencyDesc",
    mask_token: Optional[str] = None,
    max_num_labels: Optional[int] = None,
) -> List[str]:
    """
    Collects the string labels from a given column in a given dataset.

    :param dataset: Spark input dataset.
    :param column: Spark input column.
    :param column_datatype: Spark input column datatype.
    :param string_order_type: String order type, one of
    'frequencyAsc', 'frequencyDesc', 'alphabeticalAsc', 'alphabeticalDesc'
    :param mask_token: Optional value to mask when indexing.
    :param max_num_labels: Optional value to limit the number of labels.

    :returns: List of string labels.
    """
    possible_order_options = [
        "frequencyAsc",
        "frequencyDesc",
        "alphabeticalAsc",
        "alphabeticalDesc",
    ]
    if string_order_type not in possible_order_options:
        raise ValueError(
            f"string_order_type must be one of {', '.join(possible_order_options)}"
        )

    if (max_num_labels is not None) and (not isinstance(max_num_labels, int)):
        raise ValueError("max_labels_count must be an integer")

    input_col_an_array = isinstance(column_datatype, ArrayType)

    if input_col_an_array:
        # Flatten the array to a single array
        flattened_array_col = flatten_nested_arrays(
            column=column, column_data_type=column_datatype
        )
        # If the input column is an array, we need to flatten it twice
        # before we can collect the string labels
        rdd_vals = (
            dataset.select(flattened_array_col.cast("array<string>"))
            .rdd.map(lambda x: x.asDict().values())
            .flatMap(lambda x: x)
            .flatMap(lambda x: x)
        )
    else:
        # Otherwise, since it is scalar we can just collect the string labels
        rdd_vals = dataset.select(column.cast("string")).rdd.map(lambda x: x[0])

    # If the mask token is defined, we remove it from the labels.
    # This is because otherwise Tensorflow can throw an error when
    # the mask token is not in the correct position of the labels.
    rdd_vals = (
        rdd_vals.filter(lambda x: x != mask_token)
        if mask_token is not None
        else rdd_vals
    )
    sort_ascending = string_order_type in ["alphabeticalAsc", "frequencyAsc"]
    if string_order_type in ["frequencyAsc", "frequencyDesc"]:
        label_vals = (
            rdd_vals.filter(lambda x: x is not None)
            .map(lambda x: (x, 1))
            .reduceByKey(lambda x, y: x + y)
            .sortBy(lambda x: x[1], ascending=sort_ascending)
            .map(lambda x: x[0])
        )
    else:
        label_vals = (
            rdd_vals.filter(lambda x: x is not None)
            .distinct()
            .sortBy(lambda x: x, ascending=sort_ascending)
        )

    if max_num_labels is not None:
        return label_vals.take(max_num_labels)
    else:
        return label_vals.collect()


def collect_labels_array_from_multiple_columns(
    dataset: DataFrame,
    columns: List[Column],
    column_datatypes: List[DataType],
    string_order_type: str = "frequencyDesc",
    mask_token: Optional[str] = None,
    max_num_labels: Optional[int] = None,
) -> List[str]:
    """
    Collects the string labels across multiple columns in a given dataset.

    :param dataset: Spark input dataset
    :param columns: List of Spark input columns.
    :param column_datatypes: List of Spark input column datatypes.
    :param string_order_type: String order type, one of
    'frequencyAsc', 'frequencyDesc', 'alphabeticalAsc', 'alphabeticalDesc'
    :param mask_token: Optional value to mask when indexing.
    :param max_num_labels: Optional value to limit the number of labels.
    :returns: List of string labels.
    """
    flattened_arrays = []
    for column, column_datatype in zip(columns, column_datatypes):
        column_is_array = isinstance(column_datatype, ArrayType)
        if column_is_array:
            # Flatten any arrays and add to the list
            flattened_arrays.append(
                flatten_nested_arrays(column=column, column_data_type=column_datatype)
            )
        else:
            flattened_arrays.append(F.array(column))

    concatenated_array = F.concat(*flattened_arrays).cast("array<string>")

    return collect_labels_array(
        dataset=dataset,
        column=concatenated_array,
        column_datatype=ArrayType(StringType()),
        string_order_type=string_order_type,
        mask_token=mask_token,
        max_num_labels=max_num_labels,
    )
