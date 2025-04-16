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

import pytest
import tensorflow as tf
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType

from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.transformers import BaseTransformer


@pytest.fixture(scope="module")
def spark_session():
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[1]")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.dagGraph.retainedRootRDDs", "1")
        .config("spark.ui.retainedJobs", "1")
        .config("spark.ui.retainedStages", "1")
        .config("spark.ui.retainedTasks", "1")
        .config("spark.sql.ui.retainedExecutions", "1")
        .config("spark.worker.ui.retainedExecutors", "1")
        .config("spark.worker.ui.retainedDrivers", "1")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture(scope="module")
def example_dataframe(spark_session):
    example_df = spark_session.createDataFrame(
        [
            (1.0, 2.0, 3.0, "a", "c", [1.0, 2.0, 3.0]),
            (4.0, 2.0, 6.0, "b", "c", [4.0, 2.0, 6.0]),
            (7.0, 8.0, 3.0, "a", "a", [7.0, 8.0, 3.0]),
        ],
        ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3"],
    )
    return example_df


@pytest.fixture(scope="module")
def example_dataframe_with_singleton_array(spark_session):
    example_df = spark_session.createDataFrame(
        [
            (1.0, 2.0, 3.0, "a", "c", [1.0]),
            (4.0, 2.0, 6.0, "b", "c", [4.0, 2.0]),
            (7.0, 8.0, 3.0, "a", "a", [7.0, 8.0, 3.0]),
        ],
        ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3"],
    )
    return example_df


@pytest.fixture(scope="module")
def example_index_input_with_string_arrays(spark_session):
    return spark_session.createDataFrame(
        [
            (1, 2, 3, [["a", "c", "c"], ["a", "c", "c"], ["a", "a", "a"]]),
            (4, 2, 6, [["a", "d", "c"], ["a", "t", "s"], ["x", "o", "p"]]),
            (7, 8, 3, [["l", "c", "c"], ["a", "h", "c"], ["a", "w", "a"]]),
        ],
        ["col1", "col2", "col3", "col4"],
    )


@pytest.fixture(scope="module")
def example_dataframe_w_nested_arrays(spark_session):
    example_df = spark_session.createDataFrame(
        [
            (
                [[1.0, -2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, -3.0], [4.0, 2.0, -6.0]],
                [
                    [["a", "b"], ["c", "d"]],
                    [["a", "b"], ["c", "d"]],
                    [["a", "b"], ["c", "d"]],
                    [["a", "b"], ["c", "d"]],
                ],
            ),
            (
                [[4.0, -2.0, 6.0], [4.0, -2.0, 6.0], [4.0, 2.0, -6.0], [7.0, 8.0, 3.0]],
                [
                    [["e", "f"], ["g", "h"]],
                    [["e", "f"], ["g", "h"]],
                    [["e", "f"], ["g", "h"]],
                    [["e", "f"], ["g", "h"]],
                ],
            ),
            (
                [
                    [7.0, 8.0, 3.0],
                    [7.0, -8.0, 3.0],
                    [7.0, 8.0, -3.0],
                    [-1.0, 2.0, -3.0],
                ],
                [
                    [["i", "j"], ["k", "l"]],
                    [["i", "j"], ["k", "l"]],
                    [["i", "j"], ["k", "l"]],
                    [["i", "j"], ["k", "l"]],
                ],
            ),
        ],
        ["col1", "col2"],
    )
    return example_df


@pytest.fixture(scope="module")
def example_dataframe_w_multiple_numeric_nested_arrays(spark_session):
    example_df = spark_session.createDataFrame(
        [
            (
                [[1.0, -2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, -3.0], [4.0, 2.0, -6.0]],
                [
                    [4.0, 5.0, -1.2],
                    [41.0, -89.45, 56.5],
                    [14.0, -6.0, 9.5],
                    [43.45, -2.0, 4.5],
                ],
            )
        ],
        ["col1", "col2"],
    )
    return example_df


@pytest.fixture(scope="module")
def example_dataframe_w_multiple_string_nested_arrays(spark_session):
    example_df = spark_session.createDataFrame(
        [
            (
                [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"], ["j", "k", "l"]],
                [["m", "n", "o"], ["p", "q", "r"], ["s", "t", "u"], ["v", "w", "x"]],
            )
        ],
        ["col1", "col2"],
    )
    return example_df


@pytest.fixture(scope="module")
def example_dataframe_equal_rows(spark_session):
    example_df = spark_session.createDataFrame(
        [
            (1.0, 2.0, 3.0, "a", "b", [1.0, 2.0, 3.0]),
            (1.0, 2.0, 3.0, "a", "b", [1.0, 2.0, 3.0]),
            (1.0, 2.0, 3.0, "a", "b", [1.0, 2.0, 3.0]),
        ],
        ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3"],
    )
    return example_df


@pytest.fixture(scope="module")
def example_dataframe_with_padding_no_nulls(spark_session):
    example_df = spark_session.createDataFrame(
        [
            (1.0, [3.0, 2.0, 1.0, -1.0]),
            (4.0, [100.0, 6.0, 4.0, -1.0]),
            (7.0, [12.0, 8.0, -1.0, -1.0]),
        ],
        ["col1", "col2"],
    )
    return example_df


@pytest.fixture(scope="module")
def example_dataframe_with_nested_arrays_padding_no_nulls(spark_session):
    example_df = spark_session.createDataFrame(
        [
            (
                [
                    [
                        [100.0, 98.0, 2.0, 5.0, -1.0],
                        [1000.0, 67.0, 84.0, -1.0, -1.0],
                        [1000.0, 67.0, 84.0, -1.0, -1.0],
                    ]
                ],
                [[3.0, 2.0, 1.0, -1.0], [3.0, 2.0, 1.0, -1.0]],
            ),
            (
                [
                    [
                        [167.0, 9.0, 2.0, 5.0, -1.0],
                        [10.0, 6.0, 8.0, -1.0, -1.0],
                        [100.0, 7.0, 4.0, -1.0, -1.0],
                    ]
                ],
                [[100.0, 6.0, 4.0, -1.0], [100.0, 6.0, 4.0, -1.0]],
            ),
        ],
        ["col1", "col2"],
    )
    return example_df


@pytest.fixture(scope="module")
def example_dataframe_with_nulls(spark_session):
    example_df = spark_session.createDataFrame(
        [
            (None, 2.0, 3.0, "a", "c", [None, 2.0, 3.0]),
            (4.0, None, 6.0, "b", None, [4.0, None, 6.0]),
            (7.0, 8.0, None, None, "a", [7.0, 8.0, None]),
            (7.0, 8.0, None, "a", "a", [7.0, 8.0, None]),
        ],
        ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3"],
    )
    return example_df


@pytest.fixture(scope="module")
def example_dataframe_with_padding(spark_session):
    example_df = spark_session.createDataFrame(
        [
            (None, 2.0, 3.0, [4.0, 2.0, 3.0, -1.0, -1.0]),
            (4.0, None, 6.0, [4.0, 3.0, -1.0, -1.0, -1.0]),
            (7.0, 8.0, None, [7.0, -1.0, -1.0, -1.0, -1.0]),
            (7.0, 8.0, None, [7.0, 8.0, 1.0, 9.0, 0.0]),
        ],
        ["col1", "col2", "col3", "col4"],
    )
    return example_df


@pytest.fixture(scope="module")
def example_dataframe_with_padding_2(spark_session):
    example_df = spark_session.createDataFrame(
        [
            (None, 2.0, 3.0, [4.0, 2.0, 3.0, -1.0, -1.0]),
            (4.0, None, 6.0, [4.0, 3.0, -1.0, -1.0, -1.0]),
            (7.0, 8.0, None, [7.0, -1.0, -1.0, -1.0, -1.0]),
            (7.0, 8.0, None, [7.0, 8.0, 1.0, -1.0, -1.0]),
        ],
        ["col1", "col2", "col3", "col4"],
    )
    return example_df


@pytest.fixture(scope="module")
def example_dataframe_with_string_array(spark_session):
    example_df = spark_session.createDataFrame(
        [
            (1, ["-1", "a", "b", "-1"]),
            (4, ["a", "a", "b", "c"]),
            (7, ["b", "b", "b", "a"]),
        ],
        ["col1", "col2"],
    )
    return example_df


@pytest.fixture(scope="module")
def example_dataframe_with_nested_string_array(spark_session):
    example_df = spark_session.createDataFrame(
        [
            (1, [["-1", "a", "b", "-1"]]),
            (4, [["a", "a", "b", "c"]]),
            (7, [["b", "b", "b", "a"]]),
        ],
        ["col1", "col2"],
    )
    return example_df


@pytest.fixture(scope="module")
def example_dataframe_with_ragged_string_array(spark_session):
    example_df = spark_session.createDataFrame(
        [
            (1, ["a", "a", "a", "b", "c"]),
            (4, ["x", "z", "y"]),
            (7, ["a", "b"]),
            (1, ["a", "x", "a", "b"]),
            (7, []),
        ],
        ["col1", "col2"],
    )
    return example_df


@pytest.fixture(scope="module")
def example_dataframe_with_ragged_int_array(spark_session):
    example_df = spark_session.createDataFrame(
        [
            (1, [1, 2, 3, 4, 5]),
            (4, [6, 7, 8]),
            (7, [1, 2]),
            (7, []),
        ],
        ["col1", "col2"],
    )
    return example_df


@pytest.fixture(scope="module")
def example_dataframe_with_long_array(spark_session):
    example_df = spark_session.createDataFrame(
        [
            (
                1,
                [
                    1687087026136,
                    1687087026136,
                    1687087026136,
                    1687087026136,
                    1687087026136,
                ],
            ),
            (4, [1687087026136, 1687087026136, 1687087026136]),
            (7, [1687087026136, 1687087026136]),
            (7, []),
        ],
        ["col1", "col2"],
    )
    return example_df


@pytest.fixture(scope="module")
def example_dataframe_with_ragged_float_array(spark_session):
    example_df = spark_session.createDataFrame(
        [
            (1, [1.0, 2.0, 3.0, 4.0, 5.0]),
            (4, [6.0, 7.0, 8.0]),
            (7, [1.0, 2.0]),
            (7, []),
        ],
        ["col1", "col2"],
    )
    return example_df


@pytest.fixture(scope="class")
def example_dataframe_w_null_characters(spark_session):
    return spark_session.createDataFrame(
        [
            ("a", ["a", "c"]),
            ("b", ["b", "\u0000"]),
            ("\0", ["a", "a"]),
            ("\u0000", ["d", "\0"]),
        ],
        ["col1", "col2"],
    )


@pytest.fixture(scope="module")
def example_dataframe_bool(spark_session):
    example_df = spark_session.createDataFrame(
        [
            (None, 2.0, 3.0, [1.0, 0.0, 1.0, 0.0, 0.0]),
            (4.0, None, 6.0, [1.0, 0.0, 0.0, 0.0, 1.0]),
            (5.0, 6.0, None, [0.0, 0.0, 0.0, 0.0, 1.0]),
            (7.0, 8.0, None, [0.0, 1.0, 1.0, 0.0, 1.0]),
            (5.0, 8.0, None, [0.0, 1.0, 1.0, 0.0, 0.0]),
        ],
        ["col1", "col2", "col3", "col4"],
    )
    return example_df


@pytest.fixture(scope="module")
def layer_name():
    return "test_layer"


@pytest.fixture(scope="module")
def input_col():
    return "test_input"


@pytest.fixture(scope="module")
def output_col():
    return "test_output"


@pytest.fixture(scope="module")
def tf_layer():
    return tf.keras.layers.Dense(1)


@pytest.fixture(scope="module")
def test_base_transformer(layer_name, output_col, input_col, tf_layer):
    class TestTransformer(BaseTransformer, SingleInputSingleOutputParams):
        """Test transformer for testing abstract base class BaseTransformer"""

        @property
        def compatible_dtypes(self) -> Optional[List[DataType]]:
            return None

        def _transform(self, dataset: DataFrame) -> DataFrame:
            return dataset

        def get_tf_layer(self) -> tf.keras.layers.Layer:
            return tf_layer

    return (
        TestTransformer()
        .setLayerName(layer_name)
        .setInputCol(input_col)
        .setOutputCol(output_col)
    )
