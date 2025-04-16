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

import numpy as np
import pytest
import tensorflow as tf
from pyspark.errors.exceptions.captured import PythonException

from kamae.spark.transformers import HashIndexTransformer


class TestHashIndex:
    @pytest.fixture(scope="class")
    def example_dataframe_w_array_strings(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], ["a", "c"]),
                (4, 2, 6, "b", "c", [4, 2, 6], ["b", "c"]),
                (7, 8, 3, "a", "a", [7, 8, 3], ["a", "a"]),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "col4_col5"],
        )

    @pytest.fixture(scope="class")
    def hash_indexer_col4_num_bins_100_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], ["a", "c"], 39),
                (4, 2, 6, "b", "c", [4, 2, 6], ["b", "c"], 22),
                (7, 8, 3, "a", "a", [7, 8, 3], ["a", "a"], 39),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "col4_col5",
                "hash_col4",
            ],
        )

    @pytest.fixture(scope="class")
    def hash_indexer_col5_num_bins_5000_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], ["a", "c"], 4206),
                (4, 2, 6, "b", "c", [4, 2, 6], ["b", "c"], 4206),
                (7, 8, 3, "a", "a", [7, 8, 3], ["a", "a"], 3350),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "col4_col5",
                "hash_col5",
            ],
        )

    @pytest.fixture(scope="class")
    def hash_indexer_col4_col5_num_bins_5000_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], ["a", "c"], [3350, 4206]),
                (4, 2, 6, "b", "c", [4, 2, 6], ["b", "c"], [1720, 4206]),
                (7, 8, 3, "a", "a", [7, 8, 3], ["a", "a"], [3350, 3350]),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "col4_col5",
                "hash_col4_col5",
            ],
        )

    @pytest.fixture
    def hash_indexer_col1_num_bins_5000_array_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        [1.0, -2.0, 3.0],
                        [1.0, 2.0, 3.0],
                        [1.0, 2.0, -3.0],
                        [4.0, 2.0, -6.0],
                    ],
                    [
                        [["a", "b"], ["c", "d"]],
                        [["a", "b"], ["c", "d"]],
                        [["a", "b"], ["c", "d"]],
                        [["a", "b"], ["c", "d"]],
                    ],
                    [
                        [[3350, 1720], [4206, 0]],
                        [[3350, 1720], [4206, 0]],
                        [[3350, 1720], [4206, 0]],
                        [[3350, 1720], [4206, 0]],
                    ],
                ),
                (
                    [
                        [4.0, -2.0, 6.0],
                        [4.0, -2.0, 6.0],
                        [4.0, 2.0, -6.0],
                        [7.0, 8.0, 3.0],
                    ],
                    [
                        [["e", "f"], ["g", "h"]],
                        [["e", "f"], ["g", "h"]],
                        [["e", "f"], ["g", "h"]],
                        [["e", "f"], ["g", "h"]],
                    ],
                    [
                        [[4366, 631], [4320, 827]],
                        [[4366, 631], [4320, 827]],
                        [[4366, 631], [4320, 827]],
                        [[4366, 631], [4320, 827]],
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
                    [
                        [[294, 649], [514, 4795]],
                        [[294, 649], [514, 4795]],
                        [[294, 649], [514, 4795]],
                        [[294, 649], [514, 4795]],
                    ],
                ),
            ],
            ["col1", "col2", "hash_col2"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, num_bins, mask_value, expected_dataframe",
        [
            (
                "example_dataframe_w_array_strings",
                "col4",
                "hash_col4",
                100,
                None,
                "hash_indexer_col4_num_bins_100_expected",
            ),
            (
                "example_dataframe_w_array_strings",
                "col5",
                "hash_col5",
                5000,
                "d",
                "hash_indexer_col5_num_bins_5000_expected",
            ),
            (
                "example_dataframe_w_array_strings",
                "col4_col5",
                "hash_col4_col5",
                5000,
                "d",
                "hash_indexer_col4_col5_num_bins_5000_expected",
            ),
            (
                "example_dataframe_w_nested_arrays",
                "col2",
                "hash_col2",
                5000,
                "d",
                "hash_indexer_col1_num_bins_5000_array_expected",
            ),
        ],
    )
    def test_spark_hash_indexer(
        self,
        input_dataframe,
        input_col,
        output_col,
        num_bins,
        mask_value,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = HashIndexTransformer(
            inputCol=input_col,
            outputCol=output_col,
            numBins=num_bins,
            maskValue=mask_value,
        )
        actual = transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_hash_indexer_defaults(self):
        # when
        hash_indexer = HashIndexTransformer()
        # then
        assert hash_indexer.getLayerName() == hash_indexer.uid
        assert hash_indexer.getOutputCol() == f"{hash_indexer.uid}__output"

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, num_bins, mask_value",
        [
            (tf.constant(["a", "b", "c", "d", "e", "f"]), None, "float", 100, "f"),
            (tf.constant(["c", "c", "d"]), "string", "string", 200, "c"),
            (tf.constant(["e", "f", "g"]), None, "bigint", 3000, None),
            (tf.constant([1, 2, 3]), "string", None, 43, "h"),
            (tf.constant(["k", "l", "m"]), None, None, 99, None),
        ],
    )
    def test_hash_indexer_spark_tf_parity(
        self,
        spark_session,
        input_tensor,
        input_dtype,
        output_dtype,
        num_bins,
        mask_value,
    ):
        # given
        transformer = HashIndexTransformer(
            numBins=num_bins,
            inputCol="input",
            outputCol="output",
            maskValue=mask_value,
            inputDtype=input_dtype,
            outputDtype=output_dtype,
        )
        # when
        spark_df = spark_session.createDataFrame(
            [
                (v.decode("utf-8") if isinstance(v, bytes) else v,)
                for v in input_tensor.numpy().tolist()
            ],
            ["input"],
        )

        spark_values = (
            transformer.transform(spark_df)
            .select("output")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        tensorflow_values = [
            v.decode("utf-8") if isinstance(v, bytes) else v
            for v in transformer.get_tf_layer()(input_tensor).numpy().tolist()
        ]

        # then
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col",
        [
            ("example_dataframe_w_null_characters", "col1", "hash_col1"),
            ("example_dataframe_w_null_characters", "col2", "hash_col2"),
        ],
    )
    def test_hash_indexer_w_null_characters_raises_error(
        self, input_dataframe, input_col, output_col, request
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        # when
        transformer = HashIndexTransformer(
            inputCol=input_col, outputCol=output_col, numBins=10
        )
        with pytest.raises(PythonException):
            transformer.transform(input_dataframe).show()
