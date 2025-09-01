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
import pyspark.sql.functions as F
import pytest
import tensorflow as tf

from kamae.spark.transformers import MinHashIndexTransformer


class TestMinHashIndex:
    @pytest.fixture(scope="class")
    def min_hash_example_input(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    ["a", "c", "c"],
                    [["a", "c", "c"], ["a", "c", "c"], ["a", "a", "a"]],
                ),
                (
                    4,
                    ["a", "d", "c"],
                    [["a", "d", "c"], ["a", "t", "s"], ["x", "o", "p"]],
                ),
                (
                    7,
                    ["l", "c", "c"],
                    [["l", "c", "c"], ["a", "h", "c"], ["a", "w", "a"]],
                ),
            ],
            ["col1", "col2", "col3"],
        )

    @pytest.fixture(scope="class")
    def min_hash_col2_array_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    ["a", "c", "c"],
                    [["a", "c", "c"], ["a", "c", "c"], ["a", "a", "a"]],
                    [1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
                ),
                (
                    4,
                    ["a", "d", "c"],
                    [["a", "d", "c"], ["a", "t", "s"], ["x", "o", "p"]],
                    [0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
                ),
                (
                    7,
                    ["l", "c", "c"],
                    [["l", "c", "c"], ["a", "h", "c"], ["a", "w", "a"]],
                    [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                ),
            ],
            ["col1", "col2", "col3", "min_hash_col2"],
        )

    @pytest.fixture(scope="class")
    def min_hash_col3_array_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    ["a", "c", "c"],
                    [["a", "c", "c"], ["a", "c", "c"], ["a", "a", "a"]],
                    [
                        [
                            1,
                            0,
                            0,
                            1,
                            1,
                            0,
                            1,
                            1,
                            0,
                            0,
                            0,
                            0,
                            0,
                            1,
                            1,
                            0,
                            1,
                            1,
                            0,
                            1,
                            1,
                            0,
                            1,
                            1,
                            0,
                        ],
                        [
                            1,
                            0,
                            0,
                            1,
                            1,
                            0,
                            1,
                            1,
                            0,
                            0,
                            0,
                            0,
                            0,
                            1,
                            1,
                            0,
                            1,
                            1,
                            0,
                            1,
                            1,
                            0,
                            1,
                            1,
                            0,
                        ],
                        [
                            1,
                            1,
                            0,
                            1,
                            1,
                            0,
                            1,
                            1,
                            0,
                            1,
                            0,
                            0,
                            1,
                            0,
                            1,
                            0,
                            1,
                            0,
                            0,
                            1,
                            1,
                            0,
                            1,
                            0,
                            0,
                        ],
                    ],
                ),
                (
                    4,
                    ["a", "d", "c"],
                    [["a", "d", "c"], ["a", "t", "s"], ["x", "o", "p"]],
                    [
                        [
                            0,
                            1,
                            0,
                            1,
                            1,
                            0,
                            1,
                            1,
                            1,
                            0,
                            0,
                            0,
                            0,
                            1,
                            1,
                            1,
                            1,
                            1,
                            0,
                            1,
                            1,
                            0,
                            1,
                            1,
                            0,
                        ],
                        [
                            1,
                            0,
                            1,
                            1,
                            1,
                            0,
                            1,
                            0,
                            1,
                            0,
                            1,
                            0,
                            0,
                            1,
                            1,
                            0,
                            1,
                            0,
                            0,
                            0,
                            1,
                            1,
                            1,
                            0,
                            1,
                        ],
                        [
                            1,
                            0,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            0,
                            0,
                            0,
                            1,
                            1,
                            0,
                            0,
                            1,
                            1,
                            0,
                            1,
                            0,
                            0,
                            0,
                            1,
                            0,
                            1,
                        ],
                    ],
                ),
                (
                    7,
                    ["l", "c", "c"],
                    [["l", "c", "c"], ["a", "h", "c"], ["a", "w", "a"]],
                    [
                        [
                            0,
                            0,
                            0,
                            0,
                            1,
                            0,
                            0,
                            1,
                            0,
                            1,
                            0,
                            0,
                            0,
                            0,
                            1,
                            0,
                            0,
                            1,
                            0,
                            1,
                            1,
                            0,
                            1,
                            1,
                            1,
                        ],
                        [
                            0,
                            0,
                            0,
                            1,
                            1,
                            0,
                            1,
                            1,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            1,
                            0,
                            1,
                            1,
                            0,
                            1,
                            1,
                            0,
                            1,
                            0,
                            0,
                        ],
                        [
                            1,
                            1,
                            0,
                            1,
                            1,
                            0,
                            1,
                            1,
                            0,
                            0,
                            1,
                            0,
                            1,
                            1,
                            1,
                            0,
                            1,
                            0,
                            1,
                            1,
                            1,
                            0,
                            1,
                            0,
                            0,
                        ],
                    ],
                ),
            ],
            ["col1", "col2", "col3", "min_hash_col3"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, num_permutations, expected_dataframe",
        [
            (
                "min_hash_example_input",
                "col2",
                "min_hash_col2",
                10,
                "min_hash_col2_array_expected",
            ),
            (
                "min_hash_example_input",
                "col3",
                "min_hash_col3",
                25,
                "min_hash_col3_array_expected",
            ),
        ],
    )
    def test_spark_min_hash(
        self,
        input_dataframe,
        input_col,
        output_col,
        num_permutations,
        expected_dataframe,
        request,
    ):
        # given
        example_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = MinHashIndexTransformer(
            inputCol=input_col,
            outputCol=output_col,
            numPermutations=num_permutations,
        )
        actual = transformer.transform(example_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, num_permutations, mask_value",
        [
            ("example_dataframe_with_padding", "col4", "min_hash_col4", 10, "-1"),
            ("example_dataframe_with_string_array", "col2", "min_hash_col2", 25, "-1"),
        ],
    )
    def test_spark_min_hash_with_mask_equals_no_mask(
        self,
        input_dataframe,
        input_col,
        output_col,
        num_permutations,
        mask_value,
        request,
    ):
        # given
        example_dataframe_w_mask = request.getfixturevalue(input_dataframe)
        example_dataframe_wo_mask = example_dataframe_w_mask.withColumn(
            input_col,
            F.filter(
                F.col(input_col).cast("array<string>"), lambda x: x != F.lit(mask_value)
            ),
        )

        # when
        transformer = MinHashIndexTransformer(
            inputCol=input_col,
            outputCol=output_col,
            inputDtype="string",
            numPermutations=num_permutations,
            maskValue=mask_value,
        )
        actual_w_mask = transformer.transform(example_dataframe_w_mask)
        actual_wo_mask = transformer.transform(example_dataframe_wo_mask)

        # then
        diff = actual_w_mask.select(output_col).exceptAll(
            actual_wo_mask.select(output_col)
        )
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_min_hash_defaults(self):
        # when
        min_hash = MinHashIndexTransformer()
        # then
        assert min_hash.getLayerName() == min_hash.uid
        assert min_hash.getOutputCol() == f"{min_hash.uid}__output"
        assert min_hash.getNumPermutations() == 128

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, num_permutations",
        [
            (
                tf.constant([["a", "b", "c", "d", "e", "f"]]),
                None,
                "double",
                100,
            ),
            (tf.constant([[1, 2, 3]]), "string", None, 3),
            (tf.constant([["e", "f", "g"]]), None, "int", 5),
            (tf.constant([["a", "a", "b", "-1"], ["a", "a", "b", "c"]]), None, None, 6),
            (
                tf.constant([[True, False, False]]),
                "string",
                "smallint",
                4,
            ),
        ],
    )
    def test_min_hash_spark_tf_parity(
        self, spark_session, input_tensor, input_dtype, output_dtype, num_permutations
    ):
        # given
        transformer = MinHashIndexTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            numPermutations=num_permutations,
        )

        if input_tensor.dtype.name == "string":
            spark_df = spark_session.createDataFrame(
                [
                    ([v.decode("utf-8") for v in input_row],)
                    for input_row in input_tensor.numpy().tolist()
                ],
                ["input"],
            )
        else:
            spark_df = spark_session.createDataFrame(
                [
                    ([v for v in input_row],)
                    for input_row in input_tensor.numpy().tolist()
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
