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

from kamae.spark.transformers import StringConcatenateTransformer

from ..test_helpers import tensor_to_python_type


class TestStringConcatenate:
    @pytest.fixture(scope="class")
    def concat_transform_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], "a_c"),
                (4, 2, 6, "b", "c", [4, 2, 6], "b_c"),
                (7, 8, 3, "a", "a", [7, 8, 3], "a_a"),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "concat_col4_col5",
            ],
        )

    @pytest.fixture(scope="class")
    def concat_transform_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.0, 2.0, 3.0, "a", "c", [1, 2, 3], "c*a"),
                (4.0, 2.0, 6.0, "b", "c", [4, 2, 6], "c*b"),
                (7.0, 8.0, 3.0, "a", "a", [7, 8, 3], "a*a"),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "concat_col5_col4",
            ],
        )

    @pytest.fixture(scope="class")
    def concat_transform_expected_3(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.0, 2.0, 3.0, "a", "c", ["1.0", "2.0", "3.0"], "1.02.03.0"),
                (4.0, 2.0, 6.0, "b", "c", ["4.0", "2.0", "6.0"], "4.02.06.0"),
                (7.0, 8.0, 3.0, "a", "a", ["7.0", "8.0", "3.0"], "7.08.03.0"),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "concat_col1col2col3",
            ],
        )

    @pytest.fixture(scope="class")
    def concat_string_concat_array(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        ["a", "b", "c"],
                        ["d", "e", "f"],
                        ["g", "h", "i"],
                        ["j", "k", "l"],
                    ],
                    [
                        ["m", "n", "o"],
                        ["p", "q", "r"],
                        ["s", "t", "u"],
                        ["v", "w", "x"],
                    ],
                    [
                        ["a-m", "b-n", "c-o"],
                        ["d-p", "e-q", "f-r"],
                        ["g-s", "h-t", "i-u"],
                        ["j-v", "k-w", "l-x"],
                    ],
                )
            ],
            ["col1", "col2", "concat_string_array_col1_col2"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_cols, output_col, input_dtype, separator, expected_dataframe",
        [
            (
                "example_dataframe",
                ["col4", "col5"],
                "concat_col4_col5",
                None,
                None,
                "concat_transform_expected_1",
            ),
            (
                "example_dataframe",
                ["col5", "col4"],
                "concat_col5_col4",
                None,
                "*",
                "concat_transform_expected_2",
            ),
            (
                "example_dataframe",
                ["col1", "col2", "col3"],
                "concat_col1col2col3",
                "string",
                "",
                "concat_transform_expected_3",
            ),
            (
                "example_dataframe_w_multiple_string_nested_arrays",
                ["col1", "col2"],
                "concat_string_array_col1_col2",
                None,
                "-",
                "concat_string_concat_array",
            ),
        ],
    )
    def test_spark_string_concatenate_transform(
        self,
        input_dataframe,
        input_cols,
        output_col,
        input_dtype,
        separator,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = StringConcatenateTransformer(
            outputCol=output_col,
            inputDtype=input_dtype,
        )
        transformer = transformer.setInputCols(input_cols)
        if separator is not None:
            transformer = transformer.setSeparator(separator)
        actual = transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_string_concatenate_transform_defaults(self):
        # when
        string_concatenate_transformer = StringConcatenateTransformer()
        # then
        assert (
            string_concatenate_transformer.getLayerName()
            == string_concatenate_transformer.uid
        )
        assert (
            string_concatenate_transformer.getOutputCol()
            == f"{string_concatenate_transformer.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensors, input_dtype, output_dtype, separator",
        [
            ([tf.constant(["a", "b"]), tf.constant(["c", "d"])], None, "string", None),
            (
                [tf.constant(["hello", "good"]), tf.constant(["world", "bye"])],
                None,
                None,
                " ",
            ),
            (
                [tf.constant([True, False]), tf.constant(["world", "bye"])],
                "string",
                None,
                "X",
            ),
            (
                [
                    tf.constant([123, 1, 45]),
                    tf.constant([456, 2, 67]),
                    tf.constant([789, 3, 89]),
                ],
                "string",
                None,
                "!!!",
            ),
        ],
    )
    def test_string_concatenate_transform_spark_tf_parity(
        self, spark_session, input_tensors, input_dtype, output_dtype, separator
    ):
        from pyspark.sql.types import StringType

        col_names = [f"input{i}" for i in range(len(input_tensors))]

        # given
        transformer = StringConcatenateTransformer(
            inputCols=col_names,
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            separator=separator,
        )
        # when
        spark_df = spark_session.createDataFrame(
            [
                tuple([tensor_to_python_type(ti) for ti in t])
                for t in zip(*input_tensors)
            ],
            col_names,
        )
        spark_values = (
            transformer.transform(spark_df)
            .select("output")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        tensorflow_values = [
            v.decode("utf-8") if isinstance(v, bytes) else v
            for v in transformer.get_tf_layer()(input_tensors).numpy().tolist()
        ]
        # then
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
