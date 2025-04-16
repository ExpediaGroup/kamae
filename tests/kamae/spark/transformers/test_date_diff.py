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

from kamae.spark.transformers import DateDiffTransformer


class TestDateDiff:
    @pytest.fixture(scope="class")
    def example_dataframe_date(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    "2019-01-01",
                    [
                        ["2019-01-02", "2019-01-02", "2019-01-02", "2019-01-02"],
                        ["2019-01-02", "2019-01-02", "2019-01-02", "2019-01-02"],
                    ],
                    "2019-01-01 17:28:32",
                    "2019-01-05 18:28:32",
                ),
                (
                    2,
                    "2019-01-01",
                    [
                        ["2019-01-03", "2019-01-03", "2019-01-03", "2019-01-03"],
                        ["2019-01-03", "2019-01-03", "2019-01-03", "2019-01-03"],
                    ],
                    "2019-01-01 15:28:32",
                    "2019-01-03 17:40:32",
                ),
                (
                    3,
                    "2019-01-01",
                    [
                        ["2019-01-04", "2019-01-04", "2019-01-04", "2019-01-04"],
                        ["2019-01-04", "2019-01-04", "2019-01-04", "2019-01-04"],
                    ],
                    "2019-01-01 18:35:28",
                    "2019-01-04 17:15:12",
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5"],
        )

    @pytest.fixture(scope="class")
    def example_dataframe_date_w_missing(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    "",
                    [
                        ["2019-01-02", "2019-01-02", "2019-01-02", "2019-01-02"],
                        ["2019-01-02", "2019-01-02", "2019-01-02", "2019-01-02"],
                    ],
                    "2019-01-01 17:28:32",
                    "2019-01-05 18:28:32",
                ),
                (
                    2,
                    "2019-01-01",
                    [
                        ["2019-01-03", "", "2019-01-03", "2019-01-03"],
                        ["2019-01-03", "2019-01-03", "2019-01-03", ""],
                    ],
                    "2019-01-01 15:28:32",
                    "2019-01-03 17:40:32",
                ),
                (
                    3,
                    "",
                    [
                        ["2019-01-04", "2019-01-04", "2019-01-04", "2019-01-04"],
                        ["2019-01-04", "", "", "2019-01-04"],
                    ],
                    "2019-01-01 18:35:28",
                    "",
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5"],
        )

    @pytest.fixture(scope="class")
    def date_diff_transform_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            # DateDiffTransformer expected dataframe with dates example
            [
                (
                    1,
                    "2019-01-01",
                    [
                        ["2019-01-02", "2019-01-02", "2019-01-02", "2019-01-02"],
                        ["2019-01-02", "2019-01-02", "2019-01-02", "2019-01-02"],
                    ],
                    "2019-01-01 17:28:32",
                    "2019-01-05 18:28:32",
                    [[1, 1, 1, 1], [1, 1, 1, 1]],
                ),
                (
                    2,
                    "2019-01-01",
                    [
                        ["2019-01-03", "2019-01-03", "2019-01-03", "2019-01-03"],
                        ["2019-01-03", "2019-01-03", "2019-01-03", "2019-01-03"],
                    ],
                    "2019-01-01 15:28:32",
                    "2019-01-03 17:40:32",
                    [[2, 2, 2, 2], [2, 2, 2, 2]],
                ),
                (
                    3,
                    "2019-01-01",
                    [
                        ["2019-01-04", "2019-01-04", "2019-01-04", "2019-01-04"],
                        ["2019-01-04", "2019-01-04", "2019-01-04", "2019-01-04"],
                    ],
                    "2019-01-01 18:35:28",
                    "2019-01-04 17:15:12",
                    [[3, 3, 3, 3], [3, 3, 3, 3]],
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col2_col3"],
        )

    @pytest.fixture(scope="class")
    def date_diff_transform_w_missing_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    "",
                    [
                        ["2019-01-02", "2019-01-02", "2019-01-02", "2019-01-02"],
                        ["2019-01-02", "2019-01-02", "2019-01-02", "2019-01-02"],
                    ],
                    "2019-01-01 17:28:32",
                    "2019-01-05 18:28:32",
                    [[-1, -1, -1, -1], [-1, -1, -1, -1]],
                ),
                (
                    2,
                    "2019-01-01",
                    [
                        ["2019-01-03", "", "2019-01-03", "2019-01-03"],
                        ["2019-01-03", "2019-01-03", "2019-01-03", ""],
                    ],
                    "2019-01-01 15:28:32",
                    "2019-01-03 17:40:32",
                    [[2, -1, 2, 2], [2, 2, 2, -1]],
                ),
                (
                    3,
                    "",
                    [
                        ["2019-01-04", "2019-01-04", "2019-01-04", "2019-01-04"],
                        ["2019-01-04", "", "", "2019-01-04"],
                    ],
                    "2019-01-01 18:35:28",
                    "",
                    [[-1, -1, -1, -1], [-1, -1, -1, -1]],
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col2_col3"],
        )

    @pytest.fixture(scope="class")
    def date_diff_transform_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            # DateDiffTransformer expected dataframe with dates timestamp example
            [
                (
                    1,
                    "2019-01-01",
                    [
                        ["2019-01-02", "2019-01-02", "2019-01-02", "2019-01-02"],
                        ["2019-01-02", "2019-01-02", "2019-01-02", "2019-01-02"],
                    ],
                    "2019-01-01 17:28:32",
                    "2019-01-05 18:28:32",
                    4,
                ),
                (
                    2,
                    "2019-01-01",
                    [
                        ["2019-01-03", "2019-01-03", "2019-01-03", "2019-01-03"],
                        ["2019-01-03", "2019-01-03", "2019-01-03", "2019-01-03"],
                    ],
                    "2019-01-01 15:28:32",
                    "2019-01-03 17:40:32",
                    2,
                ),
                (
                    3,
                    "2019-01-01",
                    [
                        ["2019-01-04", "2019-01-04", "2019-01-04", "2019-01-04"],
                        ["2019-01-04", "2019-01-04", "2019-01-04", "2019-01-04"],
                    ],
                    "2019-01-01 18:35:28",
                    "2019-01-04 17:15:12",
                    3,
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col4_col5"],
        )

    @pytest.fixture(scope="class")
    def date_diff_transform_w_missing_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    "",
                    [
                        ["2019-01-02", "2019-01-02", "2019-01-02", "2019-01-02"],
                        ["2019-01-02", "2019-01-02", "2019-01-02", "2019-01-02"],
                    ],
                    "2019-01-01 17:28:32",
                    "2019-01-05 18:28:32",
                    4,
                ),
                (
                    2,
                    "2019-01-01",
                    [
                        ["2019-01-03", "", "2019-01-03", "2019-01-03"],
                        ["2019-01-03", "2019-01-03", "2019-01-03", ""],
                    ],
                    "2019-01-01 15:28:32",
                    "2019-01-03 17:40:32",
                    2,
                ),
                (
                    3,
                    "",
                    [
                        ["2019-01-04", "2019-01-04", "2019-01-04", "2019-01-04"],
                        ["2019-01-04", "", "", "2019-01-04"],
                    ],
                    "2019-01-01 18:35:28",
                    "",
                    -1,
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col4_col5"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_cols, output_col, default_value, expected_dataframe",
        [
            (
                "example_dataframe_date",
                ["col2", "col3"],
                "col2_col3",
                None,
                "date_diff_transform_expected_1",
            ),
            (
                "example_dataframe_date",
                ["col4", "col5"],
                "col4_col5",
                None,
                "date_diff_transform_expected_2",
            ),
            (
                "example_dataframe_date_w_missing",
                ["col2", "col3"],
                "col2_col3",
                -1,
                "date_diff_transform_w_missing_expected_1",
            ),
            (
                "example_dataframe_date_w_missing",
                ["col4", "col5"],
                "col4_col5",
                -1,
                "date_diff_transform_w_missing_expected_2",
            ),
        ],
    )
    def test_spark_date_diff_transform(
        self,
        input_dataframe,
        input_cols,
        output_col,
        default_value,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = DateDiffTransformer(
            inputCols=input_cols,
            outputCol=output_col,
            defaultValue=default_value,
        )
        actual = transformer.transform(input_dataframe)

        # then
        diff = actual.exceptAll(expected)

        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_date_diff_transform_defaults(self):
        # when
        dateDiff_transformer = DateDiffTransformer()
        # then
        assert dateDiff_transformer.getLayerName() == dateDiff_transformer.uid
        assert (
            dateDiff_transformer.getOutputCol() == f"{dateDiff_transformer.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensors, default_value, input_dtype, output_dtype",
        [
            (
                [
                    tf.constant(["2019-01-01", "2019-01-01"]),
                    tf.constant(["2019-01-02", "2019-01-06"]),
                ],
                None,
                None,
                "string",
            ),
            (
                [tf.constant(["2019-01-01"]), tf.constant(["2019-01-02"])],
                None,
                "string",
                "double",
            ),
            (
                [
                    tf.constant(["2019-01-01 17:28:32"]),
                    tf.constant(["2019-01-02 15:35:32"]),
                ],
                None,
                "string",
                "float",
            ),
            (
                [
                    tf.constant(["2019-01-01", "2019-01-01"]),
                    tf.constant(["2019-01-02", "2019-01-06"]),
                ],
                -1,
                None,
                "string",
            ),
            (
                [tf.constant(["2019-01-01"]), tf.constant(["2019-01-02"])],
                -1,
                "string",
                "double",
            ),
            (
                [
                    tf.constant([""]),
                    tf.constant(["2019-01-02 15:35:32"]),
                ],
                -1,
                "string",
                "float",
            ),
        ],
    )
    def test_date_diff_transform_spark_tf_parity(
        self, spark_session, input_tensors, default_value, input_dtype, output_dtype
    ):
        # given
        transformer = DateDiffTransformer(
            inputCols=["input_col1", "input_col2"],
            outputCol="output",
            defaultValue=default_value,
            inputDtype=input_dtype,
            outputDtype=output_dtype,
        )
        # when
        spark_df = spark_session.createDataFrame(
            [
                (t1.numpy().decode("utf-8"), t2.numpy().decode("utf-8"))
                for t1, t2 in zip(*input_tensors)
            ],
            ["input_col1", "input_col2"],
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
