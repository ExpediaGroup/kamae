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

from kamae.spark.transformers import DateAddTransformer


class TestDateAdd:
    @pytest.fixture(scope="class")
    def example_dataframe_date_add(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    "2019-01-01",
                    "2019-01-01 17:28:32",
                    10,
                    [
                        ["2020-01-25", "2019-11-22", "2002-04-30"],
                        ["2024-11-02", "2029-01-02", "2039-01-02"],
                    ],
                    [
                        [-1, 45, 23],
                        [-10, -23, 13],
                    ],
                ),
            ],
            [
                "single_date",
                "single_datetime",
                "num_days",
                "array_dates",
                "array_num_days",
            ],
        )

    @pytest.fixture(scope="class")
    def date_add_expected_single_date_dynamic_num_days(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    "2019-01-01",
                    "2019-01-01 17:28:32",
                    10,
                    [
                        ["2020-01-25", "2019-11-22", "2002-04-30"],
                        ["2024-11-02", "2029-01-02", "2039-01-02"],
                    ],
                    [
                        [-1, 45, 23],
                        [-10, -23, 13],
                    ],
                    "2019-01-11",
                ),
            ],
            [
                "single_date",
                "single_datetime",
                "num_days",
                "array_dates",
                "array_num_days",
                "single_date_output",
            ],
        )

    @pytest.fixture(scope="class")
    def date_add_expected_single_date_static_num_days_62(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    "2019-01-01",
                    "2019-01-01 17:28:32",
                    10,
                    [
                        ["2020-01-25", "2019-11-22", "2002-04-30"],
                        ["2024-11-02", "2029-01-02", "2039-01-02"],
                    ],
                    [
                        [-1, 45, 23],
                        [-10, -23, 13],
                    ],
                    "2019-03-04",
                ),
            ],
            [
                "single_date",
                "single_datetime",
                "num_days",
                "array_dates",
                "array_num_days",
                "single_date_output",
            ],
        )

    @pytest.fixture(scope="class")
    def date_add_expected_single_datetime_dynamic_num_days(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    "2019-01-01",
                    "2019-01-01 17:28:32",
                    10,
                    [
                        ["2020-01-25", "2019-11-22", "2002-04-30"],
                        ["2024-11-02", "2029-01-02", "2039-01-02"],
                    ],
                    [
                        [-1, 45, 23],
                        [-10, -23, 13],
                    ],
                    "2019-01-11",
                ),
            ],
            [
                "single_date",
                "single_datetime",
                "num_days",
                "array_dates",
                "array_num_days",
                "single_datetime_output",
            ],
        )

    @pytest.fixture(scope="class")
    def date_add_expected_single_datetime_static_num_days_37(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    "2019-01-01",
                    "2019-01-01 17:28:32",
                    10,
                    [
                        ["2020-01-25", "2019-11-22", "2002-04-30"],
                        ["2024-11-02", "2029-01-02", "2039-01-02"],
                    ],
                    [
                        [-1, 45, 23],
                        [-10, -23, 13],
                    ],
                    "2019-02-07",
                ),
            ],
            [
                "single_date",
                "single_datetime",
                "num_days",
                "array_dates",
                "array_num_days",
                "single_datetime_output",
            ],
        )

    @pytest.fixture(scope="class")
    def date_add_expected_array_dates_dynamic_num_days(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    "2019-01-01",
                    "2019-01-01 17:28:32",
                    10,
                    [
                        ["2020-01-25", "2019-11-22", "2002-04-30"],
                        ["2024-11-02", "2029-01-02", "2039-01-02"],
                    ],
                    [
                        [-1, 45, 23],
                        [-10, -23, 13],
                    ],
                    [
                        # Take the dates above and add the num_days to each date
                        ["2020-01-24", "2020-01-06", "2002-05-23"],
                        ["2024-10-23", "2028-12-10", "2039-01-15"],
                    ],
                ),
            ],
            [
                "single_date",
                "single_datetime",
                "num_days",
                "array_dates",
                "array_num_days",
                "array_dates_output",
            ],
        )

    @pytest.fixture(scope="class")
    def date_add_expected_array_dates_static_num_days_minus_13(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    "2019-01-01",
                    "2019-01-01 17:28:32",
                    10,
                    [
                        ["2020-01-25", "2019-11-22", "2002-04-30"],
                        ["2024-11-02", "2029-01-02", "2039-01-02"],
                    ],
                    [
                        [-1, 45, 23],
                        [-10, -23, 13],
                    ],
                    [
                        ["2020-01-12", "2019-11-09", "2002-04-17"],
                        ["2024-10-20", "2028-12-20", "2038-12-20"],
                    ],
                ),
            ],
            [
                "single_date",
                "single_datetime",
                "num_days",
                "array_dates",
                "array_num_days",
                "array_dates_output",
            ],
        )

    @pytest.mark.parametrize(
        "input_col, input_cols, output_col, num_days, expected_dataframe",
        [
            (
                None,
                ["single_date", "num_days"],
                "single_date_output",
                None,
                "date_add_expected_single_date_dynamic_num_days",
            ),
            (
                "single_date",
                None,
                "single_date_output",
                62,
                "date_add_expected_single_date_static_num_days_62",
            ),
            (
                None,
                ["single_datetime", "num_days"],
                "single_datetime_output",
                None,
                "date_add_expected_single_datetime_dynamic_num_days",
            ),
            (
                "single_datetime",
                None,
                "single_datetime_output",
                37,
                "date_add_expected_single_datetime_static_num_days_37",
            ),
            (
                None,
                ["array_dates", "array_num_days"],
                "array_dates_output",
                None,
                "date_add_expected_array_dates_dynamic_num_days",
            ),
            (
                "array_dates",
                None,
                "array_dates_output",
                -13,
                "date_add_expected_array_dates_static_num_days_minus_13",
            ),
        ],
    )
    def test_spark_date_add_transform(
        self,
        example_dataframe_date_add,
        input_col,
        input_cols,
        output_col,
        num_days,
        expected_dataframe,
        request,
    ):
        # given
        expected = request.getfixturevalue(expected_dataframe)
        # when
        if input_col is not None:
            transformer = DateAddTransformer(
                inputCol=input_col,
                outputCol=output_col,
                numDays=num_days,
            )
        else:
            transformer = DateAddTransformer(
                inputCols=input_cols,
                outputCol=output_col,
            )
        actual = transformer.transform(example_dataframe_date_add)

        # then
        diff = actual.exceptAll(expected)

        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_date_add_transform_defaults(self):
        # when
        date_add_transformer = DateAddTransformer()
        # then
        assert date_add_transformer.getLayerName() == date_add_transformer.uid
        assert (
            date_add_transformer.getOutputCol() == f"{date_add_transformer.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensor, num_days, input_dtype, output_dtype",
        [
            (
                tf.constant(["2019-01-01", "2019-01-01"]),
                10,
                None,
                "string",
            ),
            (
                tf.constant(
                    [
                        "2020-01-25",
                        "2019-11-22",
                        "2002-04-30",
                        "2032-05-01",
                        "2024-11-02",
                        "2029-01-02",
                        "2039-01-02",
                        "2074-01-01",
                    ],
                ),
                -13,
                "string",
                None,
            ),
            (
                tf.constant(["2019-01-01 17:28:32"]),
                100,
                None,
                None,
            ),
        ],
    )
    def test_date_add_transform_single_input_spark_tf_parity(
        self, spark_session, input_tensor, num_days, input_dtype, output_dtype
    ):
        # given
        transformer = DateAddTransformer(
            inputCol="input_col1",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            numDays=num_days,
        )
        # when
        spark_df = spark_session.createDataFrame(
            [(inp.decode("utf-8"),) for inp in input_tensor.numpy().tolist()],
            ["input_col1"],
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
        "input_tensors, input_dtype, output_dtype",
        [
            (
                [
                    tf.constant(["2019-01-01", "2019-01-01"]),
                    tf.constant([10, 62]),
                ],
                None,
                "string",
            ),
            (
                [
                    tf.constant(
                        [
                            "2020-01-25",
                            "2019-11-22",
                            "2002-04-30",
                            "2032-05-01",
                            "2024-11-02",
                            "2029-01-02",
                            "2039-01-02",
                            "2074-01-01",
                        ],
                    ),
                    tf.constant([-13, 37, 100, 8, 7, 45, 324, -7658]),
                ],
                None,
                None,
            ),
            (
                [tf.constant(["2019-01-01 17:28:32"]), tf.constant([100])],
                None,
                None,
            ),
        ],
    )
    def test_date_add_transform_multi_input_spark_tf_parity(
        self, spark_session, input_tensors, input_dtype, output_dtype
    ):
        col_names = [f"input{i}" for i in range(len(input_tensors))]
        # given
        transformer = DateAddTransformer(
            inputCols=col_names,
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
        )
        # when
        spark_df = spark_session.createDataFrame(
            [
                tuple(
                    [
                        ti.numpy().decode("utf-8")
                        if isinstance(ti.numpy(), bytes)
                        else int(ti.numpy())
                        for ti in t
                    ]
                )
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

    @pytest.mark.parametrize(
        "input_cols, output_col, input_dtype, output_dtype",
        [
            (
                ["col1", "col2"],
                "output_col",
                "string",
                "string",
            ),
            (
                ["col1", "col2"],
                "output_col",
                "string",
                None,
            ),
        ],
    )
    def test_spark_date_add_transform_raises_error_with_multiple_inputs_input_casting(
        self, input_cols, output_col, input_dtype, output_dtype
    ):
        # then
        with pytest.raises(ValueError):
            # when
            _ = DateAddTransformer(
                inputCols=input_cols,
                outputCol=output_col,
                inputDtype=input_dtype,
                outputDtype=output_dtype,
            )

        with pytest.raises(ValueError):
            # when
            t = DateAddTransformer(
                outputCol=output_col,
                inputDtype=input_dtype,
            )
            t.setInputCols(input_cols)

        with pytest.raises(ValueError):
            # when
            t = DateAddTransformer(
                inputCols=input_cols,
                outputCol=output_col,
            )
            t.setInputDtype(input_dtype)
