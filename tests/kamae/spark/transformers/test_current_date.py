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

from unittest.mock import patch

import numpy as np
import pyspark.sql.functions as F
import pytest
import tensorflow as tf

from kamae.spark.transformers import CurrentDateTransformer


class TestCurrentDate:
    @pytest.fixture(scope="class")
    def current_date_transform_base(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "2022-01-02"),
                (4, 2, 6, "b", "2023-08-12"),
                (7, 8, 3, "a", "2020-02-29"),
            ],
            ["col1", "col2", "col3", "col4", "col5"],
        )

    @pytest.fixture(scope="class")
    def current_date_transform_base_array(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", ["2022-01-02", "2022-01-02"]),
                (4, 2, 6, "b", ["2023-08-12", "2023-08-12"]),
                (7, 8, 3, "a", ["2020-02-29", "2020-02-29"]),
            ],
            ["col1", "col2", "col3", "col4", "col5"],
        )

    @pytest.fixture(scope="class")
    def current_date_transform_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "2022-01-02", "2020-12-31"),
                (4, 2, 6, "b", "2023-08-12", "2020-12-31"),
                (7, 8, 3, "a", "2020-02-29", "2020-12-31"),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col5_current_date"],
        )

    @pytest.fixture(scope="class")
    def current_date_transform_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    ["2022-01-02", "2022-01-02"],
                    ["2020-12-31", "2020-12-31"],
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    ["2023-08-12", "2023-08-12"],
                    ["2020-12-31", "2020-12-31"],
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    ["2020-02-29", "2020-02-29"],
                    ["2020-12-31", "2020-12-31"],
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col5_current_date_arr"],
        )

    @pytest.fixture(scope="class")
    def current_date_transform_expected_3(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "2022-01-02", "1975-05-01"),
                (4, 2, 6, "b", "2023-08-12", "1975-05-01"),
                (7, 8, 3, "a", "2020-02-29", "1975-05-01"),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col5_day_of_week"],
        )

    @pytest.fixture(scope="class")
    def current_date_transform_expected_4(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "2022-01-02", "2034-11-30"),
                (4, 2, 6, "b", "2023-08-12", "2034-11-30"),
                (7, 8, 3, "a", "2020-02-29", "2034-11-30"),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col5_month"],
        )

    @pytest.fixture(scope="class")
    def current_date_transform_expected_5(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "2022-01-02", "2020-02-29"),
                (4, 2, 6, "b", "2023-08-12", "2020-02-29"),
                (7, 8, 3, "a", "2020-02-29", "2020-02-29"),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col5_month"],
        )

    @pytest.fixture(scope="class")
    def current_date_transform_expected_array(self, spark_session):
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
                        [["2024-02-29", "2024-02-29"], ["2024-02-29", "2024-02-29"]],
                        [["2024-02-29", "2024-02-29"], ["2024-02-29", "2024-02-29"]],
                        [["2024-02-29", "2024-02-29"], ["2024-02-29", "2024-02-29"]],
                        [["2024-02-29", "2024-02-29"], ["2024-02-29", "2024-02-29"]],
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
                        [["2024-02-29", "2024-02-29"], ["2024-02-29", "2024-02-29"]],
                        [["2024-02-29", "2024-02-29"], ["2024-02-29", "2024-02-29"]],
                        [["2024-02-29", "2024-02-29"], ["2024-02-29", "2024-02-29"]],
                        [["2024-02-29", "2024-02-29"], ["2024-02-29", "2024-02-29"]],
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
                        [["2024-02-29", "2024-02-29"], ["2024-02-29", "2024-02-29"]],
                        [["2024-02-29", "2024-02-29"], ["2024-02-29", "2024-02-29"]],
                        [["2024-02-29", "2024-02-29"], ["2024-02-29", "2024-02-29"]],
                        [["2024-02-29", "2024-02-29"], ["2024-02-29", "2024-02-29"]],
                    ],
                ),
            ],
            ["col1", "col2", "col2_current_date"],
        )

    @pytest.mark.parametrize(
        "input_df, input_col, output_col, test_date, expected_df",
        [
            (
                "current_date_transform_base",
                "col5",
                "col5_current_date",
                "2020-12-31",
                "current_date_transform_expected_1",
            ),
            (
                "current_date_transform_base_array",
                "col5",
                "col5_current_date_arr",
                "2020-12-31",
                "current_date_transform_expected_2",
            ),
            (
                "current_date_transform_base",
                "col5",
                "col5_current_date",
                "1975-05-01",
                "current_date_transform_expected_3",
            ),
            (
                "current_date_transform_base",
                "col5",
                "col5_current_date",
                "2034-11-30",
                "current_date_transform_expected_4",
            ),
            (
                "current_date_transform_base",
                "col5",
                "col5_current_date",
                "2020-02-29",
                "current_date_transform_expected_5",
            ),
            (
                "example_dataframe_w_nested_arrays",
                "col2",
                "col2_current_date",
                "2024-02-29",
                "current_date_transform_expected_array",
            ),
        ],
    )
    def test_current_date_transform(
        self,
        spark_session,
        input_df,
        input_col,
        output_col,
        test_date,
        expected_df,
        request,
    ):
        expected = request.getfixturevalue(expected_df)
        input_df = request.getfixturevalue(input_df)
        spark_session.conf.set("spark.sql.session.timeZone", "GMT")
        # patch for kamae.spark.transformers.current_date_transform.current_date in CurrentDateTransformer
        with patch(
            "kamae.spark.transformers.current_date.F.localtimestamp",
            lambda: F.lit(test_date).cast("timestamp"),
        ):
            current_date_transform = CurrentDateTransformer(
                inputCol=input_col,
                outputCol=output_col,
            )
            actual = current_date_transform.transform(input_df)
            diff = expected.exceptAll(actual)
            assert diff.isEmpty()

    def test_current_date_defaults(self):
        # when
        current_date_transform = CurrentDateTransformer()
        # then
        assert current_date_transform.getLayerName() == current_date_transform.uid
        assert (
            current_date_transform.getOutputCol()
            == f"{current_date_transform.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, date_string, timestamp_seconds",
        [
            (
                tf.constant(["Hello", "world", "!!"]),
                "string",
                None,
                "2020-12-31",
                1609384036.0,
            ),
            (
                tf.constant(
                    [
                        "EXPEDIA_UK",
                        "2023-08-12 18:19:20.444",
                        "EXPEDIA_US",
                    ]
                ),
                None,
                "string",
                "2020-02-29",
                1582934400.0,
            ),
        ],
    )
    def test_current_date_transform_spark_tf_parity(
        self,
        spark_session,
        input_tensor,
        input_dtype,
        output_dtype,
        date_string,
        timestamp_seconds,
    ):
        # given
        transformer = CurrentDateTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
        )

        spark_df = spark_session.createDataFrame(
            [(v.decode("utf-8"),) for v in input_tensor.numpy().tolist()], ["input"]
        )

        spark_session.conf.set("spark.sql.session.timeZone", "GMT")
        with patch(
            "kamae.spark.transformers.current_date.F.localtimestamp",
            lambda: F.lit(date_string),
        ):
            spark_values = (
                transformer.transform(spark_df)
                .select("output")
                .rdd.map(lambda r: r[0])
                .collect()
            )

        with patch(
            "kamae.tensorflow.layers.current_date.tf.timestamp",
            lambda: tf.constant(timestamp_seconds, dtype=tf.float64),
        ):
            tensorflow_values = [
                v.decode("utf-8")
                for v in transformer.get_tf_layer()(input_tensor).numpy().tolist()
            ]
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype",
        [
            (
                tf.constant(["Hello", "world", "!!"]),
                "string",
                None,
            ),
            (
                tf.constant(
                    [
                        "EXPEDIA_UK",
                        "2023-08-12 18:19:20.444",
                        "EXPEDIA_US",
                    ]
                ),
                None,
                "string",
            ),
        ],
    )
    def test_current_date_transform_spark_tf_parity_no_patch(
        self, spark_session, input_tensor, input_dtype, output_dtype
    ):
        # Patching allows us to cheat, a simple check without patching
        # should still work

        # Set the timezone to one far away from UTC. Spark will use this timezone
        # We need to ensure that all our dates are UTC
        spark_session.conf.set("spark.sql.session.timeZone", "Australia/Brisbane")
        # given
        transformer = CurrentDateTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
        )

        spark_df = spark_session.createDataFrame(
            [(v.decode("utf-8"),) for v in input_tensor.numpy().tolist()], ["input"]
        )

        spark_values = (
            transformer.transform(spark_df)
            .select("output")
            .rdd.map(lambda r: r[0])
            .collect()
        )

        tensorflow_values = [
            v.decode("utf-8")
            for v in transformer.get_tf_layer()(input_tensor).numpy().tolist()
        ]
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
