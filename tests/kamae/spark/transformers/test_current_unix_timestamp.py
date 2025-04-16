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

from kamae.spark.transformers import CurrentUnixTimestampTransformer


class TestCurrentUnixTimestamp:
    @pytest.fixture(scope="class")
    def current_unix_timestamp_transform_base(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "2022-01-02"),
                (4, 2, 6, "b", "2023-08-12"),
                (7, 8, 3, "a", "2020-02-29 11:32:55.432"),
            ],
            ["col1", "col2", "col3", "col4", "col5"],
        )

    @pytest.fixture(scope="class")
    def current_unix_timestamp_transform_base_array(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", ["2022-01-02", "2022-01-02"]),
                (4, 2, 6, "b", ["2023-08-12", "2023-08-12"]),
                (7, 8, 3, "a", ["2020-02-29 11:32:55.432", "2020-02-29 11:32:55.432"]),
            ],
            ["col1", "col2", "col3", "col4", "col5"],
        )

    @pytest.fixture(scope="class")
    def current_unix_timestamp_transform_expected_seconds_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "2022-01-02", 1609411892.454),
                (4, 2, 6, "b", "2023-08-12", 1609411892.454),
                (7, 8, 3, "a", "2020-02-29 11:32:55.432", 1609411892.454),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col5_current_unix_timestamp"],
        )

    @pytest.fixture(scope="class")
    def current_unix_timestamp_transform_expected_array_milliseconds(
        self, spark_session
    ):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    ["2022-01-02", "2022-01-02"],
                    [1609411892454, 1609411892454],
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    ["2023-08-12", "2023-08-12"],
                    [1609411892454, 1609411892454],
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    ["2020-02-29 11:32:55.432", "2020-02-29 11:32:55.432"],
                    [1609411892454, 1609411892454],
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col5_current_unix_timestamp_arr"],
        )

    @pytest.fixture(scope="class")
    def current_unix_timestamp_transform_expected_seconds_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "2022-01-02", 168165120.432),
                (4, 2, 6, "b", "2023-08-12", 168165120.432),
                (7, 8, 3, "a", "2020-02-29 11:32:55.432", 168165120.432),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col5_current_unix_timestamp"],
        )

    @pytest.fixture(scope="class")
    def current_unix_timestamp_transform_expected_milliseconds_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "2022-01-02", 2490337920432),
                (4, 2, 6, "b", "2023-08-12", 2490337920432),
                (7, 8, 3, "a", "2020-02-29 11:32:55.432", 2490337920432),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col5_current_unix_timestamp"],
        )

    @pytest.fixture(scope="class")
    def current_unix_timestamp_transform_expected_array_seconds(self, spark_session):
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
                        [
                            [1709206375.432, 1709206375.432],
                            [1709206375.432, 1709206375.432],
                        ],
                        [
                            [1709206375.432, 1709206375.432],
                            [1709206375.432, 1709206375.432],
                        ],
                        [
                            [1709206375.432, 1709206375.432],
                            [1709206375.432, 1709206375.432],
                        ],
                        [
                            [1709206375.432, 1709206375.432],
                            [1709206375.432, 1709206375.432],
                        ],
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
                        [
                            [1709206375.432, 1709206375.432],
                            [1709206375.432, 1709206375.432],
                        ],
                        [
                            [1709206375.432, 1709206375.432],
                            [1709206375.432, 1709206375.432],
                        ],
                        [
                            [1709206375.432, 1709206375.432],
                            [1709206375.432, 1709206375.432],
                        ],
                        [
                            [1709206375.432, 1709206375.432],
                            [1709206375.432, 1709206375.432],
                        ],
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
                        [
                            [1709206375.432, 1709206375.432],
                            [1709206375.432, 1709206375.432],
                        ],
                        [
                            [1709206375.432, 1709206375.432],
                            [1709206375.432, 1709206375.432],
                        ],
                        [
                            [1709206375.432, 1709206375.432],
                            [1709206375.432, 1709206375.432],
                        ],
                        [
                            [1709206375.432, 1709206375.432],
                            [1709206375.432, 1709206375.432],
                        ],
                    ],
                ),
            ],
            ["col1", "col2", "col2_current_unix_timestamp"],
        )

    @pytest.mark.parametrize(
        "input_df, input_col, output_col, unit, test_datetime, expected_df",
        [
            (
                "current_unix_timestamp_transform_base",
                "col5",
                "col5_current_unix_timestamp",
                "s",
                "2020-12-31 10:51:32.454",
                "current_unix_timestamp_transform_expected_seconds_1",
            ),
            (
                "current_unix_timestamp_transform_base_array",
                "col5",
                "col5_current_unix_timestamp_arr",
                "ms",
                "2020-12-31 10:51:32.454",
                "current_unix_timestamp_transform_expected_array_milliseconds",
            ),
            (
                "current_unix_timestamp_transform_base",
                "col5",
                "col5_current_unix_timestamp",
                "seconds",
                "1975-05-01 08:32:00.432",
                "current_unix_timestamp_transform_expected_seconds_2",
            ),
            (
                "current_unix_timestamp_transform_base",
                "col5",
                "col5_current_unix_timestamp",
                "milliseconds",
                "2048-11-30 08:32:00.432",
                "current_unix_timestamp_transform_expected_milliseconds_2",
            ),
            (
                "example_dataframe_w_nested_arrays",
                "col2",
                "col2_current_unix_timestamp",
                "s",
                "2024-02-29 11:32:55.432",
                "current_unix_timestamp_transform_expected_array_seconds",
            ),
        ],
    )
    def test_current_unix_timestamp_transform(
        self,
        spark_session,
        input_df,
        input_col,
        output_col,
        unit,
        test_datetime,
        expected_df,
        request,
    ):
        expected = request.getfixturevalue(expected_df)
        input_df = request.getfixturevalue(input_df)
        # patch for kamae.spark.transformers.current_unix_timestamp_transform.current_unix_timestamp in CurrentUnixTimestampTransformer
        spark_session.conf.set("spark.sql.session.timeZone", "GMT")
        with patch(
            "kamae.spark.transformers.current_unix_timestamp.F.current_timestamp",
            lambda: F.lit(test_datetime).cast("timestamp"),
        ):
            current_unix_timestamp_transform = CurrentUnixTimestampTransformer(
                inputCol=input_col,
                outputCol=output_col,
                unit=unit,
            )
            actual = current_unix_timestamp_transform.transform(input_df)
            diff = expected.exceptAll(actual)
            assert diff.isEmpty()

    def test_current_unix_timestamp_defaults(self):
        # when
        current_unix_timestamp_transform = CurrentUnixTimestampTransformer()
        # then
        assert (
            current_unix_timestamp_transform.getLayerName()
            == current_unix_timestamp_transform.uid
        )
        assert (
            current_unix_timestamp_transform.getOutputCol()
            == f"{current_unix_timestamp_transform.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, date_string, timestamp_seconds",
        [
            (
                tf.constant(["Hello", "world", "!!"]),
                "string",
                None,
                "2020-12-31 10:51:32.454",
                1609411892.454,
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
                "double",
                "2020-02-29 11:32:55.432",
                1582975975.432,
            ),
        ],
    )
    def test_current_unix_timestamp_transform_spark_tf_parity(
        self,
        spark_session,
        input_tensor,
        input_dtype,
        output_dtype,
        date_string,
        timestamp_seconds,
    ):
        # given
        transformer = CurrentUnixTimestampTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
        )

        spark_df = spark_session.createDataFrame(
            [(v,) for v in input_tensor.numpy().tolist()], ["input"]
        )

        spark_session.conf.set("spark.sql.session.timeZone", "GMT")
        with patch(
            "kamae.spark.transformers.current_unix_timestamp.F.current_timestamp",
            lambda: F.lit(date_string).cast("timestamp"),
        ):
            spark_values = (
                transformer.transform(spark_df)
                .select("output")
                .rdd.map(lambda r: r[0])
                .collect()
            )

        with patch(
            "kamae.tensorflow.layers.current_unix_timestamp.tf.timestamp",
            lambda: tf.constant(timestamp_seconds, dtype=tf.float64),
        ):
            tensorflow_values = [
                v for v in transformer.get_tf_layer()(input_tensor).numpy().tolist()
            ]
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, unit",
        [
            (
                tf.constant(["Hello", "world", "!!"]),
                "string",
                None,
                "s",
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
                "double",
                "ms",
            ),
        ],
    )
    def test_current_unix_timestamp_transform_spark_tf_parity_no_patch(
        self, spark_session, input_tensor, input_dtype, output_dtype, unit
    ):
        # Patching allows us to cheat, a simple check without patching
        # should still work

        # Set the timezone to one far away from UTC. Spark will use this timezone
        # We need to ensure that all our dates are UTC
        spark_session.conf.set("spark.sql.session.timeZone", "Australia/Brisbane")
        # given
        transformer = CurrentUnixTimestampTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            unit=unit,
        )

        spark_df = spark_session.createDataFrame(
            [(v,) for v in input_tensor.numpy().tolist()], ["input"]
        )

        spark_values = (
            transformer.transform(spark_df)
            .select("output")
            .rdd.map(lambda r: r[0])
            .collect()
        )

        tensorflow_values = [
            v for v in transformer.get_tf_layer()(input_tensor).numpy().tolist()
        ]
        # Set Spark and Tensorflow to numpy floats
        spark_values = np.array(spark_values).astype(np.float64)
        tensorflow_values = np.array(tensorflow_values).astype(np.float64)
        # Only check correct to the minute, since some time may have passed between
        # the two calls
        diff = np.abs(spark_values - tensorflow_values)
        if unit == "s":
            max_diff = np.ones_like(diff) * 60
            np.testing.assert_array_less(diff, max_diff)
        else:
            max_diff = np.ones_like(diff) * 60000
            np.testing.assert_array_less(diff, max_diff)
