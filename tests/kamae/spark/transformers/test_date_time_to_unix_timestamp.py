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

from kamae.spark.transformers import DateTimeToUnixTimestampTransformer


class TestDateTimeToUnixTimestamp:
    @pytest.fixture(scope="class")
    def date_time_to_unix_timestamp_transform_base(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    "2048-01-01",
                    [["2022-05-02", "2021-04-05"], ["2025-02-28", "2024-08-27"]],
                    "2048-01-01 04:09:32.566",
                    [
                        ["2022-05-02 14:45:21.675", "2021-04-05 06:32:59.345"],
                        ["2025-02-28 23:01:59.067", "2024-08-27 00:05:09.876"],
                    ],
                ),
                (
                    "2022-06-02",
                    [["2150-05-02", "2100-04-05"], ["1972-02-29", "1996-08-27"]],
                    "2022-06-02 04:09:32.566",
                    [
                        ["2150-05-02 14:45:21.675", "2100-04-05 06:32:59.345"],
                        ["1972-02-29 23:01:59.067", "1996-08-27 00:05:09.876"],
                    ],
                ),
            ],
            [
                "single_date",
                "date_array",
                "single_datetime",
                "datetime_array",
            ],
        )

    @pytest.fixture(scope="class")
    def date_time_to_unix_timestamp_transform_single_date_to_seconds(
        self, spark_session
    ):
        return spark_session.createDataFrame(
            [
                (
                    "2048-01-01",
                    [["2022-05-02", "2021-04-05"], ["2025-02-28", "2024-08-27"]],
                    "2048-01-01 04:09:32.566",
                    [
                        ["2022-05-02 14:45:21.675", "2021-04-05 06:32:59.345"],
                        ["2025-02-28 23:01:59.067", "2024-08-27 00:05:09.876"],
                    ],
                    2461449600,
                ),
                (
                    "2022-06-02",
                    [["2150-05-02", "2100-04-05"], ["1972-02-29", "1996-08-27"]],
                    "2022-06-02 04:09:32.566",
                    [
                        ["2150-05-02 14:45:21.675", "2100-04-05 06:32:59.345"],
                        ["1972-02-29 23:01:59.067", "1996-08-27 00:05:09.876"],
                    ],
                    1654128000,
                ),
            ],
            [
                "single_date",
                "date_array",
                "single_datetime",
                "datetime_array",
                "single_date_unix_timestamp_seconds",
            ],
        )

    @pytest.fixture(scope="class")
    def date_time_to_unix_timestamp_transform_date_array_to_milliseconds(
        self, spark_session
    ):
        return spark_session.createDataFrame(
            [
                (
                    "2048-01-01",
                    [["2022-05-02", "2021-04-05"], ["2025-02-28", "2024-08-27"]],
                    "2048-01-01 04:09:32.566",
                    [
                        ["2022-05-02 14:45:21.675", "2021-04-05 06:32:59.345"],
                        ["2025-02-28 23:01:59.067", "2024-08-27 00:05:09.876"],
                    ],
                    [[1651449600000, 1617580800000], [1740700800000, 1724716800000]],
                ),
                (
                    "2022-06-02",
                    [["2150-05-02", "2100-04-05"], ["1972-02-29", "1996-08-27"]],
                    "2022-06-02 04:09:32.566",
                    [
                        ["2150-05-02 14:45:21.675", "2100-04-05 06:32:59.345"],
                        ["1972-02-29 23:01:59.067", "1996-08-27 00:05:09.876"],
                    ],
                    [[5690736000000, 4110566400000], [68169600000, 841104000000]],
                ),
            ],
            [
                "single_date",
                "date_array",
                "single_datetime",
                "datetime_array",
                "date_array_unix_timestamp_milliseconds",
            ],
        )

    @pytest.fixture(scope="class")
    def date_time_to_unix_timestamp_transform_single_datetime_to_milliseconds(
        self, spark_session
    ):
        return spark_session.createDataFrame(
            [
                (
                    "2048-01-01",
                    [["2022-05-02", "2021-04-05"], ["2025-02-28", "2024-08-27"]],
                    "2048-01-01 04:09:32.566",
                    [
                        ["2022-05-02 14:45:21.675", "2021-04-05 06:32:59.345"],
                        ["2025-02-28 23:01:59.067", "2024-08-27 00:05:09.876"],
                    ],
                    2461464572566,
                ),
                (
                    "2022-06-02",
                    [["2150-05-02", "2100-04-05"], ["1972-02-29", "1996-08-27"]],
                    "2022-06-02 04:09:32.566",
                    [
                        ["2150-05-02 14:45:21.675", "2100-04-05 06:32:59.345"],
                        ["1972-02-29 23:01:59.067", "1996-08-27 00:05:09.876"],
                    ],
                    1654142972566,
                ),
            ],
            [
                "single_date",
                "date_array",
                "single_datetime",
                "datetime_array",
                "single_datetime_unix_timestamp_milliseconds",
            ],
        )

    @pytest.fixture(scope="class")
    def date_time_to_unix_timestamp_transform_datetime_array_to_seconds(
        self, spark_session
    ):
        return spark_session.createDataFrame(
            [
                (
                    "2048-01-01",
                    [["2022-05-02", "2021-04-05"], ["2025-02-28", "2024-08-27"]],
                    "2048-01-01 04:09:32.566",
                    [
                        ["2022-05-02 14:45:21.675", "2021-04-05 06:32:59.345"],
                        ["2025-02-28 23:01:59.067", "2024-08-27 00:05:09.876"],
                    ],
                    [
                        [1651502721.675, 1617604379.345],
                        [1740783719.067, 1724717109.876],
                    ],
                ),
                (
                    "2022-06-02",
                    [["2150-05-02", "2100-04-05"], ["1972-02-29", "1996-08-27"]],
                    "2022-06-02 04:09:32.566",
                    [
                        ["2150-05-02 14:45:21.675", "2100-04-05 06:32:59.345"],
                        ["1972-02-29 23:01:59.067", "1996-08-27 00:05:09.876"],
                    ],
                    [[5690789121.675, 4110589979.345], [68252519.067, 841104309.876]],
                ),
            ],
            [
                "single_date",
                "date_array",
                "single_datetime",
                "datetime_array",
                "datetime_array_unix_timestamp_seconds",
            ],
        )

    @pytest.mark.parametrize(
        "input_df, input_col, output_col, unit, expected_df",
        [
            (
                "date_time_to_unix_timestamp_transform_base",
                "single_date",
                "single_date_unix_timestamp_seconds",
                "s",
                "date_time_to_unix_timestamp_transform_single_date_to_seconds",
            ),
            (
                "date_time_to_unix_timestamp_transform_base",
                "date_array",
                "date_array_unix_timestamp_milliseconds",
                "ms",
                "date_time_to_unix_timestamp_transform_date_array_to_milliseconds",
            ),
            (
                "date_time_to_unix_timestamp_transform_base",
                "single_datetime",
                "single_datetime_unix_timestamp_milliseconds",
                "ms",
                "date_time_to_unix_timestamp_transform_single_datetime_to_milliseconds",
            ),
            (
                "date_time_to_unix_timestamp_transform_base",
                "datetime_array",
                "datetime_array_unix_timestamp_seconds",
                "s",
                "date_time_to_unix_timestamp_transform_datetime_array_to_seconds",
            ),
        ],
    )
    def test_date_time_to_unix_timestamp_transform(
        self,
        input_df,
        input_col,
        output_col,
        unit,
        expected_df,
        request,
    ):
        expected = request.getfixturevalue(expected_df)
        input_df = request.getfixturevalue(input_df)

        date_time_to_unix_timestamp_transform = DateTimeToUnixTimestampTransformer(
            inputCol=input_col,
            outputCol=output_col,
            unit=unit,
        )
        actual = date_time_to_unix_timestamp_transform.transform(input_df)
        diff = expected.exceptAll(actual)
        assert diff.isEmpty()

    def test_date_time_to_unix_timestamp_defaults(self):
        # when
        date_time_to_unix_timestamp_transform = DateTimeToUnixTimestampTransformer()
        # then
        assert (
            date_time_to_unix_timestamp_transform.getLayerName()
            == date_time_to_unix_timestamp_transform.uid
        )
        assert date_time_to_unix_timestamp_transform.getUnit() == "s"
        assert (
            date_time_to_unix_timestamp_transform.getOutputCol()
            == f"{date_time_to_unix_timestamp_transform.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, unit",
        [
            (
                tf.constant(["2150-05-02 14:45:21.675", "2100-04-05 06:32:59.345"]),
                None,
                "float",
                "s",
            ),
            (
                tf.constant(["2150-05-02 14:45:21.675", "2100-04-05 06:32:59.345"]),
                None,
                None,
                "s",
            ),
            (
                tf.constant(["2150-05-02 14:45:21.675", "2100-04-05 06:32:59.345"]),
                None,
                None,
                "ms",
            ),
            (
                tf.constant(["1972-02-29", "1996-08-27"]),
                None,
                "double",
                "ms",
            ),
        ],
    )
    def test_date_time_to_unix_timestamp_transform_spark_tf_parity(
        self,
        spark_session,
        input_tensor,
        input_dtype,
        output_dtype,
        unit,
    ):
        # given
        transformer = DateTimeToUnixTimestampTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            unit=unit,
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
            v for v in transformer.get_tf_layer()(input_tensor).numpy().tolist()
        ]
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
