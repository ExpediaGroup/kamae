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

import datetime

import pytest
import tensorflow as tf

from kamae.tensorflow.layers import UnixTimestampToDateTimeLayer


class TestUnixTimestampToDate:
    @pytest.mark.parametrize(
        "inputs, input_name, unit, include_time, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant(
                    [1677577879.531, 1622745600.0738945, 1692709647.345498],
                    dtype=tf.float64,
                ),
                "input_1",
                "s",
                False,
                None,
                None,
                tf.constant(
                    [
                        "2023-02-28",
                        "2021-06-03",
                        "2023-08-22",
                    ]
                ),
            ),
            (
                tf.constant(
                    [1677577879.531, 1622745600.0738945, 1692709647.345498],
                    dtype=tf.float64,
                ),
                "input_1",
                "s",
                True,
                "float64",
                None,
                tf.constant(
                    [
                        "2023-02-28 09:51:19.531",
                        "2021-06-03 18:40:00.074",
                        "2023-08-22 13:07:27.345",
                    ]
                ),
            ),
            (
                tf.constant(
                    [[1736168847000, 1749215247000, 1523538447000]], dtype=tf.int64
                ),
                "input_2",
                "ms",
                True,
                None,
                "string",
                tf.constant(
                    [
                        [
                            "2025-01-06 13:07:27.000",
                            "2025-06-06 13:07:27.000",
                            "2018-04-12 13:07:27.000",
                        ]
                    ]
                ),
            ),
            (
                tf.constant([["1334236047"], ["1649768847"]]),
                "input_3",
                "seconds",
                False,
                "int64",
                None,
                tf.constant(
                    [
                        ["2012-04-12"],
                        ["2022-04-12"],
                    ]
                ),
            ),
            (
                tf.constant(
                    [
                        [[1668335020], [1582981647], [1668335020]],
                        [[1582981647], [1582981647], [1582981647]],
                    ],
                    dtype="int64",
                ),
                "input_4",
                "s",
                False,
                "float64",
                "string",
                tf.constant(
                    [
                        [
                            ["2022-11-13"],
                            ["2020-02-29"],
                            ["2022-11-13"],
                        ],
                        [
                            ["2020-02-29"],
                            ["2020-02-29"],
                            ["2020-02-29"],
                        ],
                    ]
                ),
            ),
        ],
    )
    def test_unix_timestamp_to_date_time(
        self,
        inputs,
        input_name,
        unit,
        include_time,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        layer = UnixTimestampToDateTimeLayer(
            name=input_name,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            unit=unit,
            include_time=include_time,
        )
        output_tensor = layer(inputs)
        # then
        assert layer.name == input_name, "Layer name is not set properly"
        assert (
            output_tensor.dtype == expected_output.dtype
        ), "Output tensor dtype is not the same as expected tensor dtype"
        assert (
            output_tensor.shape == expected_output.shape
        ), "Output tensor shape is not the same as expected tensor shape"
        tf.debugging.assert_equal(expected_output, output_tensor)

    @pytest.mark.parametrize(
        "min_date, max_date",
        [
            ("1970-01-01T09:11:12", "2038-01-01T09:11:12"),
            ("2038-01-01T10:15:56", "2100-03-31T10:15:56"),
        ],
    )
    def test_full_dates(self, min_date, max_date):
        """
        Test that all dates between 1970-01-01 and 2100-03-31 are correctly parsed.
        While this takes a long time to run, it is important to guarantee that the logic
        is correct.
        """
        layer = UnixTimestampToDateTimeLayer(unit="s", include_time=True)

        start = datetime.datetime.fromisoformat(min_date)
        end = datetime.datetime.fromisoformat(max_date)

        day_tracker = start
        day_range = (end - day_tracker).days
        for d in range(1, day_range):
            timestamp_seconds = (
                day_tracker - datetime.datetime.utcfromtimestamp(0)
            ).total_seconds()
            timestamp_tensor = tf.constant([timestamp_seconds], dtype=tf.float64)
            expected_date_str = datetime.datetime.utcfromtimestamp(
                timestamp_seconds
            ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            computed_str = layer(timestamp_tensor)[0].numpy().decode("utf-8")

            assert (
                expected_date_str == computed_str
            ), f"Expected {expected_date_str}, got {computed_str}"
            day_tracker = day_tracker + datetime.timedelta(days=1)
