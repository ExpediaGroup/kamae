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

from kamae.tensorflow.layers import DateTimeToUnixTimestampLayer


class TestDateTimeToUnixTimestamp:
    @pytest.mark.parametrize(
        "inputs, input_name, unit, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant(
                    [
                        "2023-02-28",
                        "2021-06-03",
                        "2023-08-22",
                    ]
                ),
                "input_1",
                "s",
                None,
                None,
                tf.constant(
                    [1677542400.0, 1622678400.0, 1692662400.0],
                    dtype=tf.float64,
                ),
            ),
            (
                tf.constant(
                    [
                        "2023-02-28 09:51:19.531",
                        "2021-06-03 18:40:00.074",
                        "2023-08-22 13:07:27.345",
                    ]
                ),
                "input_1",
                "s",
                "string",
                None,
                tf.constant(
                    [1677577879.531, 1622745600.074, 1692709647.345],
                    dtype=tf.float64,
                ),
            ),
            (
                tf.constant(
                    [
                        [
                            "2025-01-06 13:07:27.000",
                            "2025-06-06 13:07:27.000",
                            "2018-04-12 13:07:27.000",
                        ]
                    ]
                ),
                "input_2",
                "ms",
                None,
                tf.int64,
                tf.constant(
                    [[1736168847000, 1749215247000, 1523538447000]], dtype=tf.int64
                ),
            ),
            (
                tf.constant(
                    [
                        ["2012-04-12"],
                        ["2022-04-12"],
                    ]
                ),
                "input_3",
                "seconds",
                None,
                "int64",
                tf.constant([[1334188800], [1649721600]], dtype=tf.int64),
            ),
            (
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
                "input_4",
                "s",
                None,
                None,
                tf.constant(
                    [
                        [[1668297600.0], [1582934400.0], [1668297600.0]],
                        [[1582934400.0], [1582934400.0], [1582934400.0]],
                    ],
                    dtype=tf.float64,
                ),
            ),
        ],
    )
    def test_date_time_to_unix_timestamp(
        self,
        inputs,
        input_name,
        unit,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        layer = DateTimeToUnixTimestampLayer(
            name=input_name,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            unit=unit,
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
        if output_tensor.dtype.is_floating:
            tf.debugging.assert_near(expected_output, output_tensor)
        else:
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
        layer = DateTimeToUnixTimestampLayer(unit="s")

        start = datetime.datetime.fromisoformat(min_date)
        end = datetime.datetime.fromisoformat(max_date)

        day_tracker = start
        day_range = (end - day_tracker).days
        for d in range(1, day_range):
            input_datetime_string = day_tracker.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            expected_unix_timestamp = (
                day_tracker - datetime.datetime.utcfromtimestamp(0)
            ).total_seconds()
            datetime_tensor = tf.constant([input_datetime_string])
            computed_timestamp = layer(datetime_tensor)[0].numpy()

            assert (
                expected_unix_timestamp == computed_timestamp
            ), f"Expected {expected_unix_timestamp}, got {computed_timestamp}"
            day_tracker = day_tracker + datetime.timedelta(days=1)
