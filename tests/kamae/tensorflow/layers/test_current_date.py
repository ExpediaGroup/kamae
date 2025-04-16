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
from functools import reduce
from unittest.mock import patch

import pytest
import tensorflow as tf

from kamae.tensorflow.layers import CurrentDateLayer


class TestCurrentDate:
    @pytest.mark.parametrize(
        "inputs, input_name, test_timestamp, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant(["item1", "2019-02-01", "55"]),
                "input_1",
                1677577879.0,
                "string",
                None,
                # expected_output being the current month of the year (currently 11)
                tf.constant(
                    [
                        "2023-02-28",
                        "2023-02-28",
                        "2023-02-28",
                    ]
                ),
            ),
            (
                tf.constant([["2021-01-01", "Expedia", "2020-02-29"]]),
                "input_2",
                1622745600.0,
                None,
                "string",
                tf.constant(
                    [
                        [
                            "2021-06-03",
                            "2021-06-03",
                            "2021-06-03",
                        ]
                    ]
                ),
            ),
            (
                tf.constant([["2021-01-01"], ["2023-08-12"], ["test"]]),
                "input_3",
                1609372800.0,
                None,
                None,
                tf.constant(
                    [
                        ["2020-12-31"],
                        ["2020-12-31"],
                        ["2020-12-31"],
                    ]
                ),
            ),
            # Complex shape with leap years
            (
                tf.constant([[[1], [2], [3]], [[4], [5], [6]]], dtype="int32"),
                "input_4",
                1668335020.0,
                "float32",
                "string",
                tf.constant(
                    [
                        [
                            ["2022-11-13"],
                            ["2022-11-13"],
                            ["2022-11-13"],
                        ],
                        [
                            ["2022-11-13"],
                            ["2022-11-13"],
                            ["2022-11-13"],
                        ],
                    ]
                ),
            ),
            (
                tf.constant([[["a"], ["b"], ["c"]], [["d"], ["e"], ["f"]]]),
                "input_4",
                886847020.0,
                None,
                None,
                tf.constant(
                    [
                        [
                            ["1998-02-07"],
                            ["1998-02-07"],
                            ["1998-02-07"],
                        ],
                        [
                            ["1998-02-07"],
                            ["1998-02-07"],
                            ["1998-02-07"],
                        ],
                    ]
                ),
            ),
            (
                tf.constant([[[1.23], [3.456], [56.7]], [[234.4], [78.9], [12.3]]]),
                "input_4",
                1709200279.329,
                "bool",
                "string",
                tf.constant(
                    [
                        [
                            ["2024-02-29"],
                            ["2024-02-29"],
                            ["2024-02-29"],
                        ],
                        [
                            ["2024-02-29"],
                            ["2024-02-29"],
                            ["2024-02-29"],
                        ],
                    ]
                ),
            ),
        ],
    )
    def test_current_date(
        self,
        inputs,
        input_name,
        test_timestamp,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # patch for tf.timestamp() in CurrentDateLayer layer with  of 1622745600.0 is 2021-06-03 00:00:00
        with patch(
            "kamae.tensorflow.layers.current_date.tf.timestamp",
            lambda: tf.constant(test_timestamp, dtype=tf.float64),
        ):
            layer = CurrentDateLayer(
                name=input_name, input_dtype=input_dtype, output_dtype=output_dtype
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
            ("1970-01-01T00:00:00", "2038-01-01T00:00:00"),
            ("2038-01-01T00:00:00", "2100-03-31T00:00:00"),
        ],
    )
    def test_full_dates(self, min_date, max_date):
        """
        Test that all dates between 1970-01-01 and 2100-03-31 are correctly parsed.
        While this takes a long time to run, it is important to guarantee that the logic
        is correct.
        """
        current_date = CurrentDateLayer()

        start = datetime.datetime.fromisoformat(min_date)
        end = datetime.datetime.fromisoformat(max_date)

        def patch_date(x):
            with patch(
                "kamae.tensorflow.layers.current_date.tf.timestamp",
                return_value=tf.constant([x], dtype=tf.float64),
            ):
                return current_date(tf.constant(1))

        day_tracker = start
        diff = []
        day_range = (end - day_tracker).days
        for d in range(1, day_range):
            timestamp_seconds = (
                day_tracker - datetime.datetime.utcfromtimestamp(0)
            ).total_seconds()
            computed_str = (
                patch_date(timestamp_seconds).numpy().decode("utf-8").replace(" ", "T")
            )
            computed = datetime.datetime.fromisoformat(computed_str)

            diff.append((day_tracker - computed).total_seconds())
            day_tracker = day_tracker + datetime.timedelta(days=1)

        assert reduce(lambda x, y: x + abs(y), diff) == 0
