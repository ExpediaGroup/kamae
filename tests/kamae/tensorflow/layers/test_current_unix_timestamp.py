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

import pytest
import tensorflow as tf

from kamae.tensorflow.layers import CurrentUnixTimestampLayer


class TestCurrentUnixTimestamp:
    @pytest.mark.parametrize(
        "inputs, input_name, test_timestamp, input_dtype, output_dtype, unit, expected_output",
        [
            (
                tf.constant(["item1", "2019-02-01", "55"]),
                "input_1",
                1677577879.675,
                None,
                None,
                "s",
                tf.constant(
                    [
                        1677577879.675,
                        1677577879.675,
                        1677577879.675,
                    ],
                    dtype=tf.float64,
                ),
            ),
            (
                tf.constant([["2021-01-01", "Expedia", "2020-02-29"]]),
                "input_2",
                1622745600.678,
                None,
                "int64",
                "ms",
                tf.constant(
                    [
                        [
                            1622745600678,
                            1622745600678,
                            1622745600678,
                        ]
                    ],
                    dtype=tf.int64,
                ),
            ),
            (
                tf.constant([["2021-01-01"], ["2023-08-12"], ["test"]]),
                "input_3",
                1609372800.723,
                None,
                None,
                "seconds",
                tf.constant(
                    [
                        [1609372800.723],
                        [1609372800.723],
                        [1609372800.723],
                    ],
                    dtype=tf.float64,
                ),
            ),
            (
                tf.constant([[[1], [2], [3]], [[4], [5], [6]]], dtype="int32"),
                "input_4",
                1668335020.786,
                "float32",
                "int64",
                "milliseconds",
                tf.constant(
                    [
                        [
                            [1668335020786],
                            [1668335020786],
                            [1668335020786],
                        ],
                        [
                            [1668335020786],
                            [1668335020786],
                            [1668335020786],
                        ],
                    ],
                    dtype=tf.int64,
                ),
            ),
        ],
    )
    def test_current_unix_timestamp(
        self,
        inputs,
        input_name,
        test_timestamp,
        input_dtype,
        output_dtype,
        unit,
        expected_output,
    ):
        # patch for tf.timestamp() in CurrentUnixTimestampLayer layer with  of 1622745600.0 is 2021-06-03 00:00:00
        with patch(
            "kamae.tensorflow.layers.current_unix_timestamp.tf.timestamp",
            lambda: tf.constant(test_timestamp, dtype=tf.float64),
        ):
            layer = CurrentUnixTimestampLayer(
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
            tf.debugging.assert_equal(expected_output, output_tensor)
