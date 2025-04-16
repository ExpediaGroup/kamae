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

import pytest
import tensorflow as tf

from kamae.tensorflow.layers import DateDiffLayer


class TestDateDiff:
    @pytest.mark.parametrize(
        "input_tensor, input_name, input_dtype, output_dtype, default_value, expected_output",
        [
            (
                [tf.constant(["2019-01-01"]), tf.constant(["2019-02-01"])],
                "input_1",
                None,
                None,
                None,
                tf.constant([31], dtype=tf.int64),
            ),
            # example with timestamp date format
            (
                [
                    tf.constant(["2019-01-01 17:30:12"]),
                    tf.constant(["2019-02-01 18:00:00"]),
                ],
                "input_2",
                None,
                "string",
                None,
                tf.constant(["31"]),
            ),
            # example with [1,3,2] shape input
            (
                [
                    tf.constant(
                        [
                            [
                                ["2019-01-01", "2019-01-01"],
                                ["2019-01-01", "2019-01-01"],
                                ["2020-01-01", "2020-01-01"],
                            ]
                        ]
                    ),
                    tf.constant(
                        [
                            [
                                ["2019-02-01", "2019-02-01"],
                                ["2020-02-01", "2020-02-29"],
                                ["2020-02-01", "2020-02-29"],
                            ]
                        ]
                    ),
                ],
                "input_3",
                "string",
                None,
                None,
                tf.constant([[[31, 31], [396, 424], [31, 59]]], dtype=tf.int64),
            ),
            (
                [tf.constant(["2019-01-01"]), tf.constant([""])],
                "input_1",
                None,
                None,
                -1,
                tf.constant([-1], dtype=tf.int64),
            ),
            # example with timestamp date format
            (
                [
                    tf.constant(["2019-01-01 17:30:12"]),
                    tf.constant([""]),
                ],
                "input_2",
                None,
                "string",
                -1,
                tf.constant(["-1"]),
            ),
            # example with [1,3,2] shape input
            (
                [
                    tf.constant(
                        [
                            [
                                ["2019-01-01", "2019-01-01"],
                                ["", "2019-01-01"],
                                ["2020-01-01", ""],
                            ]
                        ]
                    ),
                    tf.constant(
                        [
                            [
                                ["2019-02-01", ""],
                                ["2020-02-01", "2020-02-29"],
                                ["2020-02-01", "2020-02-29"],
                            ]
                        ]
                    ),
                ],
                "input_3",
                "string",
                None,
                -1,
                tf.constant([[[31, -1], [-1, 424], [31, -1]]], dtype=tf.int64),
            ),
        ],
    )
    def test_date_diff(
        self,
        input_tensor,
        input_name,
        input_dtype,
        output_dtype,
        default_value,
        expected_output,
    ):
        # when
        layer = DateDiffLayer(
            name=input_name,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            default_value=default_value,
        )
        output_tensor = layer(input_tensor)
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
        "inputs, input_name, input_dtype, output_dtype",
        [
            (
                tf.constant([1.0, 2.0, 3.0], dtype="float32"),
                "input_1",
                None,
                "string",
            )
        ],
    )
    def test_date_diff_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = DateDiffLayer(
            name=input_name, input_dtype=input_dtype, output_dtype=output_dtype
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
