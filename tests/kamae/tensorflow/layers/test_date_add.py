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

from kamae.tensorflow.layers import DateAddLayer


class TestDateAdd:
    @pytest.mark.parametrize(
        "input_tensor, num_days, input_name, input_dtype, output_dtype, expected_output",
        [
            (
                [tf.constant(["2019-01-01"]), tf.constant([10])],
                None,
                "input_1",
                None,
                None,
                tf.constant(["2019-01-11"], dtype=tf.string),
            ),
            # example with timestamp date format
            (
                [
                    tf.constant(["2019-01-01 17:30:12", "2019-02-01 18:00:00"]),
                    tf.constant([-1, 11]),
                ],
                None,
                "input_2",
                None,
                "string",
                tf.constant(["2018-12-31", "2019-02-12"]),
            ),
            # example with [1,3,2] shape input
            (
                [
                    tf.constant(
                        [
                            [
                                ["2020-01-01", "2019-07-25"],
                                ["2021-01-01", "2018-01-31"],
                                ["2020-11-01", "2020-02-10"],
                            ]
                        ]
                    ),
                ],
                -5,
                "input_3",
                "string",
                None,
                tf.constant(
                    [
                        [
                            ["2019-12-27", "2019-07-20"],
                            ["2020-12-27", "2018-01-26"],
                            ["2020-10-27", "2020-02-05"],
                        ]
                    ]
                ),
            ),
            # example with [1,3,2] shape input
            (
                [
                    tf.constant(
                        [
                            [
                                ["2020-01-01", "2019-07-25"],
                                ["2021-01-01", "2018-01-31"],
                                ["2020-11-01", "2020-02-10"],
                            ]
                        ]
                    ),
                    tf.constant(
                        [
                            [
                                [-24, 10],
                                [65, 128],
                                [100, -23],
                            ]
                        ]
                    ),
                ],
                None,
                "input_3",
                None,
                None,
                tf.constant(
                    [
                        [
                            ["2019-12-08", "2019-08-04"],
                            ["2021-03-07", "2018-06-08"],
                            ["2021-02-09", "2020-01-18"],
                        ]
                    ]
                ),
            ),
        ],
    )
    def test_date_add(
        self,
        input_tensor,
        num_days,
        input_name,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = DateAddLayer(
            name=input_name,
            num_days=num_days,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
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
    def test_date_add_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = DateAddLayer(
            name=input_name,
            num_days=1,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)

    @pytest.mark.parametrize(
        "input_dtype, output_dtype",
        [
            (
                "string",
                "string",
            ),
            (
                "string",
                None,
            ),
        ],
    )
    def test_date_add_raises_error_with_multiple_inputs_input_casting(
        self, input_dtype, output_dtype
    ):
        with pytest.raises(ValueError):
            _ = DateAddLayer(
                input_dtype=input_dtype,
                output_dtype=output_dtype,
            )
