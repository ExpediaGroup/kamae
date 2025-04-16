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

from kamae.tensorflow.layers import DateParseLayer


class TestDateParse:
    @pytest.mark.parametrize(
        "inputs, date_part, default_value, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant(["2019-01-01", "2019-02-01", "2019-03-01"]),
                "MonthOfYear",
                None,
                None,
                None,
                tf.constant([1, 2, 3], dtype=tf.int64),
            ),
            (
                tf.constant([["2021-01-01", "2023-08-12", "2020-02-29"]]),
                "DayOfWeek",
                None,
                "string",
                "string",
                tf.constant([["5", "6", "6"]]),
            ),
            (
                tf.constant([["2021-01-01"], ["2023-08-12"], ["2020-02-29"]]),
                "DayOfMonth",
                None,
                None,
                "int32",
                tf.constant([[1], [12], [29]], dtype=tf.int32),
            ),
            (
                tf.constant([["2021-01-01"], ["2023-08-12"], ["2020-02-29"]]),
                "DayOfYear",
                None,
                "string",
                "float32",
                tf.constant([[1.0], [224.0], [60.0]], dtype=tf.float32),
            ),
            (
                tf.constant([["2021-01-01"], ["2023-08-12"], ["2020-02-29"]]),
                "Year",
                None,
                "string",
                "int64",
                tf.constant([[2021], [2023], [2020]], dtype=tf.int64),
            ),
            # In the case of no timestamp
            (
                tf.constant([["2021-01-01"], ["2023-08-12"], ["2020-02-29"]]),
                "Hour",
                None,
                None,
                "float64",
                tf.constant([[0], [0], [0]], dtype=tf.float64),
            ),
            (
                tf.constant([["2021-01-01"], ["2023-08-12"], ["2020-02-29"]]),
                "Minute",
                None,
                None,
                None,
                tf.constant([[0], [0], [0]], dtype=tf.int64),
            ),
            (
                tf.constant([["2021-01-01"], ["2023-08-12"], ["2020-02-29"]]),
                "Second",
                None,
                None,
                "int16",
                tf.constant([[0], [0], [0]], dtype=tf.int16),
            ),
            (
                tf.constant([["2021-01-01"], ["2023-08-12"], ["2020-02-29"]]),
                "Millisecond",
                None,
                None,
                "string",
                tf.constant([["0"], ["0"], ["0"]]),
            ),
            # With timestamp, no Ms
            (
                tf.constant(
                    [
                        ["2021-01-01 23:48:53"],
                        ["2023-08-12 19:28:27"],
                        ["2020-02-29 11:42:32"],
                    ]
                ),
                "Hour",
                None,
                None,
                None,
                tf.constant([[23], [19], [11]], dtype=tf.int64),
            ),
            (
                tf.constant(
                    [
                        ["2021-01-01 23:48:53"],
                        ["2023-08-12 19:28:27"],
                        ["2020-02-29 11:42:32"],
                    ]
                ),
                "Minute",
                None,
                None,
                "int64",
                tf.constant([[48], [28], [42]], dtype=tf.int64),
            ),
            (
                tf.constant(
                    [
                        ["2021-01-01 23:48:53"],
                        ["2023-08-12 19:28:27"],
                        ["2020-02-29 11:42:32"],
                    ]
                ),
                "Second",
                None,
                "string",
                "float16",
                tf.constant([[53], [27], [32]], dtype=tf.float16),
            ),
            (
                tf.constant(
                    [
                        ["2021-01-01 23:48:53"],
                        ["2023-08-12 19:28:27"],
                        ["2020-02-29 11:42:32"],
                    ]
                ),
                "Millisecond",
                None,
                "string",
                None,
                tf.constant([[0], [0], [0]], dtype=tf.int64),
            ),
            # With timestamp, with Ms
            (
                tf.constant(
                    [
                        ["2021-01-01 23:48:53.369"],
                        ["2023-08-12 19:28:27.999"],
                        ["2020-02-29 11:42:32.123"],
                    ]
                ),
                "Millisecond",
                None,
                None,
                None,
                tf.constant([[369], [999], [123]], dtype=tf.int64),
            ),
            (
                tf.constant(["2019-01-01", "2019-02-01", ""]),
                "MonthOfYear",
                -1,
                None,
                None,
                tf.constant([1, 2, -1], dtype=tf.int64),
            ),
            (
                tf.constant([["2021-01-01", "", "2020-02-29"]]),
                "DayOfWeek",
                0,
                "string",
                "string",
                tf.constant([["5", "0", "6"]]),
            ),
            (
                tf.constant([[""], ["2023-08-12"], ["2020-02-29"]]),
                "DayOfMonth",
                -1,
                None,
                "int32",
                tf.constant([[-1], [12], [29]], dtype=tf.int32),
            ),
            (
                tf.constant([["2021-01-01"], [""], ["2020-02-29"]]),
                "DayOfYear",
                10,
                "string",
                "float32",
                tf.constant([[1.0], [10.0], [60.0]], dtype=tf.float32),
            ),
            (
                tf.constant([["2021-01-01"], ["2023-08-12"], ["2020-02-29"]]),
                "Year",
                -1,
                "string",
                "int64",
                tf.constant([[2021], [2023], [2020]], dtype=tf.int64),
            ),
            # In the case of no timestamp
            (
                tf.constant([["2021-01-01"], ["2023-08-12"], ["2020-02-29"]]),
                "Hour",
                -1,
                None,
                "float64",
                tf.constant([[0], [0], [0]], dtype=tf.float64),
            ),
            (
                tf.constant([["2021-01-01"], [""], ["2020-02-29"]]),
                "Minute",
                -1,
                None,
                None,
                tf.constant([[0], [-1], [0]], dtype=tf.int64),
            ),
            (
                tf.constant([["2021-01-01"], ["2023-08-12"], [""]]),
                "Second",
                10,
                None,
                "int16",
                tf.constant([[0], [0], [10]], dtype=tf.int16),
            ),
            (
                tf.constant([["2021-01-01"], ["2023-08-12"], ["2020-02-29"]]),
                "Millisecond",
                10,
                None,
                "string",
                tf.constant([["0"], ["0"], ["0"]]),
            ),
            # With timestamp, no Ms
            (
                tf.constant(
                    [
                        ["2021-01-01 23:48:53"],
                        [""],
                        ["2020-02-29 11:42:32"],
                    ]
                ),
                "Hour",
                -1,
                None,
                None,
                tf.constant([[23], [-1], [11]], dtype=tf.int64),
            ),
            (
                tf.constant(
                    [
                        [""],
                        ["2023-08-12 19:28:27"],
                        ["2020-02-29 11:42:32"],
                    ]
                ),
                "Minute",
                -1,
                None,
                "int64",
                tf.constant([[-1], [28], [42]], dtype=tf.int64),
            ),
            (
                tf.constant(
                    [
                        ["2021-01-01 23:48:53"],
                        ["2023-08-12 19:28:27"],
                        ["2020-02-29 11:42:32"],
                    ]
                ),
                "Second",
                -1,
                "string",
                "float16",
                tf.constant([[53], [27], [32]], dtype=tf.float16),
            ),
            (
                tf.constant(
                    [
                        ["2021-01-01 23:48:53"],
                        ["2023-08-12 19:28:27"],
                        ["2020-02-29 11:42:32"],
                    ]
                ),
                "Millisecond",
                -1,
                "string",
                None,
                tf.constant([[0], [0], [0]], dtype=tf.int64),
            ),
            # With timestamp, with Ms
            (
                tf.constant(
                    [
                        ["2021-01-01 23:48:53.369"],
                        ["2023-08-12 19:28:27.999"],
                        [""],
                    ]
                ),
                "Millisecond",
                -1,
                None,
                None,
                tf.constant([[369], [999], [-1]], dtype=tf.int64),
            ),
        ],
    )
    def test_date_parse(
        self,
        inputs,
        date_part,
        default_value,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        output_tensor = DateParseLayer(
            date_part=date_part,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            default_value=default_value,
        )(inputs)
        assert (
            output_tensor.dtype == expected_output.dtype
        ), "Output tensor dtype is not the same as expected tensor dtype"
        assert (
            output_tensor.shape == expected_output.shape
        ), "Output tensor shape is not the same as expected tensor shape"
        tf.debugging.assert_equal(output_tensor, expected_output)

    @pytest.mark.parametrize(
        "inputs, date_part",
        [
            (
                tf.constant([["2021-01-01"], ["2023-08-12"], ["2020-02-29"]]),
                "TotalDays",  # This shouldn't be supported in this transformer
            ),
        ],
    )
    def test_failed_date_parse(self, inputs, date_part):
        with pytest.raises(ValueError):
            x = DateParseLayer(date_part)(inputs)

    @pytest.mark.parametrize(
        "inputs, input_name, input_dtype, output_dtype",
        [
            (
                tf.constant([1.0, 2.0, 3.0]),
                "input_1",
                None,
                "float64",
            )
        ],
    )
    def test_date_parse_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = DateParseLayer(
            name=input_name,
            date_part="MonthOfYear",
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
