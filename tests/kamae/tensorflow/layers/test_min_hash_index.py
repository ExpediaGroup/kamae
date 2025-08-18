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

from kamae.tensorflow.layers import MinHashIndexLayer


class TestMinHashIndex:
    @pytest.mark.parametrize(
        "input_tensor, input_name, num_permutations, axis, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant(
                    [
                        ["Mon", "Tue", "Wed"],
                        ["Thu", "Fri", "Sat"],
                        ["Sun", "Mon", "Tue"],
                    ]
                ),
                "input_1",
                10,
                -1,
                None,
                "int64",
                tf.constant(
                    [
                        [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                        [1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
                        [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    ],
                    dtype=tf.int64,
                ),
            ),
            (
                tf.constant([[["Mon", "Tue"], ["Wed", "Thu"]]]),
                "input_2",
                5,
                -2,
                None,
                "int32",
                tf.constant(
                    [[[0, 1], [1, 1], [1, 0], [0, 0], [0, 1]]],
                    dtype=tf.int32,
                ),
            ),
            (
                tf.constant([[["Sun"], ["Sat"], ["Fri"]]]),
                "input_3",
                3,
                1,
                "string",
                "int16",
                tf.constant([[[0], [1], [1]]], dtype=tf.int16),
            ),
        ],
    )
    def test_min_hash_index(
        self,
        input_tensor,
        input_name,
        num_permutations,
        axis,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = MinHashIndexLayer(
            name=input_name,
            num_permutations=num_permutations,
            axis=axis,
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
        "input_list, input_name, num_permutations, mask_value",
        [
            (
                ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "-1", "-1", "-1"],
                "input_1",
                10,
                "-1",
            ),
            (
                ["a", "b", "c", "d", "NULL", "NULL", "NULL"],
                "input_2",
                500,
                "NULL",
            ),
            (
                [
                    "apple",
                    "banana",
                    "NOT_A_FRUIT",
                    "cherry",
                    "date",
                    "fig",
                    "grape",
                    "kiwi",
                    "NOT_A_FRUIT",
                    "NOT_A_FRUIT",
                    "NOT_A_FRUIT",
                ],
                "input_3",
                100,
                "NOT_A_FRUIT",
            ),
        ],
    )
    def test_min_hash_index_with_mask_equals_no_mask(
        self, input_list, input_name, num_permutations, mask_value
    ):
        # when
        layer = MinHashIndexLayer(
            name=input_name,
            num_permutations=num_permutations,
            mask_value=mask_value,
        )
        output_tensor_with_mask = layer(tf.constant(input_list))
        output_tensor_wo_mask = layer(
            tf.constant([inp for inp in input_list if inp != mask_value])
        )

        # then
        assert (
            output_tensor_with_mask.dtype == output_tensor_wo_mask.dtype
        ), "Output tensor dtype is not the same as expected tensor dtype"
        assert (
            output_tensor_with_mask.shape == output_tensor_wo_mask.shape
        ), "Output tensor shape is not the same as expected tensor shape"
        tf.debugging.assert_equal(output_tensor_with_mask, output_tensor_wo_mask)

    @pytest.mark.parametrize(
        "inputs, input_name, input_dtype, output_dtype",
        [
            (
                tf.constant([1.0, 2.0, 3.0], dtype="float32"),
                "input_1",
                None,
                "float64",
            )
        ],
    )
    def test_min_hash_index_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = MinHashIndexLayer(
            name=input_name,
            num_permutations=100,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
