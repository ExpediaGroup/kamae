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

from kamae.keras.core.layers import ArrayContainsLayer


class TestArrayContains:
    @pytest.mark.parametrize(
        "input_tensors, input_name, input_dtype, output_dtype, expected_output",
        [
            (
                [
                    tf.constant([[[1, 2, 3]]]),
                    tf.constant([[[2]]]),
                ],
                "input_1",
                None,
                None,
                tf.constant([[[True]]]),
            ),
            (
                [
                    tf.constant([[[1, 2, 3]]]),
                    tf.constant([[[5]]]),
                ],
                "input_2",
                None,
                None,
                tf.constant([[[False]]]),
            ),
            (
                [
                    tf.constant([[[1, 2, 3]]]),
                    tf.constant([[[2], [9]]]),
                ],
                "input_3",
                None,
                "float64",
                tf.constant([[[1.0], [0.0]]], dtype="float64"),
            ),
            (
                [
                    tf.constant(
                        [
                            [[1.5, 2.5, 3.5]],
                            [[4.5, 5.5, 6.5]],
                        ]
                    ),
                    tf.constant(
                        [
                            [[2.5]],
                            [[7.5]],
                        ]
                    ),
                ],
                "input_4",
                None,
                None,
                tf.constant([[[True]], [[False]]]),
            ),
            (
                [
                    tf.constant([["1", "2", "3"]]),
                    tf.constant([["2"]]),
                ],
                "input_5",
                "int64",
                None,
                tf.constant([[True]]),
            ),
            (
                # Integer array and integer scalar value, boolean output.
                [
                    tf.constant([[[10, 20, 30]]], dtype="int64"),
                    tf.constant([[[20]]], dtype="int64"),
                ],
                "input_6",
                None,
                None,
                tf.constant([[[True]]]),
            ),
            (
                # Integer array and integer scalar value, cast to int output.
                [
                    tf.constant([[[10, 20, 30]]], dtype="int32"),
                    tf.constant([[[40]]], dtype="int32"),
                ],
                "input_7",
                None,
                "int32",
                tf.constant([[[0]]], dtype="int32"),
            ),
        ],
    )
    def test_array_contains(
        self,
        input_tensors,
        input_name,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = ArrayContainsLayer(
            name=input_name,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        output_tensor = layer(input_tensors)
        # then
        assert layer.name == input_name, "Layer name is not set properly"
        assert (
            output_tensor.dtype == expected_output.dtype
        ), "Output tensor dtype is not the same as expected tensor dtype"
        assert (
            output_tensor.shape == expected_output.shape
        ), "Output tensor shape is not the same as expected tensor shape"

        tf.debugging.assert_equal(
            tf.cast(output_tensor, "float64"), tf.cast(expected_output, "float64")
        )

    @pytest.mark.parametrize(
        "input_tensors, axis, keepdims, expected_output",
        [
            (
                # Default: search over last axis, keep the collapsed dimension.
                [
                    tf.constant([[[1, 2, 3]]]),
                    tf.constant([[[2]]]),
                ],
                -1,
                True,
                tf.constant([[[True]]]),
            ),
            (
                # keepdims=False drops the collapsed axis.
                [
                    tf.constant([[[1, 2, 3]]]),
                    tf.constant([[[2]]]),
                ],
                -1,
                False,
                tf.constant([[True]]),
            ),
            (
                # Search over a non-final axis.
                [
                    tf.constant([[[1], [2], [3]]]),
                    tf.constant([[[2]]]),
                ],
                1,
                True,
                tf.constant([[[True]]]),
            ),
        ],
    )
    def test_array_contains_axis_keepdims(
        self,
        input_tensors,
        axis,
        keepdims,
        expected_output,
    ):
        # when
        layer = ArrayContainsLayer(axis=axis, keepdims=keepdims)
        output_tensor = layer(input_tensors)
        # then
        assert (
            output_tensor.shape == expected_output.shape
        ), "Output tensor shape is not the same as expected tensor shape"
        tf.debugging.assert_equal(output_tensor, expected_output)

    @pytest.mark.parametrize(
        "input_tensors",
        [
            (
                [
                    # Too many input tensors
                    tf.constant([[[1, 2, 3]]]),
                    tf.constant([[[2]]]),
                    tf.constant([[[3]]]),
                ],
            ),
            (
                [
                    # Not enough input tensors
                    tf.constant([[[1, 2, 3]]]),
                ],
            ),
        ],
    )
    def test_array_contains_raises_error(self, input_tensors):
        # when
        layer = ArrayContainsLayer()
        # then
        with pytest.raises(ValueError):
            layer(input_tensors)
