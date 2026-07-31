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
                tf.constant([[[1.0]]]),
            ),
            (
                [
                    tf.constant([[[1, 2, 3]]]),
                    tf.constant([[[5]]]),
                ],
                "input_2",
                None,
                None,
                tf.constant([[[0.0]]]),
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
                tf.constant([[[1.0]], [[0.0]]]),
            ),
            (
                [
                    tf.constant([["1", "2", "3"]]),
                    tf.constant([["2"]]),
                ],
                "input_5",
                "int64",
                None,
                tf.constant([[1.0]]),
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

        tf.debugging.assert_near(output_tensor, expected_output, atol=1e-6)

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
