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

from kamae.tensorflow.layers import CosineSimilarityLayer


class TestCosineSimilarity:
    @pytest.mark.parametrize(
        "input_tensors, input_name, axis, keepdims, input_dtype, output_dtype, expected_output",
        [
            (
                [
                    tf.constant(
                        [
                            [40.7486, 45.8797, 40.7486],
                        ]
                    ),
                    tf.constant([[-73.9864, 23.0297, 67.90784]]),
                ],
                "input_1",
                -1,
                False,
                None,
                None,
                tf.constant([0.10658249]),
            ),
            (
                [
                    tf.constant(
                        [
                            [40.7486, 45.8797, 40.7486],
                            [58.7834, 23.0297, 67.90784],
                            [-13.89, 67.057, -0.12],
                            [40.7486, 45.8797, 40.7486],
                        ]
                    ),
                    tf.constant(
                        [
                            [-73.9864, 23.0297, 67.90784],
                            [23.0297, -12.90784, 34.7834],
                            [58.057, -7.12, -78.089],
                            [40.7486, 45.8797, 40.7486],
                        ]
                    ),
                ],
                "input_2",
                -1,
                False,
                "float64",
                None,
                tf.constant(
                    [0.10658249, 0.84431064, -0.19075146, 1.000000], dtype="float64"
                ),
            ),
            (
                [
                    tf.constant(
                        [
                            [67.90784, 45.8797, 40.7486, 40.7486],
                            [58.7834, 23.0297, 67.90784, 40.7486],
                            [-13.89, 67.057, -0.12, 40.7486],
                        ]
                    ),
                    tf.constant(
                        [
                            [-22.456, 24.90784, 90.8797, 40.7486],
                            [23.0297, -12.90784, 34.7834, 40.7486],
                            [58.057, -7.12, -78.089, 40.7486],
                        ]
                    ),
                ],
                "input_3",
                0,
                True,
                None,
                "float32",
                tf.constant([[-0.16206239, 0.15057813, 0.61478025, 1.00000]]),
            ),
            (
                [
                    tf.constant(
                        [
                            ["67.90784", "45.8797", "40.7486", "40.7486"],
                            ["58.7834", "23.0297", "67.90784", "40.7486"],
                            ["-13.89", "67.057", "-0.12", "40.7486"],
                        ]
                    ),
                    tf.constant(
                        [
                            [-22.456, 24.90784, 90.8797, 40.7486],
                            [23.0297, -12.90784, 34.7834, 40.7486],
                            [58.057, -7.12, -78.089, 40.7486],
                        ]
                    ),
                ],
                "input_4",
                0,
                False,
                "float64",
                "float32",
                tf.constant([-0.16206239, 0.15057813, 0.61478025, 1.00000]),
            ),
            (
                [
                    tf.constant(
                        [
                            [[67.90784, 45.8797, 40.7486, 40.7486]],
                            [[58.7834, 23.0297, 67.90784, 40.7486]],
                            [[-13.89, 67.057, -0.12, 40.7486]],
                        ]
                    ),
                    tf.constant(
                        [
                            [[-22.456, 24.90784, 90.8797, 40.7486]],
                            [[23.0297, -12.90784, 34.7834, 40.7486]],
                            [[58.057, -7.12, -78.089, 40.7486]],
                        ]
                    ),
                ],
                "input_5",
                -1,
                True,
                "float32",
                "float32",
                tf.constant([[[0.47313264]], [[0.83961886]], [[0.045808047]]]),
            ),
        ],
    )
    def test_cosine_similarity(
        self,
        input_tensors,
        input_name,
        input_dtype,
        output_dtype,
        expected_output,
        axis,
        keepdims,
    ):
        # when
        layer = CosineSimilarityLayer(
            name=input_name,
            axis=axis,
            keepdims=keepdims,
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
                    tf.constant([40.7486]),
                    tf.constant([-273.9864]),
                    tf.constant([58.7834]),
                    tf.constant([23.0297]),
                ],
            ),
            (
                [
                    # Not enough input tensors
                    tf.constant([[[40.7486, 98.7834]]]),
                ],
            ),
        ],
    )
    def test_cosine_similarity_raises_error(self, input_tensors):
        # when
        layer = CosineSimilarityLayer()
        # then
        with pytest.raises(ValueError):
            layer(input_tensors)

    @pytest.mark.parametrize(
        "inputs, input_name, input_dtype, output_dtype",
        [
            (
                tf.constant(["1.0", "2.0", "3.0"], dtype="string"),
                "input_1",
                None,
                "float64",
            )
        ],
    )
    def test_cosine_similarity_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = CosineSimilarityLayer(
            name=input_name, input_dtype=input_dtype, output_dtype=output_dtype
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
