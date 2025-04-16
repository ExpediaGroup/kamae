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

from kamae.tensorflow.layers import HaversineDistanceLayer


class TestHaversineDistance:
    @pytest.mark.parametrize(
        "input_tensors, input_name, lat_lon_constant, input_dtype, output_dtype, expected_output",
        [
            (
                [
                    tf.constant(["40.7486"]),
                    tf.constant(["-73.9864"]),
                    tf.constant(["58.7834"]),
                    tf.constant(["23.0297"]),
                ],
                "input_1",
                None,
                "float64",
                None,
                tf.constant([6595.852599811664], dtype=tf.float64),
            ),
            (
                [
                    tf.constant([40.7486]),
                    tf.constant([-73.9864]),
                ],
                "input_1_constant",
                [58.7834, 23.0297],
                None,
                None,
                tf.constant([6595.852323146841], dtype=tf.float64),
            ),
            (
                [
                    tf.constant([0.0], dtype=tf.float32),
                    tf.constant([180.0], dtype=tf.float32),
                    tf.constant([0.0], dtype=tf.float32),
                    tf.constant([-180.0], dtype=tf.float32),
                ],
                "input_2",
                None,
                "float64",
                None,
                tf.constant([0.0], dtype=tf.float64),
            ),
            (
                [
                    tf.constant([0.0], dtype=tf.float64),
                    tf.constant([180.0], dtype=tf.float64),
                ],
                "input_2_constant",
                [0.0, -180.0],
                None,
                None,
                tf.constant([0.0], dtype=tf.float64),
            ),
            (
                [
                    tf.constant(
                        [
                            [67.90784, 45.8797, 40.7486],
                            [58.7834, 23.0297, 67.90784],
                            [-13.89, 67.057, -0.12],
                        ]
                    ),
                    tf.constant(
                        [
                            [-22.456, 24.90784, 90.8797],
                            [23.0297, -12.90784, 34.7834],
                            [58.057, -7.12, -78.089],
                        ]
                    ),
                ],
                "input_3",
                [0.0, -180.0],
                "float32",
                "float64",
                tf.constant(
                    [
                        [12269.172778, 14361.32897127, 9933.4394774],
                        [13175.2262778, 17098.72741762, 12008.23056797],
                        [13443.93498902, 12537.90685054, 11331.98356799],
                    ],
                    dtype=tf.float64,
                ),
            ),
        ],
    )
    def test_haversine_distance(
        self,
        input_tensors,
        input_name,
        lat_lon_constant,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = HaversineDistanceLayer(
            name=input_name,
            lat_lon_constant=lat_lon_constant,
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

        if expected_output.dtype == tf.string:
            tf.debugging.assert_equal(output_tensor, expected_output)
        else:
            tf.debugging.assert_near(output_tensor, expected_output, atol=1e-6)

    @pytest.mark.parametrize(
        "input_tensors",
        [
            (
                [
                    tf.constant([40.7486], dtype=tf.float64),
                    # Longitude is out of range
                    tf.constant([-273.9864], dtype=tf.float64),
                    tf.constant([58.7834], dtype=tf.float64),
                    tf.constant([23.0297], dtype=tf.float64),
                ],
            ),
            (
                [
                    # Latitude is out of range
                    tf.constant([[[40.7486, 98.7834]]], dtype=tf.float64),
                    tf.constant([[[-73.9864, 23.0297]]], dtype=tf.float64),
                    tf.constant([[[-73.9864, 23.0297]]], dtype=tf.float64),
                    tf.constant([[[-73.9864, 23.0297]]], dtype=tf.float64),
                ],
            ),
        ],
    )
    def test_haversine_distance_raises_error(self, input_tensors):
        # when
        layer = HaversineDistanceLayer()
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
    def test_haversine_distance_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = HaversineDistanceLayer(
            name=input_name, input_dtype=input_dtype, output_dtype=output_dtype
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
