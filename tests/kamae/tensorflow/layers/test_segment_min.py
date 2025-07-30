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

import numpy as np
import pytest
import tensorflow as tf

from kamae.tensorflow.layers import SegmentMinLayer


class TestSegmentMin:
    @pytest.mark.parametrize(
        "inputs, input_dtype, output_dtype, expected_output",
        [
            (
                [
                    tf.constant(
                        [
                            [[2.0], [5.0], [8.0], [float("NaN")], [9.0], [1.0]],
                        ]
                    ),
                    tf.constant(
                        [
                            [[2], [2], [2], [1], [1], [1]],
                        ]
                    ),
                ],
                None,
                tf.float32,
                tf.constant(
                    [
                        [[2.0], [2.0], [2.0], [1.0], [1.0], [1.0]],
                    ],
                    dtype=tf.float32,
                ),
            ),
            # segment variable is a string
            (
                [
                    tf.constant(
                        [
                            [[2], [5], [8], [8], [9], [1]],
                        ]
                    ),
                    tf.constant(
                        [
                            [["2"], ["2"], ["2"], ["1"], ["1"], ["1"]],
                        ]
                    ),
                ],
                None,
                tf.float32,
                tf.constant(
                    [
                        [[2.0], [2.0], [2.0], [1.0], [1.0], [1.0]],
                    ],
                    dtype=tf.float32,
                ),
            ),
        ],
    )
    def test_segment_min(
        self,
        inputs,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        name = "segment_min_test"
        layer = SegmentMinLayer(
            name=name,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )

        output_tensor = layer(inputs)
        assert layer.name == name, "Layer name is not set properly"
        assert (
            output_tensor.dtype == expected_output.dtype
        ), "Output tensor dtype is not the same as expected tensor dtype"
        assert (
            output_tensor.shape == expected_output.shape
        ), "Output tensor shape is not the same as expected tensor shape"
        tf.debugging.assert_equal(output_tensor, expected_output)
