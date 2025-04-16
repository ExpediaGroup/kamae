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

from kamae.tensorflow.layers import LogLayer


class TestLog:
    @pytest.mark.parametrize(
        "input_tensor, input_name, alpha, input_dtype, output_dtype, expected_output",
        [
            (
                [tf.constant([1.0, 2.0, 3.0])],
                "input_1",
                1.0,
                None,
                None,
                tf.constant([0.6931472, 1.0986123, 1.3862944]),
            ),
            (
                tf.constant(["5.0", "2.0"]),
                "input_2",
                0.0,
                "float32",
                None,
                tf.constant([1.609438, 0.6931472]),
            ),
            (
                tf.constant([[[1.5, 2.5, 30.0]]]),
                "input_3",
                3.0,
                None,
                "float64",
                tf.constant(
                    [[[1.504077434539795, 1.7047480344772339, 3.4965076446533203]]],
                    dtype="float64",
                ),
            ),
            (
                tf.constant([[7], [4], [3]], dtype="int32"),
                "input_4",
                9.0,
                "float64",
                "float32",
                tf.constant([[2.7725887], [2.5649493], [2.4849067]]),
            ),
        ],
    )
    def test_log(
        self,
        input_tensor,
        input_name,
        alpha,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        # when
        layer = LogLayer(
            name=input_name,
            alpha=alpha,
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
        tf.debugging.assert_near(output_tensor, expected_output)

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
    def test_log_with_bad_types_raises_error(
        self, inputs, input_name, input_dtype, output_dtype
    ):
        # when
        layer = LogLayer(
            name=input_name, input_dtype=input_dtype, output_dtype=output_dtype
        )
        # then
        with pytest.raises(TypeError):
            layer(inputs)
