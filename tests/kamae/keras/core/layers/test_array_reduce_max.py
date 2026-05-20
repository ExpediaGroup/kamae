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

from kamae.keras.core.layers import ArrayReduceMaxLayer


class TestArrayReduceMax:
    @pytest.mark.parametrize(
        "input_tensor, name, default_value, expected_output",
        [
            (
                tf.constant([[1.0, 3.0, 2.0], [5.0, 4.0, 6.0]]),
                "basic_max",
                0.0,
                tf.constant([3.0, 6.0]),
            ),
            (
                tf.constant([[-5.0, -1.0, -3.0]]),
                "negative_max",
                0.0,
                tf.constant([-1.0]),
            ),
            (
                tf.constant([[7.0]]),
                "single_element",
                0.0,
                tf.constant([7.0]),
            ),
            (
                tf.constant([[float("nan"), 2.0, 3.0]]),
                "nan_handling",
                -1.0,
                tf.constant([-1.0]),
            ),
            (
                tf.constant([[float("nan"), float("nan")]]),
                "all_nan",
                -99.0,
                tf.constant([-99.0]),
            ),
            (
                tf.constant([[1.0, 2.0, 3.0]]),
                "custom_default",
                42.0,
                tf.constant([3.0]),
            ),
        ],
    )
    def test_array_reduce_max(self, input_tensor, name, default_value, expected_output):
        layer = ArrayReduceMaxLayer(name=name, default_value=default_value)
        output_tensor = layer(input_tensor)

        assert layer.name == name
        assert output_tensor.shape == expected_output.shape
        tf.debugging.assert_near(output_tensor, expected_output, atol=1e-6)

    def test_array_reduce_max_batch(self):
        input_tensor = tf.constant([[1.0, 5.0, 3.0], [9.0, 2.0, 7.0], [4.0, 4.0, 4.0]])
        layer = ArrayReduceMaxLayer(name="batch_test")
        output_tensor = layer(input_tensor)
        expected = tf.constant([5.0, 9.0, 4.0])
        tf.debugging.assert_near(output_tensor, expected, atol=1e-6)

    def test_get_config(self):
        layer = ArrayReduceMaxLayer(name="config_test", default_value=5.0)
        config = layer.get_config()
        assert config["default_value"] == 5.0
        assert config["name"] == "config_test"
