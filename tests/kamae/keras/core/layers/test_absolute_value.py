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

import keras
import pytest
import tensorflow as tf

from kamae.keras.core.layers.absolute_value import AbsoluteValueLayer


class TestAbsoluteValue:
    """Tests for portable AbsoluteValueLayer"""

    @pytest.mark.parametrize(
        "input_tensor, expected_output",
        [
            (
                tf.constant([-1.0, -2.0, 3.0], dtype=tf.float32),
                tf.constant([1.0, 2.0, 3.0], dtype=tf.float32),
            ),
            (
                tf.constant([[-1, -2], [3, -4]], dtype=tf.int32),
                tf.constant([[1, 2], [3, 4]], dtype=tf.int32),
            ),
            (
                tf.constant([-5, 0, 5], dtype=tf.int64),
                tf.constant([5, 0, 5], dtype=tf.int64),
            ),
            (
                tf.constant([1.5, -2.5, 3.5], dtype=tf.float64),
                tf.constant([1.5, 2.5, 3.5], dtype=tf.float64),
            ),
        ],
    )
    def test_absolute_value(self, input_tensor, expected_output):
        """Test absolute value layer with various dtypes"""
        layer = AbsoluteValueLayer(name="test_abs")
        output = layer(input_tensor)
        tf.debugging.assert_equal(output, expected_output)
        assert keras.backend.standardize_dtype(
            output.dtype
        ) == keras.backend.standardize_dtype(input_tensor.dtype)

    def test_absolute_value_with_dtype_casting(self):
        """Test absolute value with dtype casting"""
        layer = AbsoluteValueLayer(
            name="test_abs", input_dtype="float32", output_dtype="float64"
        )
        x = tf.constant([-1, -2, 3], dtype=tf.int32)
        output = layer(x)
        expected = tf.constant([1.0, 2.0, 3.0], dtype=tf.float64)
        tf.debugging.assert_near(output, expected)
        assert keras.backend.standardize_dtype(output.dtype) == "float64"

    def test_absolute_value_serialization(self):
        """Test serialization round-trip"""
        original = AbsoluteValueLayer(
            name="test_abs", input_dtype="float32", output_dtype="float64"
        )
        config = original.get_config()
        recreated = AbsoluteValueLayer.from_config(config)

        assert recreated.name == original.name
        assert recreated._input_dtype == original._input_dtype
        assert recreated._output_dtype == original._output_dtype

        # Test that recreated layer works
        x = tf.constant([-1.0, -2.0, 3.0])
        output = recreated(x)
        assert keras.backend.standardize_dtype(output.dtype) == "float64"

    def test_absolute_value_incompatible_dtype_raises(self):
        """Test that incompatible dtype raises error"""
        layer = AbsoluteValueLayer(name="test_abs")
        # bfloat16 is not in compatible_dtypes
        x = tf.constant([-1.0, -2.0], dtype=tf.bfloat16)
        with pytest.raises(TypeError, match="not a compatible dtype"):
            layer(x)

    def test_absolute_value_complex(self):
        """Test absolute value with complex numbers"""
        layer = AbsoluteValueLayer(name="test_abs_complex")
        x = tf.constant([3 + 4j, -5 + 12j], dtype=tf.complex64)
        output = layer(x)
        expected = tf.constant([5.0, 13.0], dtype=tf.float32)
        tf.debugging.assert_near(output, expected)
