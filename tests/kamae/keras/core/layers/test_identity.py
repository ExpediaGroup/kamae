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

from kamae.keras.core.layers.identity import IdentityLayer


class TestIdentity:
    """Tests for portable IdentityLayer (numeric operations only)"""

    @pytest.mark.parametrize(
        "input_tensor, input_name, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([1, 2, 3], dtype="float32"),
                "input_1",
                "float64",
                None,
                tf.constant([1, 2, 3], dtype="float64"),
            ),
            (
                tf.constant([[1, 2, 3], [4, 5, 6]], dtype="float32"),
                "input_4",
                "int32",
                "int32",
                tf.constant([[1, 2, 3], [4, 5, 6]], dtype="int32"),
            ),
            (
                tf.constant([[1, 2, 3], [4, 5, 6]], dtype="int32"),
                "input_5",
                "float32",
                "float64",
                tf.constant([[1, 2, 3], [4, 5, 6]], dtype="float64"),
            ),
            (
                tf.constant([1.5, 2.5, 3.5], dtype="float32"),
                "input_float",
                None,
                None,
                tf.constant([1.5, 2.5, 3.5], dtype="float32"),
            ),
            (
                tf.constant([10, 20, 30], dtype="int64"),
                "input_int64",
                None,
                "int32",
                tf.constant([10, 20, 30], dtype="int32"),
            ),
        ],
    )
    def test_identity(
        self, input_tensor, input_name, input_dtype, output_dtype, expected_output
    ):
        """Test identity layer with various numeric dtypes"""
        # when
        layer = IdentityLayer(
            name=input_name, input_dtype=input_dtype, output_dtype=output_dtype
        )
        output_tensor = layer(input_tensor)
        # then
        assert layer.name == input_name, "Layer name is not set properly"
        assert keras.backend.standardize_dtype(
            expected_output.dtype
        ) == keras.backend.standardize_dtype(
            output_tensor.dtype
        ), "Output tensor dtype is not the same as expected tensor dtype"
        assert (
            expected_output.shape == output_tensor.shape
        ), "Output tensor shape is not the same as expected tensor shape"
        # Use assert_equal for exact comparison (works with int and float)
        tf.debugging.assert_equal(expected_output, output_tensor)

    def test_identity_no_casting(self):
        """Test identity without dtype casting"""
        layer = IdentityLayer(name="test_identity")
        x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        output = layer(x)
        tf.debugging.assert_equal(x, output)
        assert keras.backend.standardize_dtype(
            x.dtype
        ) == keras.backend.standardize_dtype(output.dtype)

    def test_identity_serialization(self):
        """Test identity layer serialization"""
        original = IdentityLayer(
            name="test_identity", input_dtype="float32", output_dtype="float64"
        )
        config = original.get_config()

        recreated = IdentityLayer.from_config(config)
        assert recreated.name == original.name
        assert recreated._input_dtype == original._input_dtype
        assert recreated._output_dtype == original._output_dtype

        # Test that recreated layer works
        x = tf.constant([[1.0, 2.0]])
        output = recreated(x)
        assert keras.backend.standardize_dtype(output.dtype) == "float64"

    def test_identity_with_list_input(self):
        """Test identity layer with list input (should take first element)"""
        layer = IdentityLayer(name="test_identity")
        x = tf.constant([1.0, 2.0, 3.0])
        output = layer([x])  # Pass as list
        tf.debugging.assert_equal(x, output)

    def test_identity_with_multiple_tensors_raises(self):
        """Test identity layer raises error with multiple tensors"""
        layer = IdentityLayer(name="test_identity")
        x1 = tf.constant([1.0, 2.0])
        x2 = tf.constant([3.0, 4.0])
        with pytest.raises(ValueError, match="single tensor"):
            layer([x1, x2])
