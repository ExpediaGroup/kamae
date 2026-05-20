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

"""Tests for BaseLayer"""

from typing import Any, List, Optional

import keras
import pytest
import tensorflow as tf
from keras import ops

from kamae.keras.core.backend import ALL_BACKENDS
from kamae.keras.core.base import BaseLayer
from kamae.keras.core.utils.input_utils import enforce_single_tensor_input


@keras.saving.register_keras_serializable(package="kamae_test")
class MockLayer(BaseLayer):
    """Mock layer for testing BaseLayer"""

    supported_backends = ALL_BACKENDS
    jit_compatible = False

    @property
    def compatible_dtypes(self) -> Optional[List[str]]:
        return None

    @enforce_single_tensor_input
    def _call(self, inputs, **kwargs: Any):
        return ops.multiply(inputs, 2.0)


@keras.saving.register_keras_serializable(package="kamae_test")
class MockLayerWithCompatibleDtypes(BaseLayer):
    """Mock layer with specific compatible dtypes"""

    supported_backends = ALL_BACKENDS
    jit_compatible = False

    @property
    def compatible_dtypes(self) -> Optional[List[str]]:
        return ["float32", "float64"]

    @enforce_single_tensor_input
    def _call(self, inputs, **kwargs: Any):
        return ops.multiply(inputs, 2.0)


class TestBaseLayer:
    """Test suite for BaseLayer"""

    def test_instantiation(self):
        """Test layer instantiation"""
        layer = MockLayer(name="test_layer")
        assert layer.name == "test_layer"
        assert layer._input_dtype is None
        assert layer._output_dtype is None

    def test_instantiation_with_dtypes(self):
        """Test layer instantiation with dtype specification"""
        layer = MockLayer(
            name="test_layer", input_dtype="float32", output_dtype="float64"
        )
        assert layer._input_dtype == "float32"
        assert layer._output_dtype == "float64"

    def test_basic_call(self):
        """Test basic layer call"""
        layer = MockLayer()
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        output = layer(x)
        expected = tf.constant([[2.0, 4.0], [6.0, 8.0]])
        tf.debugging.assert_near(output, expected)

    def test_output_dtype_casting(self):
        """Test output dtype casting"""
        layer = MockLayer(output_dtype="float64")
        x = tf.constant([[1.0, 2.0]], dtype=tf.float32)
        output = layer(x)
        assert keras.backend.standardize_dtype(output.dtype) == "float64"

    def test_input_dtype_casting(self):
        """Test input dtype casting"""
        layer = MockLayer(input_dtype="float32")
        x = tf.constant([[1, 2]], dtype=tf.int32)
        output = layer(x)
        # Layer should cast int32 to float32, compute, and return float32
        assert keras.backend.standardize_dtype(output.dtype) == "float32"

    def test_input_output_dtype_casting(self):
        """Test combined input and output dtype casting"""
        layer = MockLayer(input_dtype="float32", output_dtype="float64")
        x = tf.constant([[1, 2]], dtype=tf.int32)
        output = layer(x)
        # Should cast int32 -> float32 (input), compute, cast -> float64 (output)
        assert keras.backend.standardize_dtype(output.dtype) == "float64"

    def test_compatible_dtypes_validation_pass(self):
        """Test compatible dtypes validation - should pass"""
        layer = MockLayerWithCompatibleDtypes()
        x = tf.constant([[1.0, 2.0]], dtype=tf.float32)
        output = layer(x)  # Should not raise
        assert output is not None

    def test_compatible_dtypes_validation_fail(self):
        """Test compatible dtypes validation - should fail"""
        layer = MockLayerWithCompatibleDtypes()
        x = tf.constant([[1, 2]], dtype=tf.int32)
        with pytest.raises(TypeError, match="not a compatible dtype"):
            layer(x)

    def test_compatible_dtypes_with_input_casting(self):
        """Test compatible dtypes validation with input casting"""
        layer = MockLayerWithCompatibleDtypes(input_dtype="float32")
        x = tf.constant([[1, 2]], dtype=tf.int32)
        # Should cast int32 to float32 first, then pass validation
        output = layer(x)
        assert output is not None

    def test_invalid_input_dtype_for_layer(self):
        """Test that specifying incompatible input_dtype raises error"""
        with pytest.raises(ValueError, match="not a compatible dtype"):
            layer = MockLayerWithCompatibleDtypes(input_dtype="int32")
            x = tf.constant([[1, 2]], dtype=tf.int32)
            layer(x)

    def test_force_cast_float_input_float_constant(self):
        """Test force cast with float input and float constant"""
        layer = MockLayer()
        x = tf.constant([1.5, 2.5], dtype=tf.float32)
        cast_input, cast_const = layer._force_cast_to_compatible_numeric_type(x, 3.14)
        assert keras.backend.standardize_dtype(cast_input.dtype) == "float32"
        assert keras.backend.standardize_dtype(cast_const.dtype) == "float32"
        tf.debugging.assert_near(cast_const, tf.constant(3.14, dtype=tf.float32))

    def test_force_cast_int_input_int_constant(self):
        """Test force cast with int input and int constant"""
        layer = MockLayer()
        x = tf.constant([1, 2, 3], dtype=tf.int32)
        cast_input, cast_const = layer._force_cast_to_compatible_numeric_type(x, 5)
        assert keras.backend.standardize_dtype(cast_input.dtype) == "int32"
        assert keras.backend.standardize_dtype(cast_const.dtype) == "int32"
        tf.debugging.assert_equal(cast_const, tf.constant(5, dtype=tf.int32))

    def test_force_cast_int_input_float_constant(self):
        """Test force cast with int input and float constant - should promote to float"""
        layer = MockLayer()
        x = tf.constant([1, 2, 3], dtype=tf.int64)
        cast_input, cast_const = layer._force_cast_to_compatible_numeric_type(x, 3.14)
        # Should promote to float64
        assert keras.backend.standardize_dtype(cast_input.dtype) == "float64"
        assert keras.backend.standardize_dtype(cast_const.dtype) == "float64"

    def test_force_cast_int_input_integer_valued_float(self):
        """Test force cast with int input and integer-valued float - should keep as int"""
        layer = MockLayer()
        x = tf.constant([1, 2, 3], dtype=tf.int32)
        cast_input, cast_const = layer._force_cast_to_compatible_numeric_type(x, 5.0)
        # 5.0 is integer-valued, so should keep as int32
        assert keras.backend.standardize_dtype(cast_input.dtype) == "int32"
        assert keras.backend.standardize_dtype(cast_const.dtype) == "int32"
        tf.debugging.assert_equal(cast_const, tf.constant(5, dtype=tf.int32))

    def test_get_config(self):
        """Test get_config returns correct configuration"""
        layer = MockLayer(
            name="test_layer", input_dtype="float32", output_dtype="float64"
        )
        config = layer.get_config()
        assert config["name"] == "test_layer"
        assert config["input_dtype"] == "float32"
        assert config["output_dtype"] == "float64"

    def test_serialization_round_trip(self):
        """Test layer can be serialized and deserialized"""
        original = MockLayer(
            name="test_layer", input_dtype="float32", output_dtype="float64"
        )
        config = original.get_config()
        recreated = MockLayer.from_config(config)

        assert recreated.name == original.name
        assert recreated._input_dtype == original._input_dtype
        assert recreated._output_dtype == original._output_dtype

        # Test that recreated layer works
        x = tf.constant([[1.0, 2.0]])
        output = recreated(x)
        assert keras.backend.standardize_dtype(output.dtype) == "float64"

    def test_autocast_disabled(self):
        """Test that autocast is disabled"""
        layer = MockLayer()
        assert layer._autocast is False
        assert layer._convert_input_args is False
