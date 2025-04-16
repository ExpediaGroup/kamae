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

from kamae.tensorflow.layers import ImputeLayer


class TestImpute:
    @pytest.mark.parametrize(
        "input_tensor, input_name, impute_value, mask_value, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([1.0, 1.0, -999.0]),
                "input_1",
                2.0,
                -999.1,
                "int64",
                "float32",
                tf.constant([1.0, 1.0, -999.0], dtype="float32"),
            ),
            (
                tf.constant([1.0, 1.0, -999.0]),
                "input_1",
                2.0,
                -999.0,
                None,
                None,
                tf.constant([1.0, 1.0, 2.0]),
            ),
            (
                tf.constant(["hello", "nice", "world"]),
                "input_1",
                "goodbye",
                "hello",
                None,
                None,
                tf.constant(["goodbye", "nice", "world"]),
            ),
        ],
    )
    def test_impute_with_mask(
        self,
        input_tensor,
        input_name,
        impute_value,
        mask_value,
        input_dtype,
        output_dtype,
        expected_output,
    ):
        layer = ImputeLayer(
            name=input_name,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            impute_value=impute_value,
            mask_value=mask_value,
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
        if expected_output.dtype.is_floating:
            tf.debugging.assert_near(output_tensor, expected_output)
        else:
            tf.debugging.assert_equal(output_tensor, expected_output)
