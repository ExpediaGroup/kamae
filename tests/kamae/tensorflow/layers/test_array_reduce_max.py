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
import tensorflow as tf

from kamae.tensorflow.layers import ArrayReduceMaxLayer


class TestArrayReduceMaxLayer:
    def test_returns_maximum_of_each_row(self):
        layer = ArrayReduceMaxLayer()
        inputs = tf.constant([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]])

        result = layer(inputs).numpy()

        np.testing.assert_array_almost_equal(result, [3.0, 5.0])

    def test_negative_values(self):
        layer = ArrayReduceMaxLayer()
        inputs = tf.constant([[-3.0, -1.0, -2.0], [-10.0, -5.0, -7.0]])

        result = layer(inputs).numpy()

        np.testing.assert_array_almost_equal(result, [-1.0, -5.0])

    def test_single_element_array(self):
        layer = ArrayReduceMaxLayer()
        inputs = tf.constant([[42.0]])

        result = layer(inputs).numpy()

        np.testing.assert_array_almost_equal(result, [42.0])

    def test_default_value_returned_for_nan_input(self):
        layer = ArrayReduceMaxLayer(default_value=-99.0)
        inputs = tf.constant([[float("nan"), float("nan")]])

        result = layer(inputs).numpy()

        np.testing.assert_array_almost_equal(result, [-99.0])
