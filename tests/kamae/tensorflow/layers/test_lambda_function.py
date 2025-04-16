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

from kamae.tensorflow.layers import LambdaFunctionLayer


class TestLambdaFunction:
    @pytest.fixture(scope="class")
    def single_input_single_output_tf_func_1(self):
        def my_tf_fn(x):
            return tf.math.multiply(x, 2.5) + tf.math.cosh(x)

        return my_tf_fn

    @pytest.fixture(scope="class")
    def single_input_single_output_tf_func_2(self):
        def my_tf_fn(x):
            return tf.concat(
                [tf.math.cumprod(x, axis=-1), tf.math.cumsum(x, axis=-1)], axis=-1
            )

        return my_tf_fn

    @pytest.fixture(scope="class")
    def single_input_single_output_tf_func_3(self):
        def my_tf_fn(x):
            return tf.strings.join(
                [x, "with length =", tf.strings.as_string(tf.strings.length(x))],
                separator=" ",
            )

        return my_tf_fn

    @pytest.fixture(scope="class")
    def multi_input_single_output_tf_func_1(self):
        def my_tf_fn(x):
            x0 = x[0]
            x1 = x[1]
            return tf.math.subtract(x0, x1)

        return my_tf_fn

    @pytest.fixture(scope="class")
    def multi_input_single_output_tf_func_2(self):
        def my_tf_fn(x):
            x0 = x[0]
            x1 = x[1]
            return tf.reduce_max(x0, axis=-1) - tf.reduce_min(x1, axis=-1)

        return my_tf_fn

    @pytest.fixture(scope="class")
    def multi_input_single_output_tf_func_3(self):
        def my_tf_fn(x):
            x0 = x[0]
            x1 = x[1]
            x2 = x[2]
            return tf.strings.join(
                [tf.strings.regex_replace(x0, "hello", "goodbye"), x1, x2],
                separator=" ",
            )

        return my_tf_fn

    @pytest.fixture(scope="class")
    def single_input_multi_output_tf_func_1(self):
        def my_tf_fn(x):
            return [tf.math.square(x), tf.math.log(x), tf.math.subtract(x, 1.0)]

        return my_tf_fn

    @pytest.fixture(scope="class")
    def multi_input_multi_output_tf_func_1(self):
        def my_tf_fn(x):
            x0 = x[0]
            x1 = x[1]
            x2 = x[2]
            return [
                tf.math.square(x0),
                tf.concat([tf.math.square(x0), tf.math.log(x1)], axis=-1),
                tf.math.log(x2),
            ]

        return my_tf_fn

    @pytest.mark.parametrize(
        "inputs, input_name, function, input_dtype, output_dtype, expected_output",
        [
            (
                tf.constant([1.0, 2.0, 3.0]),
                "input_1",
                "single_input_single_output_tf_func_1",
                "float32",
                None,
                tf.constant([4.043081, 8.762196, 17.567661], dtype="float32"),
            ),
            (
                tf.constant([[[1.5, 2.5, 30.0], [1.67, 6.5, -3.08]]]),
                "input_2",
                "single_input_single_output_tf_func_2",
                "float64",
                "float32",
                tf.constant(
                    [
                        [
                            [1.5, 3.75, 112.5, 1.5, 4.0, 34.0],
                            [1.67, 10.855, -33.4334, 1.67, 8.17, 5.09],
                        ]
                    ],
                    dtype="float32",
                ),
            ),
            (
                tf.constant(
                    [
                        ["I'm a string", "I'm another string", "I'm the last string"],
                        ["I'm a string", "I'm another string", "I'm the last string"],
                    ]
                ),
                "input_3",
                "single_input_single_output_tf_func_3",
                None,
                None,
                tf.constant(
                    [
                        [
                            "I'm a string with length = 12",
                            "I'm another string with length = 18",
                            "I'm the last string with length = 19",
                        ],
                        [
                            "I'm a string with length = 12",
                            "I'm another string with length = 18",
                            "I'm the last string with length = 19",
                        ],
                    ],
                    dtype="string",
                ),
            ),
            (
                [
                    tf.constant([[7.0], [4.0], [3.0]]),
                    tf.constant([[1.0], [2.0], [3.1]]),
                ],
                "input_4",
                "multi_input_single_output_tf_func_1",
                "float64",
                "float32",
                tf.constant([[6.0], [2.0], [-0.1]], dtype="float32"),
            ),
            (
                [
                    tf.constant([[7.0, 4.0, 3.0]]),
                    tf.constant([[0.0, 2.0, 3.1]]),
                ],
                "input_4",
                "multi_input_single_output_tf_func_2",
                "float64",
                "float32",
                tf.constant([7.0], dtype="float32"),
            ),
            (
                [
                    tf.constant(["hello world", "the day is bright", "I am happy"]),
                    tf.constant(["how are you", "and clear", "I guess"]),
                    tf.constant(["- well met!", "but storms be coming", "..."]),
                ],
                "input_5",
                "multi_input_single_output_tf_func_3",
                None,
                "string",
                tf.constant(
                    [
                        "goodbye world how are you - well met!",
                        "the day is bright and clear but storms be coming",
                        "I am happy I guess ...",
                    ],
                    dtype="string",
                ),
            ),
            (
                tf.constant(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                    ]
                ),
                "input_6",
                "single_input_multi_output_tf_func_1",
                None,
                None,
                [
                    tf.constant(
                        [
                            [1.0, 4.0, 9.0],
                            [16.0, 25.0, 36.0],
                            [49.0, 64.0, 81.0],
                        ],
                        dtype="float32",
                    ),
                    tf.constant(
                        [
                            [0.0, 0.6931472, 1.0986123],
                            [1.3862944, 1.609438, 1.7917595],
                            [1.9459102, 2.0794415, 2.1972246],
                        ],
                        dtype="float32",
                    ),
                    tf.constant(
                        [
                            [0.0, 1.0, 2.0],
                            [3.0, 4.0, 5.0],
                            [6.0, 7.0, 8.0],
                        ],
                        dtype="float32",
                    ),
                ],
            ),
        ],
    )
    def test_lambda_function(
        self,
        inputs,
        input_name,
        function,
        input_dtype,
        output_dtype,
        expected_output,
        request,
    ):
        # when
        layer = LambdaFunctionLayer(
            name=input_name,
            function=request.getfixturevalue(function),
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        output_tensors = layer(inputs)

        def test_output(output_tensor, expected_output):
            # then
            assert layer.name == input_name, "Layer name is not set properly"
            assert (
                output_tensor.dtype == expected_output.dtype
            ), "Output tensor dtype is not the same as expected tensor dtype"
            assert (
                output_tensor.shape == expected_output.shape
            ), "Output tensor shape is not the same as expected tensor shape"
            if output_tensor.dtype.name == "string":
                tf.debugging.assert_equal(output_tensor, expected_output)
            else:
                tf.debugging.assert_near(output_tensor, expected_output)

        if isinstance(output_tensors, list):
            for output_tensor, expected_output in zip(output_tensors, expected_output):
                test_output(output_tensor, expected_output)
        else:
            test_output(output_tensors, expected_output)
