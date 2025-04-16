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
import pandas as pd
import pytest
import tensorflow as tf

from kamae.sklearn.transformers import ArrayConcatenateTransformer


class TestArrayConcatenate:
    @pytest.fixture(scope="class")
    def array_concatenate_col1_col2_expected(self):
        return pd.DataFrame(
            {
                "col1": [1, 4, 7],
                "col2": [2, 2, 8],
                "col3": [3, 6, 3],
                "col4": ["a", "b", "a"],
                "col5": ["c", "c", "a"],
                "col1_col2_col3": [[1, 2, 3], [4, 2, 6], [7, 8, 3]],
                "vec_col1_col2": [[1, 2], [4, 2], [7, 8]],
            },
        )

    @pytest.fixture(scope="class")
    def array_concatenate_col1_col2_col3_expected(self):
        return pd.DataFrame(
            {
                "col1": [1, 4, 7],
                "col2": [2, 2, 8],
                "col3": [3, 6, 3],
                "col4": ["a", "b", "a"],
                "col5": ["c", "c", "a"],
                "col1_col2_col3": [[1, 2, 3], [4, 2, 6], [7, 8, 3]],
                "vec_col1_col2_col3": [[1, 2, 3], [4, 2, 6], [7, 8, 3]],
            },
        )

    @pytest.fixture(scope="class")
    def array_concatenate_col4_col5_expected(self):
        return pd.DataFrame(
            {
                "col1": [1, 4, 7],
                "col2": [2, 2, 8],
                "col3": [3, 6, 3],
                "col4": ["a", "b", "a"],
                "col5": ["c", "c", "a"],
                "col1_col2_col3": [[1, 2, 3], [4, 2, 6], [7, 8, 3]],
                "vec_col4_col5": [["a", "c"], ["b", "c"], ["a", "a"]],
            },
        )

    @pytest.mark.parametrize(
        "input_cols, output_col, expected_dataframe",
        [
            (["col1", "col2"], "vec_col1_col2", "array_concatenate_col1_col2_expected"),
            (
                ["col1", "col2", "col3"],
                "vec_col1_col2_col3",
                "array_concatenate_col1_col2_col3_expected",
            ),
            (["col4", "col5"], "vec_col4_col5", "array_concatenate_col4_col5_expected"),
        ],
    )
    def test_sklearn_array_concatenate(
        self,
        example_dataframe,
        input_cols,
        output_col,
        expected_dataframe,
        request,
    ):
        # given
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = ArrayConcatenateTransformer(
            input_cols=input_cols,
            output_col=output_col,
            layer_name="array_concatenate",
        )
        actual = transformer.transform(example_dataframe)
        # then
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "input_tensors",
        [
            [
                tf.constant([[1.1], [2.0], [3.0], [4.0], [5.0]]),
                tf.constant([[6.05], [7.0], [8.0], [9.0], [10.0]]),
                tf.constant([[11.01], [12.0], [13.0], [14.0], [15.0]]),
            ],
            [
                tf.constant([[6.7], [2.3], [3.7], [4.1], [5.0111]]),
                tf.constant([[4.7], [5.3], [3.7], [6.1], [8.0111]]),
                tf.constant([[2.7], [67.3], [3.7], [8.1], [9.0111]]),
                tf.constant([[45.7], [3.3], [3.7], [8.1], [10.0111]]),
                tf.constant([[6.9], [23.3], [3.7], [10.111], [15.0111]]),
            ],
            [
                tf.constant([[1.1], [2.0], [3.0], [4.0], [5.0], [7.90], [345.890]]),
                tf.constant([[6.05], [7.0], [8.0], [9.0], [10.0], [4567.0], [1000.0]]),
            ],
        ],
    )
    def test_array_concatenate_spark_tf_parity(self, input_tensors):
        col_names = [f"input{i}" for i in range(len(input_tensors))]

        # given
        transformer = ArrayConcatenateTransformer(
            input_cols=col_names,
            output_col="output",
            layer_name="array_concatenate",
        )

        # when
        pd_df = pd.DataFrame(
            {f"input{i}": inp.numpy().tolist() for i, inp in enumerate(input_tensors)}
        )
        pd_values = transformer.transform(pd_df)["output"].values.tolist()
        tensorflow_values = transformer.get_tf_layer()(input_tensors).numpy().tolist()

        # then
        np.testing.assert_almost_equal(
            pd_values,
            tensorflow_values,
            decimal=6,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
