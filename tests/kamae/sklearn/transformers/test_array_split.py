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

from kamae.sklearn.transformers import ArraySplitTransformer


class TestArraySplit:
    @pytest.fixture(scope="class")
    def array_split_col1_col2_col3_expected(self):
        return pd.DataFrame(
            {
                "col1": [1, 4, 7],
                "col2": [2, 2, 8],
                "col3": [3, 6, 3],
                "col4": ["a", "b", "a"],
                "col5": ["c", "c", "a"],
                "col1_col2_col3": [[1, 2, 3], [4, 2, 6], [7, 8, 3]],
                "slice_col1": [1, 4, 7],
                "slice_col2": [2, 2, 8],
                "slice_col3": [3, 6, 3],
            },
        )

    @pytest.mark.parametrize(
        "input_col, output_cols, expected_dataframe",
        [
            (
                "col1_col2_col3",
                ["slice_col1", "slice_col2", "slice_col3"],
                "array_split_col1_col2_col3_expected",
            ),
        ],
    )
    def test_sklearn_array_split(
        self,
        example_dataframe,
        input_col,
        output_cols,
        expected_dataframe,
        request,
    ):
        # given
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = ArraySplitTransformer(
            input_col=input_col,
            output_cols=output_cols,
            layer_name="array_split",
        )
        actual = transformer.transform(example_dataframe)
        # then
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "input_tensor",
        [
            tf.constant(
                [
                    [1.0, 6.0, 11.0],
                    [2.0, 7.0, 12.0],
                    [3.0, 8.0, 13.0],
                    [4.0, 9.0, 14.0],
                    [5.0, 10.0, 15.0],
                ]
            ),
            tf.constant(
                [
                    [6.7, 4.7, 2.7, 45.7, 6.9],
                    [2.3, 5.3, 67.3, 3.3, 23.3],
                    [3.7, 3.7, 3.7, 3.7, 3.7],
                    [4.1, 6.1, 8.1, 8.1, 10.111],
                    [5.0111, 8.0111, 9.0111, 10.0111, 15.0111],
                ]
            ),
            tf.constant(
                [
                    [1.1, 6.05],
                    [2.0, 7.0],
                    [3.0, 8.0],
                    [4.0, 9.0],
                    [5.0, 10.0],
                    [7.90, 4567.0],
                    [345.890, 1000.0],
                ]
            ),
        ],
    )
    def test_array_split_sklearn_tf_parity(self, input_tensor):
        col_names = [f"output{i}" for i in range(input_tensor.shape[1])]
        # given
        transformer = ArraySplitTransformer(
            input_col="input",
            output_cols=col_names,
            layer_name="array_split",
        )
        # when
        pd_df = pd.DataFrame(
            {
                "input": input_tensor.numpy().tolist(),
            }
        )
        pd_values = [transformer.transform(pd_df)[c].values.tolist() for c in col_names]
        tensorflow_values = [
            x.numpy().tolist() for x in transformer.get_tf_layer()(input_tensor)
        ]

        # then
        np.testing.assert_almost_equal(
            np.array(pd_values).flatten(),
            np.array(tensorflow_values).flatten(),
            decimal=6,
            err_msg="Scikit-Learn and Tensorflow transform outputs are not equal",
        )
