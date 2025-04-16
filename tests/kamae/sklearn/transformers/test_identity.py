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

from kamae.sklearn.transformers import IdentityTransformer


class TestIdentity:
    @pytest.fixture(scope="class")
    def identity_transform_col1_expected(self):
        return pd.DataFrame(
            {
                "col1": [1, 4, 7],
                "col2": [2, 2, 8],
                "col3": [3, 6, 3],
                "col4": ["a", "b", "a"],
                "col5": ["c", "c", "a"],
                "col1_col2_col3": [[1, 2, 3], [4, 2, 6], [7, 8, 3]],
                "iden_col1": [1, 4, 7],
            },
        )

    @pytest.fixture(scope="class")
    def identity_transform_col2_expected(self):
        return pd.DataFrame(
            {
                "col1": [1, 4, 7],
                "col2": [2, 2, 8],
                "col3": [3, 6, 3],
                "col4": ["a", "b", "a"],
                "col5": ["c", "c", "a"],
                "col1_col2_col3": [[1, 2, 3], [4, 2, 6], [7, 8, 3]],
                "iden_col2": [2, 2, 8],
            },
        )

    @pytest.fixture(scope="class")
    def identity_transform_col3_expected(self):
        return pd.DataFrame(
            {
                "col1": [1, 4, 7],
                "col2": [2, 2, 8],
                "col3": [3, 6, 3],
                "col4": ["a", "b", "a"],
                "col5": ["c", "c", "a"],
                "col1_col2_col3": [[1, 2, 3], [4, 2, 6], [7, 8, 3]],
                "iden_col3": [3, 6, 3],
            },
        )

    @pytest.fixture(scope="class")
    def identity_transform_col4_expected(self):
        return pd.DataFrame(
            {
                "col1": [1, 4, 7],
                "col2": [2, 2, 8],
                "col3": [3, 6, 3],
                "col4": ["a", "b", "a"],
                "col5": ["c", "c", "a"],
                "col1_col2_col3": [[1, 2, 3], [4, 2, 6], [7, 8, 3]],
                "iden_col4": ["a", "b", "a"],
            },
        )

    @pytest.fixture(scope="class")
    def identity_transform_col5_expected(self):
        return pd.DataFrame(
            {
                "col1": [1, 4, 7],
                "col2": [2, 2, 8],
                "col3": [3, 6, 3],
                "col4": ["a", "b", "a"],
                "col5": ["c", "c", "a"],
                "col1_col2_col3": [[1, 2, 3], [4, 2, 6], [7, 8, 3]],
                "iden_col5": ["c", "c", "a"],
            },
        )

    @pytest.mark.parametrize(
        "input_col, output_col, expected_dataframe",
        [
            ("col1", "iden_col1", "identity_transform_col1_expected"),
            ("col2", "iden_col2", "identity_transform_col2_expected"),
            ("col3", "iden_col3", "identity_transform_col3_expected"),
            ("col4", "iden_col4", "identity_transform_col4_expected"),
            ("col5", "iden_col5", "identity_transform_col5_expected"),
        ],
    )
    def test_sklearn_identity_transform(
        self,
        example_dataframe,
        input_col,
        output_col,
        expected_dataframe,
        request,
    ):
        # given
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = IdentityTransformer(
            input_col=input_col,
            output_col=output_col,
            layer_name="identity_transform",
        )
        actual = transformer.transform(example_dataframe)
        # then
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "input_tensor",
        [
            (tf.constant([1.0, 4.0, 7.0, 8.0])),
            (tf.constant([2.0, 5.0, 1.0])),
            (tf.constant([-1.0, 7.0])),
            (tf.constant([0.0, 6.0, 3.0])),
            (tf.constant([2.0, 5.0, 1.0, 5.0, 2.5])),
        ],
    )
    def test_identity_transform_sklearn_tf_parity(self, input_tensor):
        # given
        transformer = IdentityTransformer(
            input_col="input", output_col="output", layer_name="identity_transform"
        )
        # when
        pd_df = pd.DataFrame(
            {
                "input": input_tensor.numpy().tolist(),
            }
        )
        pd_values = transformer.transform(pd_df)["output"].values.tolist()
        tensorflow_values = transformer.get_tf_layer()(input_tensor).numpy().tolist()

        # then
        np.testing.assert_almost_equal(
            pd_values,
            tensorflow_values,
            decimal=6,
            err_msg="Sckit-Learn and Tensorflow transform outputs are not equal",
        )
