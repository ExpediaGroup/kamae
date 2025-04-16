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

from kamae.sklearn.transformers import LogTransformer


class TestLogTransformLayer:
    @pytest.fixture(scope="class")
    def log_transform_col1_alpha_1_expected(self):
        return pd.DataFrame(
            {
                "col1": [1, 4, 7],
                "col2": [2, 2, 8],
                "col3": [3, 6, 3],
                "col4": ["a", "b", "a"],
                "col5": ["c", "c", "a"],
                "col1_col2_col3": [[1, 2, 3], [4, 2, 6], [7, 8, 3]],
                "log_col1": [
                    0.6931471805599453,
                    1.6094379124341003,
                    2.0794415416798357,
                ],
            },
        )

    @pytest.fixture(scope="class")
    def log_transform_col2_alpha_5_expected(self):
        return pd.DataFrame(
            {
                "col1": [1, 4, 7],
                "col2": [2, 2, 8],
                "col3": [3, 6, 3],
                "col4": ["a", "b", "a"],
                "col5": ["c", "c", "a"],
                "col1_col2_col3": [[1, 2, 3], [4, 2, 6], [7, 8, 3]],
                "log_col2": [
                    1.9459101490553132,
                    1.9459101490553132,
                    2.5649493574615367,
                ],
            },
        )

    @pytest.mark.parametrize(
        "input_col, output_col, alpha, expected_dataframe",
        [
            ("col1", "log_col1", 1, "log_transform_col1_alpha_1_expected"),
            ("col2", "log_col2", 5, "log_transform_col2_alpha_5_expected"),
        ],
    )
    def test_sklearn_log_transform(
        self,
        example_dataframe,
        input_col,
        output_col,
        alpha,
        expected_dataframe,
        request,
    ):
        # given
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = LogTransformer(
            input_col=input_col,
            output_col=output_col,
            layer_name="log_transform",
            alpha=alpha,
        )
        actual = transformer.transform(example_dataframe)
        # then
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "input_tensor, alpha",
        [
            (tf.constant([1.0, 4.0, 7.0, 8.0]), 1),
            (tf.constant([2.0, 5.0, 1.0]), 2),
            (tf.constant([-1.0, 7.0]), 3),
            (tf.constant([0.0, 6.0, 3.0]), 4),
            (tf.constant([2.0, 5.0, 1.0, 5.0, 2.5]), 10),
        ],
    )
    def test_log_transform_sklearn_tf_parity(self, input_tensor, alpha):
        # given
        transformer = LogTransformer(
            input_col="input",
            output_col="output",
            alpha=alpha,
            layer_name="log_transform",
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
            err_msg="Scikit-Learn and Tensorflow transform outputs are not equal",
        )
