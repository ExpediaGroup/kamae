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

import pandas as pd
import pytest
import tensorflow as tf

from kamae.sklearn.params import SingleInputSingleOutputMixin
from kamae.sklearn.transformers import BaseTransformer


@pytest.fixture
def example_dataframe():
    example_df = pd.DataFrame(
        {
            "col1": [1, 4, 7],
            "col2": [2, 2, 8],
            "col3": [3, 6, 3],
            "col4": ["a", "b", "a"],
            "col5": ["c", "c", "a"],
            "col1_col2_col3": [[1, 2, 3], [4, 2, 6], [7, 8, 3]],
        },
    )
    return example_df


@pytest.fixture
def example_dataframe_with_nulls():
    example_df = pd.DataFrame(
        {
            "col1": [None, 4, 7, 7],
            "col2": [2, None, 2, 8],
            "col3": [3, 6, None, None],
            "col4": ["a", "b", None, "a"],
            "col5": ["c", None, "a", "a"],
            "col1_col2_col3": [[None, 2, 3], [4, None, 6], [7, 8, None], [7, 8, None]],
        },
    )
    return example_df


@pytest.fixture
def layer_name():
    return "test_layer"


@pytest.fixture
def input_col():
    return "test_input"


@pytest.fixture
def output_col():
    return "test_output"


@pytest.fixture
def tf_layer():
    return tf.keras.layers.Dense(1)


@pytest.fixture
def base_transformer(layer_name, output_col, input_col, tf_layer):
    class TestTransformer(
        BaseTransformer,
        SingleInputSingleOutputMixin,
    ):
        """Test transformer for testing abstract base class LayerTransformer"""

        def __init__(self, input_col, output_col, layer_name):
            super().__init__(
                input_col=input_col, output_col=output_col, layer_name=layer_name
            )
            self.input_col = input_col
            self.output_col = output_col
            self.layer_name = layer_name

        def fit(self, X: pd.DataFrame, y=None, **kwargs):
            return self

        def transform(self, X: pd.DataFrame, y=None, **kwargs):
            return X

        def get_tf_layer(self) -> tf.keras.layers.Layer:
            return tf_layer

    return TestTransformer(
        input_col=input_col, output_col=output_col, layer_name=layer_name
    )
