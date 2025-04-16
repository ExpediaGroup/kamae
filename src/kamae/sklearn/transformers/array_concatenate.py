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

from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf

from kamae.sklearn.params import MultiInputSingleOutputMixin
from kamae.tensorflow.layers import ArrayConcatenateLayer

from .base import BaseTransformer


class ArrayConcatenateTransformer(
    BaseTransformer,
    MultiInputSingleOutputMixin,
):
    """
    Vector Assembler Scikit-Learn Transformer for use in Scikit-Learn pipelines.
    This transformer assembles multiple columns into a single array column.
    """

    def __init__(self, input_cols: List[str], output_col: str, layer_name: str) -> None:
        super().__init__()
        self.input_cols = input_cols
        self.output_col = output_col
        self.layer_name = layer_name

    def fit(self, X: pd.DataFrame, y=None) -> "ArrayConcatenateTransformer":
        """
        Fits the transformer to the data. Does nothing since
        this is transformer not an estimator.

        :param X: Pandas dataframe to fit the transformer to.
        :param y: Not used, present here for API consistency by convention.
        :returns: Fit pipeline, in this case the transformer itself.
        """
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Transform the input dataset. Creates a new column named outputCol which is a
        concatenated array of all input columns.

        :param X: Pandas dataframe to transform.
        :param y: Not used, present here for API consistency by convention.
        :returns: Transformed data.
        """

        # Check which columns are arrays, this gives a dict like:
        # {'col1': True, 'col2': False, 'col3': True}
        is_col_an_array_dict = (
            X.head(1)[self.input_cols]
            .applymap(lambda x: pd.api.types.is_list_like(x))
            .to_dict(orient="records")[0]
        )

        new_input_cols = []
        for col_name, col_an_array in is_col_an_array_dict.items():
            if col_an_array:
                # If the column is an array then we need to create a
                # numpy array of arrays
                # TODO: Can we make this more this efficient?
                values = X[col_name].to_numpy()
                new_input_cols.append(np.array([np.array(x) for x in values]))
            else:
                # If the column is not an array then we just need to extend
                # the numpy array to have an extra dimension. This is so we can concat
                # the arrays later.
                values = X[col_name].to_numpy()
                new_input_cols.append(values[:, None])

        # Concatenate the arrays, this creates an N x M array
        # where N is the number of rows, M is the number of features
        concatenated_array = np.concatenate(new_input_cols, axis=-1)
        # Add this to the dataframe, convert the numpy array to a list
        # of 1-D numpy arrays
        X[self.output_col] = list(concatenated_array)

        return X

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer that concatenates the input tensors.

        :returns: Tensorflow keras layer with name equal to the layerName parameter
        that concatenates the input tensors.
        """
        return ArrayConcatenateLayer(name=self.layer_name, axis=-1)
