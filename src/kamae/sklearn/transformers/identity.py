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
import tensorflow as tf

from kamae.sklearn.params import SingleInputSingleOutputMixin
from kamae.tensorflow.layers import IdentityLayer

from .base import BaseTransformer


class IdentityTransformer(BaseTransformer, SingleInputSingleOutputMixin):
    """
    Identity Scikit-Learn Transformer for use in Scikit-Learn pipelines.
    This transformer simply passes the input to the output unchanged.
    Used for cases where you want to keep the input the same.
    """

    def __init__(self, input_col: str, output_col: str, layer_name: str) -> None:
        """
        Intializes an IdentityTransformer transformer.

        :param input_col: Input column name.
        :param output_col: Output column name.
        :param layer_name: Name of the layer. Used as the name of the tensorflow layer
        in the keras model.
        :returns: None - class instantialized.
        """
        super().__init__()
        self.input_col = input_col
        self.output_col = output_col
        self.layer_name = layer_name

    def fit(self, X: pd.DataFrame, y=None) -> "IdentityTransformer":
        """
        Fits the transformer to the data. Does nothing since
        this is an identity transformer.

        :param X: Pandas dataframe to fit the transformer to.
        :param y: Not used, present here for API consistency by convention.
        :returns: Fit pipeline, in this case the transformer itself.
        """
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Transforms the data using the transformer. Creates a new column with name
        `output_col`, which is the same as the `input_col`.

        :param X: Pandas dataframe to transform.
        :param y: Not used, present here for API consistency by convention.
        :returns: Transformed data.
        """
        X[self.output_col] = X[self.input_col]
        return X

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the identity transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
         performs an Identity operation.
        """
        return IdentityLayer(
            name=self.layer_name,
        )
