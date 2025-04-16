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

import pandas as pd
import tensorflow as tf

from kamae.sklearn.params import SingleInputMultiOutputMixin
from kamae.tensorflow.layers import ArraySplitLayer

from .base import BaseTransformer


class ArraySplitTransformer(
    BaseTransformer,
    SingleInputMultiOutputMixin,
):
    """
    VectorSlicer Scikit-Learn Transformer for use in Scikit-Learn pipelines.
    This transformer slices an array column into multiple columns.
    """

    def __init__(self, input_col: str, output_cols: List[str], layer_name: str) -> None:
        super().__init__()
        self.input_col = input_col
        self.output_cols = output_cols
        self.layer_name = layer_name

    def fit(self, X: pd.DataFrame, y=None) -> "ArraySplitTransformer":
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
        Transforms the input dataset. Creates a new column for each output column equal
        to the value of the input column at the given index.

        :param X: Pandas dataframe to transform.
        :param y: Not used, present here for API consistency by convention.
        :returns: Transformed data.
        """
        X[self.output_cols] = pd.DataFrame(X[self.input_col].tolist(), index=X.index)
        return X

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for that unstacks the input tensor and reshapes
        to the original shape.

        :returns: Tensorflow keras layer with name equal to the layerName parameter
        that slices the input tensors.
        """
        return ArraySplitLayer(name=self.layer_name, axis=-1)
