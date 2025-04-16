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

from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from kamae.sklearn.params import SingleInputSingleOutputMixin
from kamae.tensorflow.layers import LogLayer

from .base import BaseTransformer


class LogTransformer(BaseTransformer, SingleInputSingleOutputMixin):
    """
    Log Scikit-Learn Transformer for use in Scikit-Learn pipelines.
    This transformer applies a log(alpha + x) transform to the input column.
    """

    def __init__(
        self,
        input_col: str,
        output_col: str,
        layer_name: str,
        alpha: Optional[float] = None,
    ) -> None:
        """
        Intializes a LogTransformLayer transformer. Sets the default values of:

        - alpha: 1

        :param input_col: Input column name.
        :param output_col: Output column name.
        :param layer_name: Name of the layer. Used as the name of the tensorflow layer
        :param alpha: Value to use in log transform: log(alpha + x). Default is 1.
        :returns: None - class instantialized.
        """
        super().__init__()
        self.input_col = input_col
        self.output_col = output_col
        self.layer_name = layer_name
        self.alpha = float(alpha) if alpha is not None else 1.0

    def fit(self, X: pd.DataFrame, y=None) -> "LogTransformer":
        """
        Fits the transformer. Does nothing since this is just a transformer.

        :param X: Pandas dataframe to fit the transformer to.
        :param y: Not used, present here for API consistency by convention.
        :returns: Fit pipeline, in this case the transformer itself.
        """
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Transforms the data using the transformer. Creates a new column with name
        `output_col`, which applies log(alpha + x) transform to the `input_col`.

        :param X: Pandas dataframe to transform.
        :param y: Not used, present here for API consistency by convention.
        :returns: Transformed data.
        """
        X[self.output_col] = np.log(X[self.input_col] + self.alpha)
        return X

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer that performs the log transform.

        :returns: Tensorflow keras layer with name equal to the layerName parameter
        that performs the log(alpha + x) operation.
        """
        alpha = self.alpha
        return LogLayer(name=self.layer_name, alpha=alpha)
