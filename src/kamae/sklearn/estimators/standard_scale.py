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
from sklearn.preprocessing import StandardScaler

from kamae.sklearn.params import SingleInputSingleOutputMixin
from kamae.sklearn.transformers import BaseTransformerMixin
from kamae.tensorflow.layers import StandardScaleLayer


class StandardScaleEstimator(
    StandardScaler,
    BaseTransformerMixin,
    SingleInputSingleOutputMixin,
):
    """
    Standard Scikit-Learn Estimator for use in Scikit-Learn pipelines.
    Wrapper over the existing implementation of the StandardScaler in Scikit-Learn,
    however operates on array columns and returns array columns. This is to align
    with the Spark implementation of the StandardScaler.

    Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample `x` is calculated as:

        z = (x - u) / s

    where `u` is the mean of the training samples
    and `s` is the standard deviation of the training samples
    """

    def __init__(self, input_col: str, output_col: str, layer_name: str) -> None:
        """
        Intializes a StandardScale estimator.

        :param input_col: Input column name.
        :param output_col: Output column name.
        :param layer_name: Name of the layer. Used as the name of the tensorflow layer
        """
        super().__init__(with_mean=True, with_std=True)
        self.input_col = input_col
        self.output_col = output_col
        self.layer_name = layer_name

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> "StandardScaleEstimator":
        """
        Fits the transformer to the data. Since the scikit-learn StandardScaler
        takes scalar values, we need to convert the numpy array to a list of scalars.
        This is to mimic the behavior of the Spark StandardScaler.

        In this, the input to our transformer is an array, and the output is a scaled
        array.

        :param X: Pandas dataframe to fit the transformer to.
        :param y: Not used, present here for API consistency by convention.
        :returns: Fit pipeline.
        """
        # Get array column as a list of scalars
        feature_array = X[self.input_col].tolist()
        super().fit(X=feature_array, y=y, sample_weight=None)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Transforms the data using the transformer. Standardises the array `input_col`,
        creating a new standardised `output_col`.

        :param X: Pandas dataframe to transform.
        :param y: Not used, present here for API consistency by convention.
        :returns: Transformed data.
        """
        # Get array column as a list of scalars
        feature_array = X[self.input_col].tolist()
        # Transform the list of scalars
        transformed_list_of_scalars = super().transform(feature_array)
        # Set the output column to an array of the transformed list of scalars
        X[self.output_col] = list(transformed_list_of_scalars)
        return X

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the standard scaler transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter
         that performs the standardization.
        """
        return StandardScaleLayer(
            name=self.layer_name, mean=self.mean_, variance=self.var_
        )
