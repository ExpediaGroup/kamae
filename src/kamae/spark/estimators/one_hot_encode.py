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

# pylint: disable=unused-argument
# pylint: disable=invalid-name
# pylint: disable=too-many-ancestors
# pylint: disable=no-member
from typing import List, Optional

import pyspark.sql.functions as F
from pyspark import keyword_only
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType, IntegerType, LongType, ShortType, StringType

from kamae.spark.params import (
    DropUnseenParams,
    SingleInputSingleOutputParams,
    StringIndexParams,
)
from kamae.spark.transformers import OneHotEncodeTransformer
from kamae.spark.utils import collect_labels_array

from .base import BaseEstimator


class OneHotEncodeEstimator(
    BaseEstimator,
    DropUnseenParams,
    SingleInputSingleOutputParams,
    StringIndexParams,
):
    """
    One-hot encoder Spark Estimator for use in Spark pipelines.
    This estimator is used to collect all the string labels for a given column.
    When fit is called it returns a OneHotEncodeTransformer which can be used
    to create one-hot arrays from additional feature columns using the
    same string labels.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        stringOrderType: str = "frequencyDesc",
        maskToken: Optional[str] = None,
        numOOVIndices: int = 1,
        dropUnseen: bool = False,
        maxNumLabels: Optional[int] = None,
    ) -> None:
        """
        Initializes the OneHotEncoder estimator.
        Sets all parameters to given inputs.

        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param inputDtype: Input data type to cast input column(s) to before
        transforming.
        :param outputDtype: Output data type to cast the output column(s) to after
        transforming.
        :param: stringOrderType: How to order the string indices.
        Options are 'frequencyAsc', 'frequencyDesc', 'alphabeticalAsc',
        'alphabeticalDesc'. Defaults to 'frequencyDesc'.
        :param maskToken: Token to use for masking.
        If set, the token will be indexed as 0.
        :param numOOVIndices: Number of out of vocabulary indices to use. The
        out of vocabulary indices are used to represent unseen labels and are
        placed at the beginning of the one-hot encoding. Defaults to 1.
        :param dropUnseen: Whether to drop unseen label indices. If set to True,
        the transformer will not add an extra dimension for unseen labels in the
        one-hot encoding. Defaults to False.
        :param maxNumLabels: Optional value to limit the size of the vocabulary.
        Defaults to None to consider the full list.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(
            stringOrderType="frequencyDesc",
            numOOVIndices=1,
            dropUnseen=False,
            maskToken=None,
            maxNumLabels=None,
        )
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return [ShortType(), IntegerType(), LongType(), StringType()]

    def _fit(self, dataset: DataFrame) -> "OneHotEncodeTransformer":
        """
        Fits the OneHotEncodeEstimator estimator to the given dataset.
        Returns a OneHotEncodeTransformer which can be used to one-hot columns using
        the collected string labels.

        It re-uses the StringIndexEstimator to collect the string labels.

        :param dataset: Pyspark dataframe to fit the estimator to.
        :returns: OneHotEncodeTransformer instance with collected string labels.
        """
        column_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )
        labels = collect_labels_array(
            dataset=dataset,
            column=F.col(self.getInputCol()),
            column_datatype=column_datatype,
            string_order_type=self.getStringOrderType(),
            mask_token=self.getMaskToken(),
            max_num_labels=self.getMaxNumLabels(),
        )

        self.setLabelsArray(labels)

        return OneHotEncodeTransformer(
            inputCol=self.getInputCol(),
            outputCol=self.getOutputCol(),
            layerName=self.getLayerName(),
            inputDtype=self.getInputDtype(),
            outputDtype=self.getOutputDtype(),
            labelsArray=self.getLabelsArray(),
            stringOrderType=self.getStringOrderType(),
            maskToken=self.getMaskToken(),
            numOOVIndices=self.getNumOOVIndices(),
            dropUnseen=self.getDropUnseen(),
        )
