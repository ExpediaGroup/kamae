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
from pyspark.sql.types import DataType, StringType

from kamae.spark.params import SingleInputSingleOutputParams, StringIndexParams
from kamae.spark.transformers import StringIndexTransformer
from kamae.spark.utils import collect_labels_array

from .base import BaseEstimator


class StringIndexEstimator(
    BaseEstimator,
    SingleInputSingleOutputParams,
    StringIndexParams,
):
    """
    String indexer Spark Estimator for use in Spark pipelines.
    This estimator is used to collect all the string labels for a given column.
    When fit is called it returns a StringIndexerLayerModel which can be used
    to index additional feature columns using the same string labels.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        stringOrderType: Optional[str] = None,
        maskToken: Optional[str] = None,
        numOOVIndices: int = 1,
        maxNumLabels: Optional[int] = None,
    ) -> None:
        """
        Initializes the StringIndexEstimator estimator.
        Sets all parameters to given inputs.

        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param: stringOrderType: How to order the string indices.
        Options are 'frequencyAsc', 'frequencyDesc', 'alphabeticalAsc',
        'alphabeticalDesc'.
        :param maskToken: Token to use for masking.
        If set, the token will be indexed as 0.
        :param numOOVIndices: Number of out of vocabulary indices to use.
        :param maxNumLabels: Optional value to limit the size of the vocabulary.
        Default is 1.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(
            stringOrderType="frequencyDesc",
            numOOVIndices=1,
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
        return [StringType()]

    def _fit(self, dataset: DataFrame) -> "StringIndexTransformer":
        """
        Fits the StringIndexEstimator estimator to the given dataset.
        Returns a StringIndexerLayerModel which can be used to index columns using
        the collected string labels.

        :param dataset: Pyspark dataframe to fit the estimator to.
        :returns: StringIndexerLayerModel instance with collected string labels.
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

        return StringIndexTransformer(
            inputCol=self.getInputCol(),
            outputCol=self.getOutputCol(),
            layerName=self.getLayerName(),
            inputDtype=self.getInputDtype(),
            outputDtype=self.getOutputDtype(),
            labelsArray=self.getLabelsArray(),
            stringOrderType=self.getStringOrderType(),
            numOOVIndices=self.getNumOOVIndices(),
            maskToken=self.getMaskToken(),
        )
