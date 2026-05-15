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
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType, LongType, ShortType, StringType

from kamae.params.shared_specs import DROP_UNSEEN_PARAMS, STRING_INDEX_PARAMS
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.transformers import OneHotEncodeTransformer
from kamae.spark.utils import collect_labels_array

from .base import BaseEstimator


class OneHotEncodeEstimator(
    BaseEstimator,
    SingleInputSingleOutputParams,
):
    """
    One-hot encoder Spark Estimator for use in Spark pipelines.
    This estimator is used to collect all the string labels for a given column.
    When fit is called it returns a OneHotEncodeTransformer which can be used
    to create one-hot arrays from additional feature columns using the
    same string labels.
    """

    _compatible_dtypes = [ShortType(), IntegerType(), LongType(), StringType()]
    _params = {**STRING_INDEX_PARAMS, **DROP_UNSEEN_PARAMS}

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
