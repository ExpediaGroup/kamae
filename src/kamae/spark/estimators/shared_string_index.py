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
from pyspark.sql.types import StringType

from kamae.params.shared_specs import STRING_INDEX_PARAMS
from kamae.spark.params import MultiInputMultiOutputParams
from kamae.spark.transformers import SharedStringIndexTransformer
from kamae.spark.utils import collect_labels_array_from_multiple_columns

from .base import BaseEstimator


class SharedStringIndexEstimator(
    BaseEstimator,
    MultiInputMultiOutputParams,
):
    """
    Shared vocab String indexer Spark Estimator for use in Spark pipelines.
    This estimator is used to collect all the string labels across multiple columns
    and keeps a shared list of string labels.
    When fit is called it returns a SharedStringIndexerLayerModel which can be used
    to index additional feature columns using the same string labels.
    """

    _compatible_dtypes = [StringType()]
    _params = {**STRING_INDEX_PARAMS}

    def _fit(self, dataset: DataFrame) -> "SharedStringIndexTransformer":
        """
        Fits the SharedStringIndexEstimator estimator to the given dataset.
        Returns a SharedStringIndexerLayerModel which can be used to index columns using
        the collected string labels.

        :param dataset: Pyspark dataframe to fit the estimator to.
        :returns: SharedStringIndexerLayerModel instance with collected string labels.
        """

        column_datatypes = [
            self.get_column_datatype(dataset=dataset, column_name=i)
            for i in self.getInputCols()
        ]
        labels = collect_labels_array_from_multiple_columns(
            dataset=dataset,
            columns=[F.col(i) for i in self.getInputCols()],
            column_datatypes=column_datatypes,
            string_order_type=self.getStringOrderType(),
            mask_token=self.getMaskToken(),
            max_num_labels=self.getMaxNumLabels(),
        )
        self.setLabelsArray(labels)

        return SharedStringIndexTransformer(
            inputCols=self.getInputCols(),
            outputCols=self.getOutputCols(),
            layerName=self.getLayerName(),
            inputDtype=self.getInputDtype(),
            outputDtype=self.getOutputDtype(),
            labelsArray=self.getLabelsArray(),
            stringOrderType=self.getStringOrderType(),
            numOOVIndices=self.getNumOOVIndices(),
            maskToken=self.getMaskToken(),
        )
