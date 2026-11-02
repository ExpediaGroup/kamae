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

from typing import Any, Dict, List, Tuple

import pyspark.sql.functions as F
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import ArrayType

from kamae.spark.transformers import BaseTransformer
from kamae.spark.utils import labels_from_distinct_values


class LabelsAggregationFitMixin:
    """
    Fit by aggregation for the label-collecting estimators (StringIndex,
    OneHotEncode and their shared variants), used only when they are fitted fused
    with other estimators (see `KamaeSparkPipeline` `fuseEstimatorFits`): the
    distinct labels of each input column are collected with `F.collect_set`, then
    ordered exactly as `collect_labels_array` orders them. The estimator implements
    `_transformer_from_labels`. Their normal fit keeps `collect_labels_array`, whose
    distinct and sort are distributed.

    Only for an alphabetical string order, single-value input columns and no
    maxNumLabels: frequency orders need label counts, array inputs are flattened
    first, and `F.collect_set` gathers every distinct label in one aggregation
    buffer before maxNumLabels would cut them, so those keep the estimator's own fit
    (typically high-cardinality columns are the ones limited with maxNumLabels).
    """

    def _label_input_cols(self) -> List[str]:
        """
        The input column(s) the labels are collected from.

        :returns: List of input column names.
        """
        if self.hasParam("inputCols") and self.isDefined("inputCols"):
            return self.getInputCols()
        return [self.getInputCol()]

    def can_fit_by_aggregation(self, dataset: DataFrame) -> bool:
        """
        Whether the labels can be collected with `F.collect_set`.

        :param dataset: The input dataset.
        :returns: True for an alphabetical string order, no array input column and
        no maxNumLabels.
        """
        return (
            self.getMaxNumLabels() is None
            and self.getStringOrderType()
            in [
                "alphabeticalAsc",
                "alphabeticalDesc",
            ]
            and not any(
                isinstance(self.get_column_datatype(dataset, column), ArrayType)
                for column in self._label_input_cols()
            )
        )

    def fit_aggregations(
        self, dataset: DataFrame, prefix: str
    ) -> Tuple[Dict[str, Column], List[Column]]:
        """
        Collects the distinct string values of each input column.

        :param dataset: Pyspark dataframe to fit the estimator to.
        :param prefix: Prefix of the aggregation aliases.
        :returns: Tuple of no helper columns and one aggregation per input column.
        """
        return {}, [
            F.collect_set(F.col(column).cast("string")).alias(f"{prefix}labels_{i}")
            for i, column in enumerate(self._label_input_cols())
        ]

    def fit_from_row(self, row: Dict[str, Any], prefix: str) -> BaseTransformer:
        """
        Orders the collected values into the labels and builds the transformer.

        :param row: The aggregation result row, as a dict of alias -> value.
        :param prefix: The prefix given to `fit_aggregations`.
        :returns: The fitted transformer.
        """
        values = set()
        for i in range(len(self._label_input_cols())):
            values.update(row[f"{prefix}labels_{i}"])
        labels = labels_from_distinct_values(
            values=list(values),
            string_order_type=self.getStringOrderType(),
            mask_token=self.getMaskToken(),
            max_num_labels=self.getMaxNumLabels(),
        )
        return self._transformer_from_labels(labels)
