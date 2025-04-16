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
from typing import List, Tuple

import pyspark.sql.functions as F
from pyspark.ml.param import Params
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import DataType


class InputOutputExtractor(Params):
    """
    Mixin class for extracting input & output information from a transformer/estimator.

    Used across all transformers/estimators to facilitate the construction
    of the pipeline graph.
    """

    def _get_single_or_multi_col(self, ingress: bool) -> List[str]:
        """
        Gets the input or output column name(s) of the layer. If the layer has a
        single input/output column, then the name of the column is returned as a
        singleton list. If the layer has multiple input/output columns, then the list of
        column names is returned.
        :param ingress: Whether to get input (True) or output (False) column names.
        :return: List of input or output column names.
        """
        single_prefix = "inputCol" if ingress else "outputCol"
        multi_prefix = f"{single_prefix}s"
        if self.hasParam(multi_prefix) and self.isDefined(multi_prefix):
            return self.getOrDefault(multi_prefix)
        elif self.hasParam(single_prefix) and self.isDefined(single_prefix):
            return [self.getOrDefault(single_prefix)]
        else:
            return []

    def get_layer_inputs_outputs(self) -> Tuple[List[str], List[str]]:
        """
        Gets the input & output information of the layer. Returns a tuple of lists,
        the first containing the input column names and the second containing the
        output column names.

        :returns: Tuple of lists containing the input and output column names.
        """
        inputs = self._get_single_or_multi_col(ingress=True)
        outputs = self._get_single_or_multi_col(ingress=False)
        return inputs, outputs

    @staticmethod
    def get_column_datatype(dataset: DataFrame, column_name: str) -> DataType:
        """
        Gets the datatype of a column in a dataset. First selects the column to ensure
        that if it is a struct type, the datatype of the struct element is returned.

        :param dataset: Dataset to get the column datatype from.
        :param column_name: Name of the column to get the datatype of.
        :return: Datatype of the column.
        """
        return (
            dataset.select(F.col(column_name).alias("input")).schema["input"].dataType
        )

    def get_multiple_input_cols(
        self, constant_param_name: str, input_cols_limit: int = None
    ) -> List[Column]:
        """
        Gets the (possibly multiple) input columns for the transformer.
        If using multiple input columns, we get them, raising an error if the number is
        greater than the limit. If using a single input column, we get the input column
        and the constant_param_name.

        :returns: List of columns
        """
        if self.isDefined("inputCols"):
            # If multiple input columns are defined, we get them, throwing an error
            # if there are more than the limit.
            input_cols = self.getOrDefault("inputCols")
            if input_cols_limit is not None and len(input_cols) > input_cols_limit:
                raise ValueError(
                    f"""
                    Number of input columns ({len(input_cols)})
                    exceeds limit ({input_cols_limit}).
                    """
                )

            return [F.col(c) for c in input_cols]
        elif (
            self.isDefined("inputCol")
            and self.getOrDefault(constant_param_name) is not None
        ):
            # If only one input column is defined, we use it
            # alongside the constant.
            return [
                F.col(self.getOrDefault("inputCol")),
                F.lit(self.getOrDefault(constant_param_name)).alias(
                    self.uid + constant_param_name
                ),
            ]
        else:
            # If neither inputCols nor inputCol & constant_param_name are defined,
            # we raise an error.
            raise ValueError(
                f"""
                Either inputCols or inputCol & {constant_param_name} must be defined.
                """
            )
