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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from pyspark.ml import Estimator
from pyspark.sql import DataFrame

from kamae.spark.common import SparkOperation
from kamae.spark.transformers import BaseTransformer

if TYPE_CHECKING:
    from pyspark.ml._typing import ParamMap


class BaseEstimator(Estimator, SparkOperation):
    def __init__(self):
        """
        Initializes the estimator.
        """
        super().__init__()

    def fit(
        self,
        dataset: DataFrame,
        params: Optional[Union["ParamMap", List["ParamMap"], Tuple["ParamMap"]]] = None,
    ) -> BaseTransformer:
        """
        Overrides the fit method of the parent class to add casting of input columns
        to the preferred data type.

        :param dataset: Input dataset.
        :param params: Optional additional parameters.
        :returns: Fitted transformer companion object.
        """
        try:
            dataset = self._create_casted_input_output_columns(
                dataset=dataset, ingress=True
            )
            self._check_input_dtypes_compatible(
                dataset, self._get_single_or_multi_col(ingress=True)
            )

            # Set estimator input columns to casted columns
            self.set_input_columns_to_from_casted(
                dataset=dataset,
                suffix=self.tmp_column_suffix,
            )

            # Replicate the logic from the existing abstract estimator fit method
            transformer = super().fit(dataset, params)

            # Reset input columns from casted columns
            self.set_input_columns_to_from_casted(
                dataset=dataset,
                suffix=self.tmp_column_suffix,
                reverse=True,
            )

            # Reset input columns for transformer
            transformer.set_input_columns_to_from_casted(
                dataset=dataset,
                suffix=self.tmp_column_suffix,
                reverse=True,
            )

            return transformer

        except Exception as e:
            param_dict = {
                param[0].name: param[1] for param in self.extractParamMap().items()
            }
            raise e.__class__(
                f"Error in estimator: {self.uid} with params: {param_dict}"
            ).with_traceback(e.__traceback__)

    def construct_layer_info(self) -> Dict[str, Any]:
        """
        Constructs the layer info dictionary.
        Contains the layer name, the tensorflow layer, and the inputs and outputs.
        This is used when constructing the pipeline graph.

        layer is set to None because estimators do not have a defined tf computation,
        we will not use this information in the pipeline graph (for estimators).

        :returns: Dictionary containing layer information such as
        name, tensorflow layer, inputs, and outputs.
        """
        inputs, outputs = self.get_layer_inputs_outputs()
        return {
            "name": self.getOrDefault("layerName"),
            # Estimators do not have a defined tf computation, only transformers do.
            "layer": None,
            "inputs": inputs,
            "outputs": outputs,
        }
