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

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import tensorflow as tf
from pyspark.ml import Transformer
from pyspark.sql import DataFrame

from kamae.spark.common import SparkOperation

if TYPE_CHECKING:
    from pyspark.ml._typing import ParamMap


class BaseTransformer(Transformer, SparkOperation):
    """
    Abstract class for all transformers.
    """

    def __init__(self):
        """
        Initializes the transformer.
        """
        super().__init__()

    def transform(
        self, dataset: DataFrame, params: Optional["ParamMap"] = None
    ) -> DataFrame:
        """
        Overrides the transform method of the parent class to add casting of input and
        output columns to the preferred data type.

        :param dataset: Input dataset.
        :param params: Optional additional parameters.
        :returns: Transformed dataset.
        """
        try:
            dataset = self._create_casted_input_output_columns(
                dataset=dataset, ingress=True
            )
            self._check_input_dtypes_compatible(
                dataset, self._get_single_or_multi_col(ingress=True)
            )

            # Set transformer input columns to casted columns
            self.set_input_columns_to_from_casted(
                dataset=dataset,
                suffix=self.tmp_column_suffix,
            )

            # Call the super transform method
            transformed_dataset = super().transform(dataset=dataset, params=params)

            # Reset the transformer input columns from casted columns
            self.set_input_columns_to_from_casted(
                dataset=dataset,
                suffix=self.tmp_column_suffix,
                reverse=True,
            )

            # Drop the temporary casted columns
            transformed_dataset = self.drop_tmp_casted_input_columns(
                transformed_dataset
            )

            transformed_dataset = self._create_casted_input_output_columns(
                dataset=transformed_dataset, ingress=False
            )
            return transformed_dataset
        except Exception as e:
            param_dict = {
                param[0].name: param[1] for param in self.extractParamMap().items()
            }
            raise e.__class__(
                f"Error in transformer: {self.uid} with params: {param_dict}"
            ).with_traceback(e.__traceback__)

    @abstractmethod
    def get_tf_layer(self) -> Union[tf.keras.layers.Layer, List[tf.keras.layers.Layer]]:
        """
        Gets the tensorflow layer to be used in the model.
        This is the only abstract method that must be implemented.
        :returns: Tensorflow Layer
        """
        raise NotImplementedError

    def construct_layer_info(self) -> Dict[str, Any]:
        """
        Constructs the layer info dictionary.
        Contains the layer name, the tensorflow layer, and the inputs and outputs.
        This is used when constructing the pipeline graph.

        :returns: Dictionary containing layer information such as
        name, tensorflow layer, inputs, and outputs.
        """
        inputs, outputs = self.get_layer_inputs_outputs()
        return {
            "name": self.getOrDefault("layerName"),
            "layer": self.get_tf_layer(),
            "inputs": inputs,
            "outputs": outputs,
        }
