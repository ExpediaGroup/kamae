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

import keras
import tensorflow as tf
from pyspark.ml import Transformer
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType

from kamae.params import ParamSpec
from kamae.spark.common import SparkOperation
from kamae.spark.common.init_builder import build_init, collect_io_params
from kamae.spark.params.param_spec import (
    build_keras_layer_from_specs,
    install_param_specs,
)

if TYPE_CHECKING:
    from pyspark.ml._typing import ParamMap


class BaseTransformer(Transformer, SparkOperation):
    """
    Abstract class for all transformers.

    Subclasses may define the following class attributes for declarative
    configuration:

    ``_params``
        Dict mapping param names to :class:`ParamSpec` instances.  The base
        class will auto-generate ``Param`` objects, getters, setters, and a
        ``@keyword_only __init__`` from this dict.

    ``_compatible_dtypes``
        List of ``DataType`` instances (or ``None`` for any-type).  Replaces
        the need to override the ``compatible_dtypes`` property.

    ``_keras_layer_class``
        A Keras layer class.  When set, ``get_keras_layer()`` is auto-
        generated using the param values and standard name/dtype kwargs.
    """

    _params: Dict[str, ParamSpec] = {}
    _compatible_dtypes: Optional[List[DataType]] = None
    _keras_layer_class: Optional[type] = None

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        return self._compatible_dtypes

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if cls is BaseTransformer:
            return

        custom_specs = cls.__dict__.get("_params", {})

        if custom_specs:
            install_param_specs(cls, custom_specs)

        if "__init__" not in cls.__dict__:
            io_names = collect_io_params(cls)
            cls.__init__ = build_init(io_names, custom_specs, cls)

        keras_cls = cls.__dict__.get("_keras_layer_class")
        if keras_cls is not None and "get_keras_layer" not in cls.__dict__:
            layer_class = keras_cls
            specs = custom_specs

            def get_keras_layer(self):
                return build_keras_layer_from_specs(self, layer_class, specs)

            cls.get_keras_layer = get_keras_layer

    def __init__(self) -> None:
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
    def get_keras_layer(
        self,
    ) -> Union[keras.layers.Layer, List[keras.layers.Layer]]:
        """
        Gets the Keras layer to be used in the model.
        This is the only abstract method that must be implemented.
        :returns: Keras Layer
        """
        raise NotImplementedError

    def construct_layer_info(self) -> Dict[str, Any]:
        """
        Constructs the layer info dictionary.
        Contains the layer name, the Keras layer, and the inputs and outputs.
        This is used when constructing the pipeline graph.

        :returns: Dictionary containing layer information such as
        name, Keras layer, inputs, and outputs.
        """
        inputs, outputs = self.get_layer_inputs_outputs()
        return {
            "name": self.getOrDefault("layerName"),
            "layer": self.get_keras_layer(),
            "inputs": inputs,
            "outputs": outputs,
        }
