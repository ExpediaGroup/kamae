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
import tensorflow as tf
from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import DataType, StringType

from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_scalar_transform
from kamae.tensorflow.layers import StringAffixLayer

from .base import BaseTransformer


class StringAffixParams(Params):
    """
    Mixin class containing parameters needed for string affixing.
    transforms.
    """

    prefix = Param(
        Params._dummy(),
        "prefix",
        "Value to use as a prefix when joining the strings.",
        typeConverter=TypeConverters.toString,
    )
    suffix = Param(
        Params._dummy(),
        "suffix",
        "Value to use as a suffix when joining the strings.",
        typeConverter=TypeConverters.toString,
    )

    def setPrefix(self, value: str) -> "StringAffixParams":
        """
        Sets the prefix parameter.

        :param value: String value to use as a prefix when joining the strings.
        :returns: Instance of class mixed in.
        """
        return self._set(prefix=value)

    def getPrefix(self) -> str:
        """
        Gets the prefix parameter.

        :returns: String value to use as a prefix when joining the strings.
        """
        return self.getOrDefault(self.prefix)

    def setSuffix(self, value: str) -> "StringAffixParams":
        """
        Sets the suffix parameter.

        :param value: String value to use as a suffix when joining the strings.
        :returns: Instance of class mixed in.
        """
        return self._set(suffix=value)

    def getSuffix(self) -> str:
        """
        Gets the suffix parameter.

        :returns: String value to use as a suffix when joining the strings.
        """
        return self.getOrDefault(self.suffix)


class StringAffixTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
    StringAffixParams,
):
    """
    String Affix Spark Transformer for use in Spark pipelines.
    This transformer takes in a column and pre- and su- fixes it.
    Input columns must be of type string.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> None:
        """
        Initializes the string affix transformer.
        :param inputCol: column to combine with prefix or suffix. Must be type string.
        :param outputCol: column to output the affixed string to.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param prefix: String to use as a prefix when joining the strings.
        :param suffix: String to use as a suffix when joining the strings.
        """
        super().__init__()
        self._setDefault(prefix=None, suffix=None)
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

    def _validate_params(self) -> None:
        """
        Validates the parameters passed to the transformer.
        """
        prefix = self.getPrefix()
        suffix = self.getSuffix()
        if (prefix is None or prefix == "") and (suffix is None or suffix == ""):
            raise ValueError(
                "Either prefix or suffix must be set. Otherwise nothing to affix."
            )

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        where the value is origin column combined with prefix and or suffix.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        self._validate_params()

        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )

        def add_prefix_suffix(
            column: Column, prefix: Optional[str] = None, suffix: Optional[str] = None
        ):
            if prefix is not None and prefix != "":
                column = F.concat(F.lit(prefix), column)
            if suffix is not None and suffix != "":
                column = F.concat(column, F.lit(suffix))
            return column

        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: add_prefix_suffix(x, self.getPrefix(), self.getSuffix()),
        )

        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the string affix transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
         performs prefixing and suffixing.
        """
        return StringAffixLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            prefix=self.getPrefix(),
            suffix=self.getSuffix(),
        )
