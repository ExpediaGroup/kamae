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

from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import (
    HasInputCol,
    HasInputCols,
    HasOutputCol,
    HasOutputCols,
)
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from kamae.utils import DType

from .name import HasLayerName
from .utils import InputOutputExtractor


class HasInputDtype(Params):
    """
    Mixin class for a transformer input datatype.
    """

    inputDtype = Param(
        Params._dummy(),
        "inputDtype",
        "Input datatype of the transformer",
        typeConverter=TypeConverters.toString,
    )

    def setInputDtype(self, value: str) -> "HasInputDtype":
        """
        Sets the parameter inputDtype to the given string value.

        :param value: String to set the inputDtype parameter to.
        :raises ValueError: If the input dtype is not supported.
        :returns: Instance of class mixed in.
        """
        dtype_enums = [dtype.dtype_name for dtype in DType]
        if value not in dtype_enums:
            raise ValueError(
                f"""Input dtype {value} not supported.
                Supported dtypes are: {dtype_enums}"""
            )
        return self._set(inputDtype=value)

    def getInputDtype(self) -> str:
        """
        Gets the value of the inputDtype parameter.
        :returns: Input datatype.
        """
        return self.getOrDefault(self.inputDtype)

    def getInputTFDtype(self) -> Optional[str]:
        """
        Gets the tensorflow datatype string from the inputDtype parameter.
        Uses the DType enum within Kamae to map the inputDtype to the tensorflow
        datatype string.
        :returns: String of the tensorflow datatype.
        """
        input_dtype = self.getInputDtype()
        if input_dtype is None:
            return None
        dtypes_map = {dtype.dtype_name: dtype.tf_dtype.name for dtype in DType}
        return dtypes_map[input_dtype]


class HasOutputDtype(Params):
    """
    Mixin class for a transformer output datatype.
    """

    outputDtype = Param(
        Params._dummy(),
        "outputDtype",
        "Output datatype of the transformer",
        typeConverter=TypeConverters.toString,
    )

    def setOutputDtype(self, value: str) -> "HasOutputDtype":
        """
        Sets the parameter outputDtype to the given string value.

        :param value: String to set the outputDtype parameter to.
        :raises ValueError: If the output dtype is not supported.
        :returns: Instance of class mixed in.
        """
        dtype_enums = [dtype.dtype_name for dtype in DType]
        if value not in dtype_enums:
            raise ValueError(
                f"""Output dtype {value} not supported.
                Supported dtypes are: {dtype_enums}"""
            )
        return self._set(outputDtype=value)

    def getOutputDtype(self) -> str:
        """
        Gets the value of the outputDtype parameter.
        :returns: Output datatype.
        """
        return self.getOrDefault(self.outputDtype)

    def getOutputTFDtype(self) -> Optional[str]:
        """
        Gets the tensorflow datatype string from the outputDtype parameter.
        Uses the DType enum within Kamae to map the outputDtype to the tensorflow
        datatype string.
        :returns: String of the tensorflow datatype.
        """

        output_dtype = self.getOutputDtype()
        if output_dtype is None:
            return None
        dtypes_map = {dtype.dtype_name: dtype.tf_dtype.name for dtype in DType}
        return dtypes_map[output_dtype]


class SingleInputParams(HasInputCol):
    """
    Mixin class containing set methods for the single input column scenario.
    """

    def setInputCol(self, value: str) -> "SingleInputParams":
        """
        Sets the parameter inputCol to the given string value.

        :param value: String to set the inputCol parameter to.
        :returns: Instance of class mixed in.
        """
        return self._set(inputCol=value)


class MultiInputParams(HasInputCols):
    """
    Mixin class containing set methods for the multiple input columns scenario.
    """

    def setInputCols(self, value: List[str]) -> "MultiInputParams":
        """
        Sets the parameter inputCols to the given list of strings.

        :param value: List of strings to set the inputCols parameter to.
        :returns: Instance of class mixed in.
        """
        return self._set(inputCols=value)


class SingleOutputParams(HasLayerName, HasOutputCol, HasOutputDtype):
    """
    Mixin class containing set methods for the single output column scenario.
    """

    def setLayerName(self, value: str) -> "SingleOutputParams":
        """
        Sets the parameter layerName to the given string value.
        Throws an error if the value is the same as the output column name,
        as this causes issues when constructing the pipeline graph.

        :param value: String to set the layerName parameter to.
        :returns: Instance of class mixed in.
        """
        if self.hasParam("outputCol") and self.isDefined("outputCol"):
            if value == self.getOutputCol():
                raise ValueError("Layer name and output column name must be different.")
        return self._set(layerName=value)

    def setOutputCol(self, value: str) -> "SingleOutputParams":
        """
        Sets the parameter outputCol to the given string value.
        Throws an error if the value is the same as the layer name,
        as this causes issues when constructing the pipeline graph.

        :param value: String to set the outputCol parameter to.
        :returns: Instance of class mixed in.
        """
        if self.hasParam("layerName") and self.isDefined("layerName"):
            if value == self.getLayerName():
                raise ValueError("Layer name and output column name must be different.")
        return self._set(outputCol=value)


class MultiOutputParams(HasLayerName, HasOutputCols, HasOutputDtype):
    """
    Mixin class containing set methods for the multiple output columns scenario.
    """

    def setLayerName(self, value: str) -> "MultiOutputParams":
        """
        Sets the parameter layerName to the given string value.
        Throws an error if the value is the same as one of the output column names,
        as this causes issues when constructing the pipeline graph.

        :param value: String to set the layerName parameter to.
        :returns: Instance of class mixed in.
        """
        if self.hasParam("outputCol") and self.isDefined("outputCols"):
            if value in self.getOutputCols():
                raise ValueError(
                    "Layer name and output column names must be different."
                )
        return self._set(layerName=value)

    def setOutputCols(self, value: List[str]) -> "MultiOutputParams":
        """
        Sets the parameter outputCols to the given list of strings.
        Throws an error if one of the output column names is the same as the layer name,
        as this causes issues when constructing the pipeline graph.

        :param value: List of strings to set the outputCols parameter to.
        :returns: Instance of class mixed in.
        """
        if self.hasParam("layerName") and self.isDefined("layerName"):
            if self.getLayerName() in value:
                raise ValueError(
                    "Layer name and output column names must be different."
                )
        return self._set(outputCols=value)


class SingleInputSingleOutputParams(
    SingleInputParams,
    SingleOutputParams,
    InputOutputExtractor,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Mixin class containing set methods for the single input
    and single output column scenario.
    """


class SingleInputMultiOutputParams(
    SingleInputParams,
    MultiOutputParams,
    InputOutputExtractor,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Mixin class containing set methods for the single input
    and multiple output columns scenario.
    """


class MultiInputSingleOutputParams(
    MultiInputParams,
    SingleOutputParams,
    InputOutputExtractor,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Mixin class containing set methods for the multiple input
    and single output column scenario.
    """


class MultiInputMultiOutputParams(
    MultiInputParams,
    MultiOutputParams,
    InputOutputExtractor,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Mixin class containing set methods for the multiple input
    and multiple output columns scenario.
    """
