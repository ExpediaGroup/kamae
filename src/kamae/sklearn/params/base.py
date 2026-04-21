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

from typing import List

from .name import LayerNameMixin


class SingleInputMixin:
    """
    Mixin class containing set methods for the single input column scenario.
    """

    _input_col: str

    @property
    def input_col(self) -> str:
        """
        Gets the input column name.

        :returns: Input column name.
        """
        return self._input_col

    @input_col.setter
    def input_col(self, value: str) -> None:
        """
        Sets the input column name.

        :param value: String to set the input_col parameter to.
        :returns: None, input_col is set to the given value.
        """
        self._input_col = value


class MultiInputMixin:
    """
    Mixin class containing set methods for the multiple input columns scenario.
    """

    _input_cols: List[str]

    @property
    def input_cols(self) -> List[str]:
        """
        Gets the input column names.

        :returns: List of strings of input column names.
        """
        return self._input_cols

    @input_cols.setter
    def input_cols(self, value: List[str]) -> None:
        """
        Sets the input column names. to the given list of strings.

        :param value: List of strings to set the input_col parameter to.
        :returns: None, input_col is set to the given value.
        """
        self._input_cols = value


class SingleOutputMixin(LayerNameMixin):
    """
    Mixin class containing set methods for the single output column scenario.
    """

    _output_col: str

    @property
    def output_col(self) -> str:
        """
        Gets the output column name.

        :returns: List of strings of output column names.
        """
        return self._output_col

    @output_col.setter
    def output_col(self, value: str) -> None:
        """
        Sets the output column name to the given string value.

        :param value: String to set the output_col parameter to.
        :returns: None, output_col is set to the given value.
        """
        if value is None:
            # Set default output column name
            self._output_col = "output"
        self._output_col = value

    @LayerNameMixin.layer_name.setter
    def layer_name(self, value: str) -> None:
        """
        Sets the layer name to the given string value.

        :param value: String to set the layer_name parameter to.
        :returns: None, layer_name is set to the given value.
        """
        self._layer_name = value if value is not None else self.__repr__()


class MultiOutputMixin(LayerNameMixin):
    """
    Mixin class containing set methods for the multiple output columns scenario.
    """

    _output_cols: List[str]

    @property
    def output_cols(self) -> List[str]:
        """
        Gets the output column names.

        :returns: List of strings of output column names.
        """
        return self._output_cols

    @LayerNameMixin.layer_name.setter
    def layer_name(self, value: str) -> None:
        """
        Sets the layer name to the given string value.

        :param value: String to set the layer_name parameter to.
        :returns: None, layer_name is set to the given value.
        """
        self._layer_name = value if value is not None else self.__repr__()

    @output_cols.setter
    def output_cols(self, value: List[str]) -> None:
        """
        Sets the output column names to the given list of strings.

        :param value: List of strings to set the output_cols parameter to.
        :returns: None, output_cols is set to the given value.
        """
        self._output_cols = value


class SingleInputSingleOutputMixin(SingleInputMixin, SingleOutputMixin):
    """
    Mixin for a layer that takes a single input and returns a single output
    """


class SingleInputMultiOutputMixin(SingleInputMixin, MultiOutputMixin):
    """
    Mixin for a layer that takes a single input and returns multiple outputs
    """


class MultiInputSingleOutputMixin(MultiInputMixin, SingleOutputMixin):
    """
    Mixin for a layer that takes multiple inputs and returns a single output
    """


class MultiInputMultiOutputMixin(MultiInputMixin, MultiOutputMixin):
    """
    Mixin for a layer that takes multiple inputs and returns multiple outputs
    """
