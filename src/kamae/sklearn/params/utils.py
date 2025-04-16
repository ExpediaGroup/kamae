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

from typing import List, Tuple


class InputOutputExtractor:
    """
    Mixin class containing methods for extracting input and output column names.
    """

    def get_layer_inputs_outputs(self) -> Tuple[List[str], List[str]]:
        """
        Gets the input & output information of the layer. Returns a tuple of lists,
        the first containing the input column names and the second containing the
        output column names.

        :returns: Tuple of lists containing the input and output column names.
        """

        if hasattr(self, "input_cols") and getattr(self, "input_cols") is not None:
            inputs = self.input_cols
        elif hasattr(self, "input_col") and getattr(self, "input_col") is not None:
            inputs = [self.input_col]
        else:
            inputs = []

        if hasattr(self, "output_cols") and getattr(self, "output_cols") is not None:
            outputs = self.output_cols
        elif hasattr(self, "output_col") and getattr(self, "output_col") is not None:
            outputs = [self.output_col]
        else:
            outputs = []

        return inputs, outputs
