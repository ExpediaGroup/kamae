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


class TestBaseTransformer:
    def test_construct_layer_info(
        self,
        test_base_transformer,
        layer_name,
        output_col,
        input_col,
        tf_layer,
    ):
        # when
        layer_info = test_base_transformer.construct_layer_info()
        # then
        assert layer_info["name"] == layer_name
        assert layer_info["layer"] == tf_layer
        assert layer_info["inputs"] == [input_col]
        assert layer_info["outputs"] == [output_col]
