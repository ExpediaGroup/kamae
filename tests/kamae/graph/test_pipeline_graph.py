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

import networkx as nx
import pytest
import tensorflow as tf

from kamae.graph import PipelineGraph


class TestPipelineGraph:
    def test_update_layer_store_with_key(self):
        # given
        pipeline_graph = PipelineGraph(stage_dict={})
        # when
        pipeline_graph.update_layer_store_with_key("key", "value")
        # then
        assert pipeline_graph.layer_store == {
            "key": {"output": "value", "reused": False}
        }

    def test_update_layer_store_with_key_when_reused(self):
        # given
        pipeline_graph = PipelineGraph(stage_dict={})
        pipeline_graph.layer_store = {"key": {"output": "value", "reused": False}}
        # when
        pipeline_graph.update_layer_store_with_key("key", "value")
        # then
        assert pipeline_graph.layer_store == {
            "key": {"output": "value", "reused": True}
        }

    def test_update_layer_store(self):
        # given
        pipeline_graph = PipelineGraph(stage_dict={})
        # when
        pipeline_graph.update_layer_store({"key": "value"})
        # then
        assert pipeline_graph.layer_store == {
            "key": {"output": "value", "reused": False}
        }

    def test_update_layer_store_list(self):
        # given
        pipeline_graph = PipelineGraph(stage_dict={})
        pipeline_graph.layer_store = {"key_0": {"output": "value_0", "reused": False}}
        # when
        pipeline_graph.update_layer_store(
            {"key_0": "value_0", "key_1": "value_1", "key_2": "value_2"}
        )
        # then
        assert pipeline_graph.layer_store == {
            "key_0": {"output": "value_0", "reused": True},
            "key_1": {"output": "value_1", "reused": False},
            "key_2": {"output": "value_2", "reused": False},
        }

    @pytest.mark.parametrize(
        "layer_name, expected",
        [
            ("layer", "layer"),
            ("layer1_0", "layer1_0"),
            ("layer1_1", "layer1_1"),
            ("layer2_3", "layer2_3"),
        ],
    )
    def test_get_layer_output_from_layer_store(self, layer_name, expected):
        # given
        pipeline_graph = PipelineGraph(stage_dict={})
        pipeline_graph.layer_store = {
            "layer": {"output": "layer", "reused": False},
            "layer1_0": {"output": "layer1_0", "reused": False},
            "layer1_1": {"output": "layer1_1", "reused": False},
            "layer2_3": {"output": "layer2_3", "reused": False},
        }
        # when
        output = pipeline_graph.get_layer_output_from_layer_store(layer_name)
        # then
        assert output == expected, f"Expected {expected} but got {output}"

    @pytest.mark.parametrize(
        "stage_dict, expected_edges",
        [
            (
                {
                    "layer1": {
                        "name": "layer1",
                        "layer": None,
                        "inputs": ["input1"],
                        "outputs": ["layer1_output0", "layer1_output1"],
                    },
                    "layer2": {
                        "name": "layer2",
                        "layer": None,
                        "inputs": ["layer1_output0"],
                        "outputs": ["layer2_output0"],
                    },
                },
                [
                    ("input1", "layer1"),
                    ("layer1", "layer1_output0"),
                    ("layer1", "layer1_output1"),
                    ("layer1_output0", "layer2"),
                    ("layer2", "layer2_output0"),
                ],
            ),
            (
                {
                    "layer1": {
                        "name": "layer1",
                        "layer": None,
                        "inputs": ["input1"],
                        "outputs": ["layer1_output0", "layer1_output1"],
                    },
                    "layer2": {
                        "name": "layer2",
                        "layer": None,
                        "inputs": ["layer1_output0", "layer1_output1"],
                        "outputs": ["layer2_output0"],
                    },
                    "layer3": {
                        "name": "layer3",
                        "layer": None,
                        "inputs": ["layer1_output0", "layer2_output0"],
                        "outputs": ["layer3_output0"],
                    },
                    "layer4": {
                        "name": "layer3",
                        "layer": None,
                        "inputs": [
                            "layer1_output1",
                            "layer2_output0",
                            "layer3_output0",
                        ],
                        "outputs": ["layer4_output0", "layer4_output1"],
                    },
                },
                [
                    ("input1", "layer1"),
                    ("layer1", "layer1_output0"),
                    ("layer1", "layer1_output1"),
                    ("layer1_output0", "layer2"),
                    ("layer1_output1", "layer2"),
                    ("layer2", "layer2_output0"),
                    ("layer1_output0", "layer3"),
                    ("layer2_output0", "layer3"),
                    ("layer3", "layer3_output0"),
                    ("layer1_output1", "layer4"),
                    ("layer2_output0", "layer4"),
                    ("layer3_output0", "layer4"),
                    ("layer4", "layer4_output0"),
                    ("layer4", "layer4_output1"),
                ],
            ),
        ],
    )
    def test_add_stage_edges(self, stage_dict, expected_edges):
        # given
        pipeline_graph = PipelineGraph(stage_dict=stage_dict)
        # when
        graph = pipeline_graph.add_stage_edges(nx.DiGraph())
        # then
        actual = set(graph.edges())
        expected = set(expected_edges)
        assert actual == expected, f"Expected {expected} but got {actual}"

    @pytest.mark.parametrize(
        "layer_store, inputs, expected_outputs",
        [
            (
                {
                    "layer_1": {"output": tf.constant([1.0, 2.0]), "reused": True},
                    "layer_2": {"output": tf.constant([3.0, 4.0]), "reused": False},
                    "layer_3": {"output": tf.constant([5.0, 6.0]), "reused": False},
                },
                {
                    "layer_2": None,
                },
                {"layer_3": tf.constant([5.0, 6.0])},
            ),
            (
                {
                    "layer_1": {"output": tf.constant([1.0, 2.0]), "reused": False},
                    "layer_2": {"output": tf.constant([3.0, 4.0]), "reused": True},
                    "layer_3": {"output": tf.constant([5.0, 6.0]), "reused": False},
                    "layer_4": {"output": tf.constant([7.0, 8.0]), "reused": False},
                },
                {
                    "layer_1": None,
                    "layer_2": None,
                },
                {
                    "layer_3": tf.constant([5.0, 6.0]),
                    "layer_4": tf.constant([7.0, 8.0]),
                },
            ),
        ],
    )
    def test_get_model_outputs(self, layer_store, inputs, expected_outputs):
        # given
        pipeline_graph = PipelineGraph(stage_dict={})
        pipeline_graph.layer_store = layer_store
        pipeline_graph.inputs = inputs
        # when
        outputs = pipeline_graph.get_model_outputs()
        # then
        for key, value in outputs.items():
            tf.debugging.assert_equal(value, expected_outputs[key])

    @pytest.mark.parametrize(
        "layer_name, stage_dict, input_dict, expected_outputs",
        [
            (
                "layer_1",
                {
                    "layer_1": {
                        "name": "layer_1",
                        "layer": None,
                        "inputs": ["input_1", "input_2", "input_3"],
                        "outputs": ["output_1", "output_2"],
                    },
                    "layer_2": {
                        "name": "layer_2",
                        "layer": None,
                        "inputs": ["input_2", "input_4"],
                        "outputs": ["output_3"],
                    },
                },
                {
                    "input_2": "input_tensor_2",
                    "input_3": "input_tensor_3",
                    "input_1": "input_tensor_1",
                },
                [
                    "input_tensor_1",
                    "input_tensor_2",
                    "input_tensor_3",
                ],
            ),
            (
                "layer_2",
                {
                    "layer_1": {
                        "name": "layer_1",
                        "layer": None,
                        "inputs": ["input_1", "input_2", "input_3"],
                        "outputs": ["output_1", "output_2"],
                    },
                    "layer_2": {
                        "name": "layer_2",
                        "layer": None,
                        "inputs": ["input_2", "input_4"],
                        "outputs": ["output_3"],
                    },
                },
                {
                    "input_4": "input_tensor_4",
                    "input_2": "input_tensor_2",
                },
                [
                    "input_tensor_2",
                    "input_tensor_4",
                ],
            ),
        ],
    )
    def test_sort_inputs(self, layer_name, stage_dict, input_dict, expected_outputs):
        # given
        pipeline_graph = PipelineGraph(stage_dict=stage_dict)
        # when
        outputs = pipeline_graph.sort_inputs(
            layer_name=layer_name, input_dict=input_dict
        )
        # then
        assert outputs == expected_outputs

    @pytest.mark.parametrize(
        "tf_input_schema, expected_inputs, expected_layer_store",
        [
            (
                [
                    {
                        "name": "input_1",
                        "shape": (None, 4),
                        "dtype": tf.float32,
                    },
                ],
                {"input_1": tf.keras.layers.Input(shape=(None, 4), dtype=tf.float32)},
                {
                    "input_1": {
                        "output": tf.keras.layers.Input(
                            shape=(None, 4), dtype=tf.float32
                        ),
                        "reused": False,
                    },
                },
            )
        ],
    )
    def test_build_keras_inputs(
        self,
        tf_input_schema,
        expected_inputs,
        expected_layer_store,
    ):
        # given
        pipeline_graph = PipelineGraph(stage_dict={})
        # when
        pipeline_graph.build_keras_inputs(
            tf_input_schema=tf_input_schema,
        )
        # then
        for key, value in pipeline_graph.inputs.items():
            assert key in expected_inputs
            assert value.shape == expected_inputs[key].shape
            assert value.dtype == expected_inputs[key].dtype

        for key, value in pipeline_graph.layer_store.items():
            assert key in expected_layer_store
            assert value["output"].shape == expected_layer_store[key]["output"].shape
            assert value["output"].dtype == expected_layer_store[key]["output"].dtype
            assert value["reused"] == expected_layer_store[key]["reused"]

    @pytest.mark.parametrize(
        "node, in_edges, layer_store, stage_dict, inputs, expected",
        [
            (
                "layer_1",
                [
                    ("input_1", "layer_1"),
                    ("input_2", "layer_1"),
                ],
                {
                    "input_1": {"output": "input_1_output", "reused": False},
                    "input_2": {"output": "input_2_output", "reused": False},
                },
                {
                    "layer_1": {
                        "name": "layer_1",
                        "layer": None,
                        "inputs": ["input_1", "input_2"],
                        "outputs": ["output_1"],
                    },
                },
                {
                    "input_1": "input_1_output",
                    "input_2": "input_2_output",
                },
                [
                    "input_1_output",
                    "input_2_output",
                ],
            ),
            (
                "layer_1",
                [
                    ("input_1", "layer_1"),
                    ("input_2", "layer_1"),
                    ("layer_2_output", "layer_1"),
                    ("layer_3_output", "layer_1"),
                ],
                {
                    "input_1": {"output": "input_1_output", "reused": False},
                    "input_2": {"output": "input_2_output", "reused": False},
                    "layer_2_output": {"output": "layer_2_output", "reused": False},
                    "layer_3_output": {"output": "layer_3_output", "reused": False},
                },
                {
                    "layer_1": {
                        "name": "layer_1",
                        "layer": None,
                        "inputs": [
                            "input_1",
                            "input_2",
                            "layer_2_output",
                            "layer_3_output",
                        ],
                        "outputs": ["layer_1_output"],
                    },
                    "layer_2": {
                        "name": "layer_2",
                        "layer": None,
                        "inputs": ["input_1"],
                        "outputs": ["layer_2_output"],
                    },
                    "layer_3": {
                        "name": "layer_3",
                        "layer": None,
                        "inputs": ["input_2"],
                        "outputs": ["layer_3_output"],
                    },
                },
                {"input_1": "input_1_output", "input_2": "input_2_output"},
                [
                    "input_1_output",
                    "input_2_output",
                    "layer_2_output",
                    "layer_3_output",
                ],
            ),
        ],
    )
    def test_build_transform_layer_inputs(
        self, node, in_edges, layer_store, stage_dict, inputs, expected
    ):
        # given
        pipeline_graph = PipelineGraph(stage_dict=stage_dict)
        pipeline_graph.layer_store = layer_store
        pipeline_graph.inputs = inputs
        # when
        actual = pipeline_graph.build_transform_layer_inputs(
            node=node, in_edges=in_edges
        )
        # then
        assert set(actual) == set(expected)
