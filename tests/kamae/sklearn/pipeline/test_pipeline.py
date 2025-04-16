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

import os
from shutil import rmtree

import joblib
import pandas as pd
import pytest
import tensorflow as tf

from kamae.sklearn.estimators import StandardScaleEstimator
from kamae.sklearn.pipeline import KamaeSklearnPipeline
from kamae.sklearn.transformers import (
    ArrayConcatenateTransformer,
    ArraySplitTransformer,
    IdentityTransformer,
    LogTransformer,
)


class TestKamaeSklearnPipeline:
    """
    Tests both the pipeline and the pipeline model (fit and transform)
    """

    @pytest.fixture(scope="class")
    def test_dir(self):
        path = "./tmp_sklearn_test"
        os.makedirs(path, exist_ok=True)
        yield path
        rmtree(path)

    @pytest.fixture(scope="class")
    def valid_stages_transforms_only_0(self):
        return [
            LogTransformer(
                input_col="col1",
                output_col="log_col1",
                alpha=0.1,
                layer_name="log_transform_0",
            ),
            ArrayConcatenateTransformer(
                input_cols=["log_col1", "col2", "col3"],
                output_col="features",
                layer_name="vector_assembler_0",
            ),
            ArraySplitTransformer(
                input_col="features",
                output_cols=["log_col1_sliced", "col2_sliced", "col3_sliced"],
                layer_name="vector_slicer_0",
            ),
        ]

    @pytest.fixture(scope="class")
    def valid_stages_transforms_only_1(self):
        return [
            LogTransformer(
                input_col="col2",
                output_col="log_col2",
                alpha=5,
                layer_name="log_transform_1",
            ),
            IdentityTransformer(
                input_col="col1",
                output_col="col1_identity",
                layer_name="identity_transform_1",
            ),
            ArrayConcatenateTransformer(
                input_cols=["col1_identity", "log_col2", "col3"],
                output_col="features",
                layer_name="vector_assembler_1",
            ),
            ArraySplitTransformer(
                input_col="features",
                output_cols=["col1_sliced", "log_col2_sliced", "col3_sliced"],
                layer_name="vector_slicer_1",
            ),
        ]

    @pytest.fixture(scope="class")
    def valid_stages_0(self):
        return [
            ArrayConcatenateTransformer(
                input_cols=["col1", "col2", "col3"],
                output_col="features",
                layer_name="vector_assembler_0",
            ),
            StandardScaleEstimator(
                input_col="features",
                output_col="features_scaled",
                layer_name="standard_scaler_0",
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_dataframe_stage_0(self):
        return pd.DataFrame(
            {
                "col1": [1, 4, 7],
                "col2": [2, 2, 8],
                "col3": [3, 6, 3],
                "col4": ["a", "b", "a"],
                "col5": ["c", "c", "a"],
                "col1_col2_col3": [[1, 2, 3], [4, 2, 6], [7, 8, 3]],
                "features": [[1, 2, 3], [4, 2, 6], [7, 8, 3]],
                "features_scaled": [
                    [-1.2247448713915892, -0.7071067811865475, -0.7071067811865475],
                    [0.0, -0.7071067811865475, 1.414213562373095],
                    [1.2247448713915892, 1.414213562373095, -0.7071067811865475],
                ],
            }
        )

    @pytest.fixture(scope="class")
    def valid_stages_1(self):
        return [
            LogTransformer(
                input_col="col3",
                output_col="log_col3",
                alpha=0.1,
                layer_name="log_transform_2",
            ),
            ArrayConcatenateTransformer(
                input_cols=["col1_col2_col3", "log_col3"],
                output_col="features",
                layer_name="vector_assembler_2",
            ),
            StandardScaleEstimator(
                input_col="features",
                output_col="features_scaled",
                layer_name="standard_scaler_2",
            ),
            IdentityTransformer(
                input_col="col4",
                output_col="col4_identity",
                layer_name="identity_transform_2",
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_dataframe_stage_1(self):
        return pd.DataFrame(
            {
                "col1": [1, 4, 7],
                "col2": [2, 2, 8],
                "col3": [3, 6, 3],
                "col4": ["a", "b", "a"],
                "col5": ["c", "c", "a"],
                "col1_col2_col3": [[1, 2, 3], [4, 2, 6], [7, 8, 3]],
                "log_col3": [
                    1.1314021114911006,
                    1.8082887711792655,
                    1.1314021114911006,
                ],
                "features": [
                    [1, 2, 3, 1.1314021114911006],
                    [4, 2, 6, 1.8082887711792655],
                    [7, 8, 3, 1.1314021114911006],
                ],
                "features_scaled": [
                    [
                        -1.2247448713915892,
                        -0.7071067811865475,
                        -0.7071067811865475,
                        -0.7071067811865468,
                    ],
                    [0.0, -0.7071067811865475, 1.414213562373095, 1.4142135623730956],
                    [
                        1.2247448713915892,
                        1.414213562373095,
                        -0.7071067811865475,
                        -0.7071067811865468,
                    ],
                ],
                "col4_identity": ["a", "b", "a"],
            }
        )

    @pytest.mark.parametrize(
        "stages",
        [
            "valid_stages_0",
            "valid_stages_1",
        ],
    )
    def test_sklearn_read_write_pipeline(
        self, example_dataframe, test_dir, stages, request
    ):
        stages = request.getfixturevalue(stages)
        pipeline = KamaeSklearnPipeline(steps=[(s.layer_name, s) for s in stages])
        joblib.dump(pipeline, f"{test_dir}/pipeline")
        pipeline_loaded = joblib.load(f"{test_dir}/pipeline")
        pipeline.fit(example_dataframe)
        pipeline_loaded.fit(example_dataframe)
        orig_actual = pipeline.transform(example_dataframe)
        loaded_actual = pipeline_loaded.transform(example_dataframe)
        pd.testing.assert_frame_equal(orig_actual, loaded_actual)

    @pytest.mark.parametrize(
        "stages, expected_dataframe",
        [
            ("valid_stages_0", "expected_dataframe_stage_0"),
            ("valid_stages_1", "expected_dataframe_stage_1"),
        ],
    )
    def test_sklearn_pipeline(
        self, stages, example_dataframe, expected_dataframe, request
    ):
        stages = request.getfixturevalue(stages)
        pipeline = KamaeSklearnPipeline(steps=[(s.layer_name, s) for s in stages])

        pipeline.fit(example_dataframe)

        transformed_df = pipeline.transform(example_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        pd.testing.assert_frame_equal(transformed_df, expected)

    @pytest.mark.parametrize(
        "stages, input_tensors, tf_input_schema, expected_output",
        [
            (
                "valid_stages_0",
                {
                    "col1": tf.constant(
                        [
                            [[1], [4], [7]],
                        ],
                        dtype=tf.float32,
                    ),
                    "col2": tf.constant(
                        [
                            [[2], [2], [8]],
                        ],
                        dtype=tf.float32,
                    ),
                    "col3": tf.constant(
                        [
                            [[3], [6], [3]],
                        ],
                        dtype=tf.float32,
                    ),
                },
                [
                    {
                        "name": "col1",
                        "dtype": "float32",
                        "shape": (None, 1),
                    },
                    {
                        "name": "col2",
                        "dtype": "float32",
                        "shape": (None, 1),
                    },
                    {
                        "name": "col3",
                        "dtype": "float32",
                        "shape": (None, 1),
                    },
                ],
                tf.constant(
                    [
                        [
                            [-1.2247448, -0.70710677, -0.70710677],
                            [0.0, -0.70710677, 1.4142135],
                            [1.2247448, 1.4142135, -0.70710677],
                        ]
                    ]
                ),
            ),
            (
                "valid_stages_1",
                {
                    "col1_col2_col3": tf.constant(
                        [
                            [[1, 2, 3], [4, 2, 6], [7, 8, 3]],
                        ],
                        dtype=tf.float32,
                    ),
                    "col3": tf.constant(
                        [
                            [[3], [6], [3]],
                        ],
                        dtype=tf.float32,
                    ),
                    "col4": tf.constant(
                        [
                            [["a"], ["b"], ["a"]],
                        ],
                        dtype=tf.string,
                    ),
                },
                [
                    {
                        "name": "col1_col2_col3",
                        "dtype": "float32",
                        "shape": (None, 3),
                    },
                    {
                        "name": "col3",
                        "dtype": "float32",
                        "shape": (None, 1),
                    },
                    {
                        "name": "col4",
                        "dtype": "string",
                        "shape": (None, 1),
                    },
                ],
                [
                    tf.constant(
                        [
                            [["a"], ["b"], ["a"]],
                        ],
                        dtype=tf.string,
                    ),
                    tf.constant(
                        [
                            [
                                [-1.2247448, -0.70710677, -0.70710677, -0.7071067],
                                [0.0, -0.70710677, 1.4142135, 1.4142138],
                                [1.2247448, 1.4142135, -0.70710677, -0.7071067],
                            ]
                        ],
                        dtype=tf.float32,
                    ),
                ],
            ),
            (
                "valid_stages_transforms_only_0",
                {
                    "col1": tf.constant(
                        [
                            [[1.0], [4.0], [7.0]],
                        ],
                        dtype=tf.float32,
                    ),
                    "col2": tf.constant(
                        [
                            [[2.0], [2.0], [8.0]],
                        ],
                        dtype=tf.float32,
                    ),
                    "col3": tf.constant(
                        [
                            [[3.0], [6.0], [3.0]],
                        ],
                        dtype=tf.float32,
                    ),
                },
                [
                    {
                        "name": "col1",
                        "dtype": "float32",
                        "shape": (None, 1),
                    },
                    {
                        "name": "col2",
                        "dtype": "float32",
                        "shape": (None, 1),
                    },
                    {
                        "name": "col3",
                        "dtype": "float32",
                        "shape": (None, 1),
                    },
                ],
                [
                    tf.constant(
                        [
                            [[0.0953102], [1.4109869], [1.9600948]],
                        ],
                        dtype=tf.float32,
                    ),
                    tf.constant(
                        [
                            [[2.0], [2.0], [8.0]],
                        ],
                        dtype=tf.float32,
                    ),
                    tf.constant(
                        [
                            [[3.0], [6.0], [3.0]],
                        ],
                        dtype=tf.float32,
                    ),
                ],
            ),
            (
                "valid_stages_transforms_only_1",
                {
                    "col1": tf.constant(
                        [
                            [[1.0], [4.0], [7.0]],
                        ],
                        dtype=tf.float32,
                    ),
                    "col2": tf.constant(
                        [
                            [[2.0], [2.0], [8.0]],
                        ],
                        dtype=tf.float32,
                    ),
                    "col3": tf.constant(
                        [
                            [[3.0], [6.0], [3.0]],
                        ],
                        dtype=tf.float32,
                    ),
                },
                [
                    {
                        "name": "col1",
                        "dtype": "float32",
                        "shape": (None, 1),
                    },
                    {
                        "name": "col2",
                        "dtype": "float32",
                        "shape": (None, 1),
                    },
                    {
                        "name": "col3",
                        "dtype": "float32",
                        "shape": (None, 1),
                    },
                ],
                [
                    tf.constant(
                        [
                            [[1.0], [4.0], [7.0]],
                        ],
                        dtype=tf.float32,
                    ),
                    tf.constant(
                        [
                            [[1.9459101], [1.9459101], [2.5649493]],
                        ],
                        dtype=tf.float32,
                    ),
                    tf.constant(
                        [
                            [[3.0], [6.0], [3.0]],
                        ],
                        dtype=tf.float32,
                    ),
                ],
            ),
        ],
    )
    def test_keras_model(
        self,
        stages,
        input_tensors,
        tf_input_schema,
        expected_output,
        example_dataframe,
        request,
    ):
        stages = request.getfixturevalue(stages)
        pipeline = KamaeSklearnPipeline(
            steps=[(stage.layer_name, stage) for stage in stages]
        )

        pipeline.fit(example_dataframe)

        keras_model = pipeline.build_keras_model(tf_input_schema=tf_input_schema)

        actual = keras_model(input_tensors)

        if isinstance(actual, list):
            for a, e in zip(actual, expected_output):
                if a.dtype == "string":
                    tf.debugging.assert_equal(a, e)
                else:
                    tf.debugging.assert_near(a, e, atol=1e-6)
        elif isinstance(actual, dict):
            for a, e in zip(actual.values(), expected_output):
                if a.dtype == "string":
                    tf.debugging.assert_equal(a, e)
                else:
                    tf.debugging.assert_near(a, e, atol=1e-6)
        else:
            if actual.dtype == "string":
                tf.debugging.assert_equal(actual, expected_output)
            else:
                tf.debugging.assert_near(actual, expected_output, atol=1e-6)
