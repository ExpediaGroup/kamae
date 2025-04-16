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

import pytest
import tensorflow as tf
from pyspark.sql.types import DoubleType

from kamae.spark.estimators import StandardScaleEstimator, StringIndexEstimator
from kamae.spark.pipeline import KamaeSparkPipeline, KamaeSparkPipelineModel
from kamae.spark.transformers import (
    ArrayConcatenateTransformer,
    ArraySplitTransformer,
    BucketizeTransformer,
    HashIndexTransformer,
    IdentityTransformer,
    LogTransformer,
    SubtractTransformer,
)


class TestPipeline:
    """
    Tests both the pipeline and the pipeline model (fit and transform)
    """

    @pytest.fixture
    def test_dir(self):
        path = "./tmp_test"
        os.makedirs(path, exist_ok=True)
        yield path
        rmtree(path)

    @pytest.fixture(scope="class")
    def valid_stages_transforms_only_0(self):
        return [
            LogTransformer(
                inputCol="col1",
                outputCol="log_col1",
                alpha=0.1,
            ),
            ArrayConcatenateTransformer(
                inputCols=["log_col1", "col2", "col3"],
                outputCol="features",
            ),
            ArraySplitTransformer(
                inputCol="features",
                outputCols=["log_col1_sliced", "col2_sliced", "col3_sliced"],
            ),
        ]

    @pytest.fixture(scope="class")
    def valid_stages_transforms_only_1(self):
        return [
            LogTransformer(
                inputCol="col2",
                outputCol="log_col2",
                alpha=5,
            ),
            IdentityTransformer(
                inputCol="col1",
                outputCol="col1_identity",
            ),
            ArrayConcatenateTransformer(
                inputCols=["col1_identity", "log_col2", "col3"],
                outputCol="features",
            ),
            ArraySplitTransformer(
                inputCol="features",
                outputCols=["col1_sliced", "log_col2_sliced", "col3_sliced"],
            ),
        ]

    @pytest.fixture(scope="class")
    def valid_stages_0(self):
        return [
            ArrayConcatenateTransformer(
                inputCols=["col1", "col2", "col3"],
                outputCol="features",
            ),
            StandardScaleEstimator(inputCol="features", outputCol="features_scaled"),
        ]

    @pytest.fixture(scope="class")
    def expected_dataframe_stage_0(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "c",
                    [1, 2, 3],
                    [1, 2, 3],
                    [-1.2247448713915892, -0.7071067811865475, -0.7071067811865475],
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "c",
                    [4, 2, 6],
                    [4, 2, 6],
                    [0.0, -0.7071067811865475, 1.414213562373095],
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "a",
                    [7, 8, 3],
                    [7, 8, 3],
                    [1.2247448713915892, 1.414213562373095, -0.7071067811865475],
                ),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "features",
                "features_scaled",
            ],
        )

    @pytest.fixture(scope="class")
    def valid_stages_1(self):
        return [
            ArrayConcatenateTransformer(
                inputCols=["col1", "col2", "col3"],
                outputCol="features",
            ),
            StandardScaleEstimator(inputCol="features", outputCol="features_scaled"),
            StringIndexEstimator(inputCol="col4", outputCol="col4_indexed"),
        ]

    @pytest.fixture(scope="class")
    def valid_stages_1_expanded_stages(self, valid_stages_1):
        return valid_stages_1

    @pytest.fixture(scope="class")
    def valid_stages_1_parent_stages(self, valid_stages_1_expanded_stages):
        return [valid_stages_1_expanded_stages[0]]

    @pytest.fixture(scope="class")
    def expected_dataframe_stage_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "c",
                    [1, 2, 3],
                    [1, 2, 3],
                    [-1.2247448713915892, -0.7071067811865475, -0.7071067811865475],
                    1,
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "c",
                    [4, 2, 6],
                    [4, 2, 6],
                    [0.0, -0.7071067811865475, 1.414213562373095],
                    2,
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "a",
                    [7, 8, 3],
                    [7, 8, 3],
                    [1.2247448713915892, 1.414213562373095, -0.7071067811865475],
                    1,
                ),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "features",
                "features_scaled",
                "col4_indexed",
            ],
        )

    @pytest.fixture(scope="class")
    def valid_stages_2(self):
        return [
            LogTransformer(
                inputCol="col3",
                outputCol="log_col3",
                alpha=0.1,
            ),
            ArrayConcatenateTransformer(
                inputCols=["col1_col2_col3", "log_col3"],
                outputCol="features",
            ),
            StandardScaleEstimator(
                inputCol="features",
                outputCol="features_scaled",
            ),
            IdentityTransformer(
                inputCol="col4",
                outputCol="col4_identity",
            ),
        ]

    @pytest.fixture(scope="class")
    def valid_stages_2_expanded_stages(self, valid_stages_2):
        return valid_stages_2

    @pytest.fixture(scope="class")
    def valid_stages_2_parent_stages(self, valid_stages_2_expanded_stages):
        return valid_stages_2_expanded_stages[:2]

    @pytest.fixture(scope="class")
    def valid_stages_3_pipeline(self):
        return [
            KamaeSparkPipeline(
                stages=[
                    LogTransformer(
                        inputCol="col3",
                        outputCol="log_col3",
                        alpha=0.1,
                    ),
                    ArrayConcatenateTransformer(
                        inputCols=["col1_col2_col3", "log_col3"],
                        outputCol="features",
                    ),
                ]
            ),
            KamaeSparkPipeline(
                stages=[
                    StandardScaleEstimator(
                        inputCol="features",
                        outputCol="features_scaled",
                    ),
                    IdentityTransformer(
                        inputCol="col4",
                        outputCol="col4_identity",
                    ),
                ]
            ),
        ]

    @pytest.fixture(scope="class")
    def valid_stages_3_pipeline_expanded_stages(self, valid_stages_3_pipeline):
        return [
            *valid_stages_3_pipeline[0].getStages(),
            *valid_stages_3_pipeline[1].getStages(),
        ]

    @pytest.fixture(scope="class")
    def valid_stages_3_pipeline_parent_stages(
        self, valid_stages_3_pipeline_expanded_stages
    ):
        return valid_stages_3_pipeline_expanded_stages[:2]

    @pytest.fixture(scope="class")
    def valid_stages_4_pipeline(self):
        return [
            KamaeSparkPipeline(
                stages=[
                    LogTransformer(
                        inputCol="col3",
                        outputCol="log_col3",
                        alpha=0.1,
                    ),
                    KamaeSparkPipeline(
                        stages=[
                            ArrayConcatenateTransformer(
                                inputCols=["col1_col2_col3", "log_col3"],
                                outputCol="features",
                            ),
                        ]
                    ),
                ]
            ),
            KamaeSparkPipeline(
                stages=[
                    KamaeSparkPipeline(
                        stages=[
                            StandardScaleEstimator(
                                inputCol="features",
                                outputCol="features_scaled",
                            ),
                            IdentityTransformer(
                                inputCol="col4",
                                outputCol="col4_identity",
                            ),
                        ]
                    )
                ]
            ),
        ]

    @pytest.fixture(scope="class")
    def valid_stages_4_pipeline_expanded_stages(self, valid_stages_4_pipeline):
        return [
            valid_stages_4_pipeline[0].getStages()[0],
            *valid_stages_4_pipeline[0].getStages()[1].getStages(),
            *valid_stages_4_pipeline[1].getStages()[0].getStages(),
        ]

    @pytest.fixture(scope="class")
    def valid_stages_4_pipeline_parent_stages(
        self, valid_stages_4_pipeline_expanded_stages
    ):
        return valid_stages_4_pipeline_expanded_stages[:2]

    @pytest.fixture(scope="class")
    def expected_dataframe_stage_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "c",
                    [1, 2, 3],
                    1.1314021114911006,
                    [1.0, 2.0, 3.0, 1.1314021114911006],
                    [
                        -1.2247448713915892,
                        -0.7071067811865475,
                        -0.7071067811865475,
                        -0.7071067811865469,
                    ],
                    "a",
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "c",
                    [4, 2, 6],
                    1.8082887711792655,
                    [4.0, 2.0, 6.0, 1.8082887711792655],
                    [0.0, -0.7071067811865475, 1.414213562373095, 1.4142135623730958],
                    "b",
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "a",
                    [7, 8, 3],
                    1.1314021114911006,
                    [7.0, 8.0, 3.0, 1.1314021114911006],
                    [
                        1.2247448713915892,
                        1.414213562373095,
                        -0.7071067811865475,
                        -0.7071067811865469,
                    ],
                    "a",
                ),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "log_col3",
                "features",
                "features_scaled",
                "col4_indentity",
            ],
        )

    @pytest.fixture(scope="class")
    def valid_stages_with_same_inputs_diff_types(self):
        # col1 is DoubleType but the indexers need strings. Here we show that we don't
        # change the original schema and can reuse the same input column, even if it
        # has been cast to string for other transforms
        return [
            StringIndexEstimator(
                inputCol="col1",
                outputCol="col1_indexed",
                inputDtype="string",
            ),
            LogTransformer(
                inputCol="col1",
                outputCol="col1_logged",
                alpha=0.1,
            ),
            HashIndexTransformer(
                inputCol="col1",
                outputCol="col1_hashed",
                numBins=100,
                inputDtype="string",
            ),
            BucketizeTransformer(
                inputCol="col1",
                outputCol="col1_bucketed",
                splits=[0, 1, 2, 3],
            ),
            SubtractTransformer(
                inputCol="col1",
                outputCol="col1_subtracted",
                mathFloatConstant=1.0,
            ),
        ]

    @pytest.fixture(scope="class")
    def valid_stages_with_uid_set_same_as_input(self):
        # Previously setting the uid to a subset of the input column name would cause
        # an error. This test ensures that the uid can be set to the same value as the
        # input column name.
        return [
            StringIndexEstimator(
                inputCol="col1",
                outputCol="col1_indexed",
                inputDtype="string",
                outputDtype="double",
            ),
            LogTransformer(
                inputCol="col1_indexed",
                outputCol="col1_indexed_logged",
                alpha=0.1,
            )._resetUid("indexed"),
        ]

    @pytest.mark.parametrize(
        "stages",
        [
            "valid_stages_0",
            "valid_stages_1",
            "valid_stages_2",
        ],
    )
    def test_spark_read_write_pipeline(self, test_dir, stages, request):
        stages = request.getfixturevalue(stages)
        pipeline = KamaeSparkPipeline(stages=stages)
        pipeline.save(f"{test_dir}/pipeline")
        pipeline_loaded = KamaeSparkPipeline.load(f"{test_dir}/pipeline")
        assert pipeline.stages == pipeline_loaded.stages

    @pytest.mark.parametrize(
        "stages, expanded_stages",
        [
            ("valid_stages_1", "valid_stages_1_expanded_stages"),
            ("valid_stages_2", "valid_stages_2_expanded_stages"),
            ("valid_stages_3_pipeline", "valid_stages_3_pipeline_expanded_stages"),
            ("valid_stages_4_pipeline", "valid_stages_4_pipeline_expanded_stages"),
        ],
    )
    def test_spark_pipeline_expand_stages(self, stages, expanded_stages, request):
        stages = request.getfixturevalue(stages)
        expanded_stages = request.getfixturevalue(expanded_stages)
        pipeline = KamaeSparkPipeline(stages=stages)
        assert pipeline.expand_pipeline_stages() == expanded_stages

    @pytest.mark.parametrize(
        "stages, parent_stages",
        [
            ("valid_stages_1_expanded_stages", "valid_stages_1_parent_stages"),
            ("valid_stages_2_expanded_stages", "valid_stages_2_parent_stages"),
            (
                "valid_stages_3_pipeline_expanded_stages",
                "valid_stages_3_pipeline_parent_stages",
            ),
            (
                "valid_stages_4_pipeline_expanded_stages",
                "valid_stages_4_pipeline_parent_stages",
            ),
        ],
    )
    def test_collect_estimator_parents(self, stages, parent_stages, request):
        stages = request.getfixturevalue(stages)
        parent_stages = request.getfixturevalue(parent_stages)
        pipeline = KamaeSparkPipeline(stages=stages)
        assert pipeline.collect_estimator_parents(stages) == parent_stages

    @pytest.mark.parametrize(
        "stages",
        [
            "valid_stages_transforms_only_0",
            "valid_stages_transforms_only_1",
        ],
    )
    def test_spark_read_write_pipeline_model(
        self, test_dir, stages, example_dataframe, request
    ):
        stages = request.getfixturevalue(stages)
        pipeline_model = KamaeSparkPipelineModel(stages=stages)
        pipeline_model.save(f"{test_dir}/pipeline_model")
        pipeline_model_loaded = KamaeSparkPipelineModel.load(
            f"{test_dir}/pipeline_model"
        )

        transformed_data = pipeline_model.transform(example_dataframe)
        transformed_data_loaded = pipeline_model_loaded.transform(example_dataframe)
        diff = transformed_data.exceptAll(transformed_data_loaded)

        assert (
            diff.isEmpty()
        ), f"PipelineModelKeras loaded from disk is not the same as the original one."

    @pytest.mark.parametrize(
        "stages, expected_dataframe",
        [
            ("valid_stages_0", "expected_dataframe_stage_0"),
            ("valid_stages_1", "expected_dataframe_stage_1"),
            ("valid_stages_2", "expected_dataframe_stage_2"),
            ("valid_stages_3_pipeline", "expected_dataframe_stage_2"),
            ("valid_stages_4_pipeline", "expected_dataframe_stage_2"),
        ],
    )
    def test_spark_pipeline(
        self, stages, example_dataframe, expected_dataframe, request
    ):
        stages = request.getfixturevalue(stages)
        pipeline = KamaeSparkPipeline(stages=stages)

        pipeline_model = pipeline.fit(example_dataframe)

        transformed_df = pipeline_model.transform(example_dataframe)
        diff = transformed_df.exceptAll(request.getfixturevalue(expected_dataframe))
        assert diff.isEmpty(), f"PipelineKeras output is not the same as expected."

    @pytest.mark.parametrize(
        "stages, input_col, original_dtype",
        [
            ("valid_stages_with_same_inputs_diff_types", "col1", DoubleType),
        ],
    )
    def test_spark_pipeline_with_same_inputs_diff_types(
        self, stages, input_col, original_dtype, example_dataframe, request
    ):
        """
        Tests that a pipeline using the same inputs but needing differing types for each
        stage works as expected and does not edit the original schema.
        """
        stages = request.getfixturevalue(stages)
        pipeline = KamaeSparkPipeline(stages=stages)
        pipeline_model = pipeline.fit(example_dataframe)
        transformed_df = pipeline_model.transform(example_dataframe)

        assert isinstance(transformed_df.schema[input_col].dataType, original_dtype)

    @pytest.mark.parametrize(
        "stages, input_col",
        [
            ("valid_stages_with_uid_set_same_as_input", "col1"),
        ],
    )
    def test_spark_pipeline_with_uid_same_as_input(
        self, stages, input_col, example_dataframe, request
    ):
        """
        Tests that a pipeline using the same inputs but needing differing types for each
        stage works as expected and does not edit the original schema.
        """
        stages = request.getfixturevalue(stages)
        pipeline = KamaeSparkPipeline(stages=stages)
        pipeline_model = pipeline.fit(example_dataframe)
        transformed_df = pipeline_model.transform(example_dataframe)
        transformed_df.count()

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
                    {"name": "col1", "dtype": tf.float32, "shape": (None, 1)},
                    {"name": "col2", "dtype": tf.float32, "shape": (None, 1)},
                    {"name": "col3", "dtype": tf.float32, "shape": (None, 1)},
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
                "valid_stages_2",
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
                    {"name": "col1_col2_col3", "dtype": tf.float32, "shape": (None, 3)},
                    {"name": "col3", "dtype": tf.float32, "shape": (None, 1)},
                    {"name": "col4", "dtype": tf.string, "shape": (None, 1)},
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
                    {"name": "col1", "dtype": tf.float32, "shape": (None, 1)},
                    {"name": "col2", "dtype": tf.float32, "shape": (None, 1)},
                    {"name": "col3", "dtype": tf.float32, "shape": (None, 1)},
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
                    {"name": "col1", "dtype": tf.float32, "shape": (None, 1)},
                    {"name": "col2", "dtype": tf.float32, "shape": (None, 1)},
                    {"name": "col3", "dtype": tf.float32, "shape": (None, 1)},
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
        pipeline = KamaeSparkPipeline(stages=stages)

        pipeline_model = pipeline.fit(example_dataframe)

        keras_model = pipeline_model.build_keras_model(tf_input_schema=tf_input_schema)

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
