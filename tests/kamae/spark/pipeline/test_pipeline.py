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
from unittest.mock import patch

import pytest
import tensorflow as tf
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType

from kamae.spark.estimators import (
    ConditionalStandardScaleEstimator,
    StandardScaleEstimator,
    StringIndexEstimator,
)
from kamae.spark.pipeline import KamaeSparkPipeline, KamaeSparkPipelineModel
from kamae.spark.transformers import (
    ArrayConcatenateTransformer,
    ArraySplitTransformer,
    BucketizeTransformer,
    HashIndexTransformer,
    IdentityTransformer,
    ListMeanTransformer,
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
        "stages",
        [
            "valid_stages_1",
            "valid_stages_2",
        ],
    )
    def test_spark_pipeline_checkpoint_is_transparent(
        self, stages, example_dataframe, request
    ):
        """
        checkpoint(eager=True) only truncates lineage, so fitting with a positive
        checkpointInterval must yield results identical to the default of 0.
        """
        stages = request.getfixturevalue(stages)

        baseline_model = KamaeSparkPipeline(stages=stages, checkpointInterval=0).fit(
            example_dataframe
        )
        checkpointed_model = KamaeSparkPipeline(
            stages=stages, checkpointInterval=2
        ).fit(example_dataframe)

        baseline_df = baseline_model.transform(example_dataframe)
        checkpointed_df = checkpointed_model.transform(example_dataframe)

        assert baseline_df.schema == checkpointed_df.schema
        assert baseline_df.exceptAll(checkpointed_df).isEmpty()
        assert checkpointed_df.exceptAll(baseline_df).isEmpty()

    def test_spark_pipeline_checkpoint_invocation(
        self, valid_stages_1, example_dataframe
    ):
        """
        checkpoint must be invoked during fit only when checkpointInterval > 0.
        """
        original_checkpoint = DataFrame.checkpoint

        with patch.object(
            DataFrame,
            "checkpoint",
            autospec=True,
            side_effect=original_checkpoint,
        ) as mock_checkpoint:
            KamaeSparkPipeline(stages=valid_stages_1, checkpointInterval=0).fit(
                example_dataframe
            )
            assert mock_checkpoint.call_count == 0

            mock_checkpoint.reset_mock()
            KamaeSparkPipeline(stages=valid_stages_1, checkpointInterval=2).fit(
                example_dataframe
            )
            assert mock_checkpoint.call_count > 0

    def test_spark_pipeline_checkpoint_bounds_plan_depth(self, spark_session):
        """
        The point of checkpointInterval is to bound logical-plan depth. We build a
        deep, linearly-dependent pipeline (every stage is an ancestor of the next, so
        the working DataFrame is advanced at every fit and lineage keeps growing) and
        capture the logical-plan size of the DataFrame handed to each estimator fit.
        With a positive interval the plan must stay markedly smaller than the default.
        """
        df = spark_session.createDataFrame(
            [(1.0,), (4.0,), (7.0,), (2.0,), (9.0,)],
            ["col0"],
        )

        num_blocks = 4
        transforms_per_block = 4

        def build_stages():
            stages = []
            prev = "col0"
            for b in range(num_blocks):
                for t in range(transforms_per_block):
                    out = f"t_{b}_{t}"
                    stages.append(
                        SubtractTransformer(
                            inputCol=prev, outputCol=out, mathFloatConstant=1.0
                        )
                    )
                    prev = out
                out = f"s_{b}"
                stages.append(StandardScaleEstimator(inputCol=prev, outputCol=out))
                prev = out
            return stages

        def max_fit_plan_length(interval):
            plan_lengths = []
            original_fit = StandardScaleEstimator.fit

            def spy_fit(estimator, dataset, *args, **kwargs):
                plan = dataset._jdf.queryExecution().logical().toString()
                plan_lengths.append(len(plan))
                return original_fit(estimator, dataset, *args, **kwargs)

            with patch.object(StandardScaleEstimator, "fit", spy_fit):
                KamaeSparkPipeline(
                    stages=build_stages(), checkpointInterval=interval
                ).fit(df)
            return max(plan_lengths)

        baseline_max = max_fit_plan_length(0)
        checkpointed_max = max_fit_plan_length(transforms_per_block + 1)

        # Checkpointing must keep the deepest fit-time plan well below the un-bounded
        # baseline. A strict 2x margin is robust to Spark-version plan-string changes.
        assert checkpointed_max * 2 < baseline_max, (
            f"plan not bounded: baseline_max={baseline_max}, "
            f"checkpointed_max={checkpointed_max}"
        )

    def test_spark_pipeline_prunes_unused_input_columns(
        self, valid_stages_1, example_dataframe
    ):
        """
        prune_unused_input_columns must keep the columns the pipeline reads
        (col1/col2/col3 via ArrayConcatenate, col4 via StringIndex) and drop the
        unused col5 and col1_col2_col3, while preserving every row.
        """
        pipeline = KamaeSparkPipeline(stages=valid_stages_1)

        # The required set is generous (a superset), so assert containment rather
        # than equality - it also carries output/param strings that harmlessly do
        # not match any input-DataFrame column.
        required = pipeline.collect_required_input_columns(valid_stages_1)
        assert {"col1", "col2", "col3", "col4"}.issubset(required)

        pruned = pipeline.prune_unused_input_columns(example_dataframe, valid_stages_1)

        assert pruned.columns == ["col1", "col2", "col3", "col4"]
        assert pruned.exceptAll(
            example_dataframe.select("col1", "col2", "col3", "col4")
        ).isEmpty()

    def test_collect_required_input_columns_includes_aux_columns(self):
        """
        Aux columns read at fit time via params other than inputCol(s) - here
        maskCols and relevanceCol on ConditionalStandardScaleEstimator, and
        queryIdCol on a listwise transformer - must be reported by the collector so
        pruning does not drop them. No Spark session needed.
        """
        stages = [
            ConditionalStandardScaleEstimator(
                inputCol="x",
                outputCol="x_scaled",
                maskCols=["m"],
                relevanceCol="r",
            ),
            ListMeanTransformer(
                inputCol="p",
                outputCol="p_list_mean",
                queryIdCol="q",
            ),
        ]

        required = KamaeSparkPipeline.collect_required_input_columns(stages)

        assert {"x", "m", "r", "p", "q"} <= required

    def test_collect_required_input_columns_plain_estimator(self):
        """
        A stage with no aux column params must still report its inputCol and must
        not gain spurious columns - confirms the aux sweep does not regress the
        simple case.
        """
        stages = [StandardScaleEstimator(inputCol="x", outputCol="x_scaled")]

        required = KamaeSparkPipeline.collect_required_input_columns(stages)

        assert "x" in required

    def test_spark_pipeline_prune_input_columns_is_opt_in(
        self, valid_stages_1, example_dataframe
    ):
        """
        Pruning must only happen when pruneInputColumns is True. With the default
        (False) the input DataFrame is not projected during fit.
        """
        with patch.object(
            KamaeSparkPipeline,
            "prune_unused_input_columns",
            autospec=True,
            side_effect=KamaeSparkPipeline.prune_unused_input_columns,
        ) as mock_prune:
            KamaeSparkPipeline(stages=valid_stages_1).fit(example_dataframe)
            assert mock_prune.call_count == 0

            mock_prune.reset_mock()
            KamaeSparkPipeline(stages=valid_stages_1, pruneInputColumns=True).fit(
                example_dataframe
            )
            assert mock_prune.call_count == 1

    def test_spark_pipeline_prune_keeps_aux_fit_columns(self, spark_session):
        """
        Regression: pruning must not drop columns an estimator reads at fit time
        through params other than inputCol (here maskCols). The fit must not raise,
        the genuinely-unused column must be pruned, and the fitted moments must be
        identical to a prune-disabled baseline (numerically transparent).
        """
        df = spark_session.createDataFrame(
            [(1.0, 1, 3.0, 99.0), (2.0, 0, 1.0, 99.0), (3.0, 1, 2.0, 99.0)],
            ["x", "m", "r", "junk"],
        )

        def build_pipeline(prune):
            return KamaeSparkPipeline(
                stages=[
                    ConditionalStandardScaleEstimator(
                        inputCol="x",
                        outputCol="x_scaled",
                        maskCols=["m"],
                        maskOperators=["eq"],
                        maskValues=[1.0],
                        relevanceCol="r",
                    ),
                ],
                pruneInputColumns=prune,
            )

        pruned_pipeline = build_pipeline(prune=True)

        # Aux fit columns kept, genuinely-unused column dropped.
        required = pruned_pipeline.collect_required_input_columns(
            pruned_pipeline.getStages()
        )
        assert {"x", "m", "r"} <= required
        assert "junk" not in required

        # Must NOT raise UNRESOLVED_COLUMN / "Mask column m not found".
        pruned_model = pruned_pipeline.fit(df)
        baseline_model = build_pipeline(prune=False).fit(df)

        pruned_scaler = pruned_model.stages[-1]
        baseline_scaler = baseline_model.stages[-1]

        assert pruned_scaler.getMean() == baseline_scaler.getMean()
        assert pruned_scaler.getStddev() == baseline_scaler.getStddev()

    def test_spark_pipeline_prune_is_transparent_to_fit(
        self, valid_stages_1, example_dataframe
    ):
        """
        Pruning drops only columns no stage reads, so a fitted model - and its
        transform output - must be identical whether or not the input carries an
        extra unused column when pruneInputColumns is enabled.
        """
        with_extra = example_dataframe.withColumn(
            "unused", example_dataframe["col1"] * 100.0
        )

        baseline_out = (
            KamaeSparkPipeline(stages=valid_stages_1, pruneInputColumns=True)
            .fit(example_dataframe)
            .transform(example_dataframe)
        )
        with_extra_out = (
            KamaeSparkPipeline(stages=valid_stages_1, pruneInputColumns=True)
            .fit(with_extra)
            .transform(example_dataframe)
        )

        assert baseline_out.schema == with_extra_out.schema
        assert baseline_out.exceptAll(with_extra_out).isEmpty()
        assert with_extra_out.exceptAll(baseline_out).isEmpty()

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
        "stages, input_tensors, input_schema, output_names, expected_output",
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
                None,
                {
                    "features_scaled": tf.constant(
                        [
                            [
                                [-1.2247448, -0.70710677, -0.70710677],
                                [0.0, -0.70710677, 1.4142135],
                                [1.2247448, 1.4142135, -0.70710677],
                            ]
                        ]
                    ),
                },
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
                None,
                {
                    "col4_identity": tf.constant(
                        [
                            [["a"], ["b"], ["a"]],
                        ],
                        dtype=tf.string,
                    ),
                    "features_scaled": tf.constant(
                        [
                            [
                                [-1.2247448, -0.70710677, -0.70710677, -0.7071067],
                                [0.0, -0.70710677, 1.4142135, 1.4142138],
                                [1.2247448, 1.4142135, -0.70710677, -0.7071067],
                            ]
                        ],
                        dtype=tf.float32,
                    ),
                },
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
                None,
                {
                    "log_col1_sliced": tf.constant(
                        [
                            [[0.0953102], [1.4109869], [1.9600948]],
                        ],
                        dtype=tf.float32,
                    ),
                    "col2_sliced": tf.constant(
                        [
                            [[2.0], [2.0], [8.0]],
                        ],
                        dtype=tf.float32,
                    ),
                    "col3_sliced": tf.constant(
                        [
                            [[3.0], [6.0], [3.0]],
                        ],
                        dtype=tf.float32,
                    ),
                },
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
                None,
                {
                    "col1_sliced": tf.constant(
                        [
                            [[1.0], [4.0], [7.0]],
                        ],
                        dtype=tf.float32,
                    ),
                    "log_col2_sliced": tf.constant(
                        [
                            [[1.9459101], [1.9459101], [2.5649493]],
                        ],
                        dtype=tf.float32,
                    ),
                    "col3_sliced": tf.constant(
                        [
                            [[3.0], [6.0], [3.0]],
                        ],
                        dtype=tf.float32,
                    ),
                },
            ),
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
                ["features_scaled"],
                {
                    "features_scaled": tf.constant(
                        [
                            [
                                [-1.2247448, -0.70710677, -0.70710677],
                                [0.0, -0.70710677, 1.4142135],
                                [1.2247448, 1.4142135, -0.70710677],
                            ]
                        ]
                    ),
                },
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
                ["col4_identity"],
                {
                    "col4_identity": tf.constant(
                        [
                            [["a"], ["b"], ["a"]],
                        ],
                        dtype=tf.string,
                    ),
                },
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
                ["log_col1_sliced", "col3_sliced"],
                {
                    "log_col1_sliced": tf.constant(
                        [
                            [[0.0953102], [1.4109869], [1.9600948]],
                        ],
                        dtype=tf.float32,
                    ),
                    "col3_sliced": tf.constant(
                        [
                            [[3.0], [6.0], [3.0]],
                        ],
                        dtype=tf.float32,
                    ),
                },
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
                ["col1_sliced", "log_col2_sliced"],
                {
                    "col1_sliced": tf.constant(
                        [
                            [[1.0], [4.0], [7.0]],
                        ],
                        dtype=tf.float32,
                    ),
                    "log_col2_sliced": tf.constant(
                        [
                            [[1.9459101], [1.9459101], [2.5649493]],
                        ],
                        dtype=tf.float32,
                    ),
                },
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
                    "col4": tf.constant(
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
                    {"name": "col4", "dtype": tf.float32, "shape": (None, 1)},
                ],
                ["col1_sliced", "log_col2_sliced", "col4"],
                {
                    "col1_sliced": tf.constant(
                        [
                            [[1.0], [4.0], [7.0]],
                        ],
                        dtype=tf.float32,
                    ),
                    "log_col2_sliced": tf.constant(
                        [
                            [[1.9459101], [1.9459101], [2.5649493]],
                        ],
                        dtype=tf.float32,
                    ),
                    "col4": tf.constant(
                        [
                            [[3.0], [6.0], [3.0]],
                        ],
                        dtype=tf.float32,
                    ),
                },
            ),
        ],
    )
    def test_keras_model(
        self,
        stages,
        input_tensors,
        input_schema,
        output_names,
        expected_output,
        example_dataframe,
        request,
    ):
        stages = request.getfixturevalue(stages)
        pipeline = KamaeSparkPipeline(stages=stages)

        pipeline_model = pipeline.fit(example_dataframe)

        keras_model = pipeline_model.build_keras_model(
            input_schema=input_schema, output_names=output_names
        )

        actual = keras_model(input_tensors)

        for k, v in actual.items():
            expected = expected_output[k]
            if v.dtype == "string":
                tf.debugging.assert_equal(v, expected)
            else:
                tf.debugging.assert_near(v, expected, atol=1e-6)
