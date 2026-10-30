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

"""Edge cases of fuseEstimatorFits: the fused fit must behave like the normal fit."""

import warnings

import numpy as np
import pytest

from kamae.spark.estimators import (
    ConditionalStandardScaleEstimator,
    StringIndexEstimator,
)
from kamae.spark.pipeline import KamaeSparkPipeline
from kamae.spark.transformers import LogTransformer


def _fit(stages_fn, dataset, fuse):
    return KamaeSparkPipeline(stages=stages_fn(), fuseEstimatorFits=fuse).fit(dataset)


def _fitted_values(model):
    values = []
    for stage in model.stages:
        if stage.hasParam("labelsArray") and stage.isDefined("labelsArray"):
            values.append((stage.getInputCol(), stage.getLabelsArray()))
        elif stage.hasParam("mean") and stage.isDefined("mean"):
            values.append((stage.getInputCol(), stage.getMean(), stage.getStddev()))
    return values


def _assert_fused_matches_normal(stages_fn, dataset):
    expected = _fitted_values(_fit(stages_fn, dataset, fuse=False))
    actual = _fitted_values(_fit(stages_fn, dataset, fuse=True))
    assert len(actual) == len(expected)
    for e, a in zip(expected, actual):
        assert e[0] == a[0]
        if len(e) == 2:
            assert e[1] == a[1]
        else:
            np.testing.assert_allclose(e[1], a[1], rtol=1e-12)
            np.testing.assert_allclose(e[2], a[2], rtol=1e-12)
    return actual


def test_relevance_column_produced_by_a_transform(spark_session):
    # rel is produced by a transform and only read as relevanceCol by a layer-1
    # scaler: it must count as a dependency of that scaler.
    df = spark_session.createDataFrame(
        [(1.0, "a"), (3.0, "b"), (0.0, "a"), (2.0, "c")], "r double, s string"
    )

    def stages():
        return [
            LogTransformer(inputCol="r", outputCol="rel", layerName="rel"),
            ConditionalStandardScaleEstimator(
                inputCol="rel", outputCol="rel_scaled", layerName="rel_scaled"
            ),
            StringIndexEstimator(
                inputCol="s",
                outputCol="s_idx",
                layerName="s_idx",
                stringOrderType="alphabeticalAsc",
            ),
            ConditionalStandardScaleEstimator(
                inputCol="s_idx",
                outputCol="s_idx_scaled",
                layerName="s_idx_scaled",
                inputDtype="double",
                relevanceCol="rel",
            ),
        ]

    layers = KamaeSparkPipeline.estimator_layers(stages())
    assert [[e.getLayerName() for e in layer] for layer in layers] == [
        ["rel_scaled", "s_idx"],
        ["s_idx_scaled"],
    ]
    _assert_fused_matches_normal(stages, df)


def test_array_size_probed_on_first_row_meeting_the_masks(spark_session):
    df = spark_session.createDataFrame(
        [(None, 0), ([1.0, 2.0], 1), ([3.0, 5.0], 1)], "x array<double>, m int"
    ).coalesce(1)

    def stages():
        return [
            ConditionalStandardScaleEstimator(
                inputCol="x",
                outputCol="x_scaled",
                layerName="x_scaled",
                maskCols=["m"],
                maskOperators=["eq"],
                maskValues=[1],
            )
        ]

    actual = _assert_fused_matches_normal(stages, df)
    assert actual[0][1] == [2.0, 3.5]


@pytest.mark.parametrize("nan_fill_value", [None, 0.0])
def test_masks_removing_every_row_raise(spark_session, nan_fill_value):
    df = spark_session.createDataFrame(
        [([1.0, 2.0], 0), ([3.0, 5.0], 0)], "x array<double>, m int"
    )

    def stages():
        return [
            ConditionalStandardScaleEstimator(
                inputCol="x",
                outputCol="x_scaled",
                layerName="x_scaled",
                maskCols=["m"],
                maskOperators=["eq"],
                maskValues=[1],
                nanFillValue=nan_fill_value,
            )
        ]

    for fuse in [False, True]:
        with pytest.raises(ValueError, match="Error in estimator"):
            _fit(stages, df, fuse=fuse)


def test_empty_dataset(spark_session):
    df = spark_session.createDataFrame([], "x array<double>, s string")

    def indexer_only():
        return [
            StringIndexEstimator(
                inputCol="s",
                outputCol="s_idx",
                layerName="s_idx",
                stringOrderType="alphabeticalAsc",
            )
        ]

    assert _assert_fused_matches_normal(indexer_only, df) == [("s", [])]

    def with_scaler():
        return indexer_only() + [
            ConditionalStandardScaleEstimator(
                inputCol="x", outputCol="x_scaled", layerName="x_scaled"
            )
        ]

    for fuse in [False, True]:
        with pytest.raises(ValueError, match="Error in estimator"):
            _fit(with_scaler, df, fuse=fuse)


def test_copied_estimators_sharing_a_uid(spark_session):
    df = spark_session.createDataFrame(
        [("a", "x", [1.0], [10.0]), ("b", "y", [3.0], [30.0])],
        "s string, t string, u array<double>, v array<double>",
    )

    def stages():
        index_s = StringIndexEstimator(
            inputCol="s",
            outputCol="s_idx",
            layerName="s_idx",
            stringOrderType="alphabeticalAsc",
        )
        index_t = index_s.copy()
        index_t.setInputCol("t")
        index_t.setOutputCol("t_idx")
        index_t.setLayerName("t_idx")
        scale_u = ConditionalStandardScaleEstimator(
            inputCol="u", outputCol="u_scaled", layerName="u_scaled"
        )
        scale_v = scale_u.copy()
        scale_v.setInputCol("v")
        scale_v.setOutputCol("v_scaled")
        scale_v.setLayerName("v_scaled")
        assert index_s.uid == index_t.uid and scale_u.uid == scale_v.uid
        return [index_s, index_t, scale_u, scale_v]

    actual = _assert_fused_matches_normal(stages, df)
    assert [a[1] for a in actual] == [["a", "b"], ["x", "y"], [2.0], [20.0]]


@pytest.mark.parametrize("uid", ["feat.scaled", "feat/scaled", "feat`x.y"])
def test_uids_with_special_characters(spark_session, uid):
    df = spark_session.createDataFrame(
        [([1.0], "a"), ([3.0], "b")], "x array<double>, s string"
    )

    def stages():
        scaler = ConditionalStandardScaleEstimator(
            inputCol="x", outputCol="x_scaled", layerName="x_scaled"
        )
        scaler._resetUid(uid)
        indexer = StringIndexEstimator(
            inputCol="s",
            outputCol="s_idx",
            layerName="s_idx",
            stringOrderType="alphabeticalAsc",
        )
        indexer._resetUid(uid + "_idx")
        return [scaler, indexer]

    _assert_fused_matches_normal(stages, df)


def test_failure_resets_estimator_input_columns(spark_session):
    df = spark_session.createDataFrame([(1.0, 1)], "x double, n int")
    scaler = ConditionalStandardScaleEstimator(
        inputCol="x", outputCol="x_scaled", layerName="x_scaled", inputDtype="float"
    )
    # An int input is not compatible with StringIndexEstimator.
    indexer = StringIndexEstimator(
        inputCol="n",
        outputCol="n_idx",
        layerName="n_idx",
        stringOrderType="alphabeticalAsc",
    )
    with pytest.raises(TypeError, match="Error in estimator"):
        KamaeSparkPipeline(stages=[scaler, indexer], fuseEstimatorFits=True).fit(df)
    assert scaler.getInputCol() == "x"
    assert indexer.getInputCol() == "n"
    KamaeSparkPipeline(stages=[scaler]).fit(df)


def test_overwritten_column_falls_back_to_normal_fit(spark_session):
    df = spark_session.createDataFrame([(1.0, 2.0), (3.0, 4.0)], "a double, c double")

    def stages():
        return [
            ConditionalStandardScaleEstimator(
                inputCol="c", outputCol="c_scaled", layerName="c_scaled"
            ),
            LogTransformer(inputCol="a", outputCol="c", layerName="c", alpha=1.0),
            ConditionalStandardScaleEstimator(
                inputCol="c", outputCol="c_scaled_again", layerName="c_scaled_again"
            ),
        ]

    assert KamaeSparkPipeline.fused_fit_column_conflicts(stages()) == {"c"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = _assert_fused_matches_normal(stages, df)
    assert any("fuseEstimatorFits is ignored" in str(w.message) for w in caught)
    assert actual[0][1] == [3.0]


def test_rewritten_column_read_only_after_the_rewrite_is_fused(spark_session):
    # c exists in the input but is rewritten before anything reads it, so the fused
    # fit is exact and is used.
    df = spark_session.createDataFrame([(1.0, 2.0), (3.0, 4.0)], "a double, c double")

    def stages():
        return [
            LogTransformer(inputCol="a", outputCol="c", layerName="c", alpha=1.0),
            ConditionalStandardScaleEstimator(
                inputCol="c", outputCol="c_scaled", layerName="c_scaled"
            ),
            ConditionalStandardScaleEstimator(
                inputCol="a", outputCol="a_scaled", layerName="a_scaled"
            ),
        ]

    assert KamaeSparkPipeline.fused_fit_column_conflicts(stages()) == set()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _assert_fused_matches_normal(stages, df)
    assert not any("fuseEstimatorFits is ignored" in str(w.message) for w in caught)


@pytest.mark.parametrize("input_dtype", [None, "float"])
def test_normal_fit_releases_its_persisted_dataset(spark_session, input_dtype):
    from kamae.spark.estimators import StandardScaleEstimator

    df = spark_session.createDataFrame(
        [([1.0, 2.0],), ([3.0, 5.0],), ([4.0, 0.0],)], "x array<double>"
    )
    persistent = spark_session.sparkContext._jsc.getPersistentRDDs
    before = persistent().size()
    for estimator in [
        ConditionalStandardScaleEstimator(
            inputCol="x", outputCol="x_c", layerName="x_c", inputDtype=input_dtype
        ),
        StandardScaleEstimator(
            inputCol="x", outputCol="x_s", layerName="x_s", inputDtype=input_dtype
        ),
    ]:
        estimator.fit(df)
        assert persistent().size() == before


def test_fused_helper_columns_never_collide_with_existing_columns(
    spark_session, monkeypatch
):
    from kamae.spark.common import SparkOperation

    # The dataset already has a column named like a fused helper column, holding
    # data that would corrupt the fit if the helper overwrote or reused it.
    taken = "__fused_TAKEN_0__element_struct"
    df = spark_session.createDataFrame(
        [([1.0, 2.0], "junk"), ([3.0, 5.0], "junk"), ([4.0, 0.0], "junk")],
        f"x array<double>, `{taken}` string",
    )

    def stages():
        return [
            ConditionalStandardScaleEstimator(
                inputCol="x", outputCol="x_scaled", layerName="x_scaled"
            )
        ]

    expected = _fitted_values(_fit(stages, df, fuse=False))
    estimators = stages()  # constructed before the generator is patched
    calls = []

    def first_suggestion_taken(str_len=25):
        # The first random name collides with the existing column; later ones
        # (and those of the transformers created by the fit) are free.
        calls.append(str_len)
        return "TAKEN" if len(calls) == 1 else "FREE"

    monkeypatch.setattr(
        SparkOperation,
        "generate_tmp_column_suffix",
        staticmethod(first_suggestion_taken),
    )
    assert KamaeSparkPipeline._fused_column_prefix(df.columns) == "__fused_FREE_"

    calls.clear()
    actual = _fitted_values(
        KamaeSparkPipeline(stages=estimators, fuseEstimatorFits=True).fit(df)
    )
    assert actual == expected
    assert df.select(taken).distinct().collect()[0][0] == "junk"
