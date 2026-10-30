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

"""Dependency layering of fuseEstimatorFits: fused fit must match the normal fit."""

import numpy as np
import pytest

from kamae.spark.estimators import (
    BaseEstimator,
    ConditionalStandardScaleEstimator,
    StringIndexEstimator,
)
from kamae.spark.pipeline import KamaeSparkPipeline
from kamae.spark.transformers import ArrayConcatenateTransformer, LogTransformer

# Moments fused into one aggregation may be summed in a different order.
MOMENT_RTOL = 1e-12


@pytest.fixture(scope="module")
def layers_dataframe(spark_session):
    rng = np.random.default_rng(1)
    rows = [
        (
            [float(x) for x in rng.normal(0, 1, 3)],
            float(abs(rng.normal(10, 3))),
            int(rng.integers(0, 3)),
            str(rng.choice(["b", "a", "c", "d"])),
            int(rng.integers(0, 5)),
        )
        for _ in range(200)
    ]
    return spark_session.createDataFrame(
        rows, "a array<double>, c double, rel int, s string, n int"
    ).repartition(3)


def _scaler(name, col, **kwargs):
    return ConditionalStandardScaleEstimator(
        inputCol=col, outputCol=name, layerName=name, **kwargs
    )


def _index(name, col, **kwargs):
    return StringIndexEstimator(
        inputCol=col,
        outputCol=name,
        layerName=name,
        stringOrderType="alphabeticalAsc",
        **kwargs,
    )


def _chain():
    # s_idx (layer 0) -> s_idx_scaled (layer 1) -> s_idx_rescaled (layer 2)
    return [
        _index("s_idx", "s"),
        _scaler("s_idx_scaled", "s_idx", inputDtype="double", relevanceCol="rel"),
        _scaler("s_idx_rescaled", "s_idx_scaled"),
    ]


def _diamond():
    # s_idx and n_idx (layer 0) both feed one concatenated array -> scaler (layer 1)
    return [
        _index("s_idx", "s"),
        _index("n_idx", "n", inputDtype="string"),
        ArrayConcatenateTransformer(
            inputCols=["s_idx", "n_idx"], outputCol="sn", layerName="sn"
        ),
        _scaler("sn_scaled", "sn", inputDtype="double"),
    ]


def _transform_after_estimator():
    # c_log is defined after an unrelated estimator but is still needed in layer 0;
    # c_log_scaled (layer 0) and a_scaled (layer 0) do not depend on each other.
    return [
        _scaler("a_scaled", "a", relevanceCol="rel"),
        LogTransformer(inputCol="c", outputCol="c_log", layerName="c_log", alpha=1.0),
        _scaler("c_log_scaled", "c_log", relevanceCol="rel", skipZeros=True),
        _index("s_idx", "s"),
        _scaler("s_idx_scaled", "s_idx", inputDtype="double"),
    ]


def _nested():
    # The chain inside an inner pipeline, plus an estimator outside it that depends
    # on the inner pipeline's last output.
    return [
        KamaeSparkPipeline(stages=_chain()),
        _scaler("outer_scaled", "s_idx_rescaled", relevanceCol="rel"),
    ]


@pytest.mark.parametrize(
    "stages_fn, expected_layers",
    [
        (_chain, [["s_idx"], ["s_idx_scaled"], ["s_idx_rescaled"]]),
        (_diamond, [["s_idx", "n_idx"], ["sn_scaled"]]),
        (
            _transform_after_estimator,
            [["a_scaled", "c_log_scaled", "s_idx"], ["s_idx_scaled"]],
        ),
        (
            _nested,
            [["s_idx"], ["s_idx_scaled"], ["s_idx_rescaled"], ["outer_scaled"]],
        ),
    ],
)
def test_fused_fit_matches_normal_fit_across_layers(
    layers_dataframe, stages_fn, expected_layers, monkeypatch
):
    expanded = KamaeSparkPipeline(stages=stages_fn()).expand_pipeline_stages()
    layers = KamaeSparkPipeline.estimator_layers(expanded)
    assert [[e.getLayerName() for e in layer] for layer in layers] == expected_layers

    expected = KamaeSparkPipeline(stages=stages_fn()).fit(layers_dataframe)

    # Record which estimators each fused call fits, to check the fused path runs
    # once per layer, in layer order.
    fused_calls = []
    fit_estimators_fused = KamaeSparkPipeline.fit_estimators_fused

    def recording_fit_estimators_fused(estimators, dataset):
        fused_calls.append([e.getLayerName() for e in estimators])
        return fit_estimators_fused(estimators, dataset)

    monkeypatch.setattr(
        KamaeSparkPipeline,
        "fit_estimators_fused",
        staticmethod(recording_fit_estimators_fused),
    )
    actual = KamaeSparkPipeline(stages=stages_fn(), fuseEstimatorFits=True).fit(
        layers_dataframe
    )
    assert fused_calls == expected_layers

    # Same stages, in the same order, pointing at the same input columns.
    assert [type(s) for s in actual.stages] == [type(s) for s in expected.stages]
    assert [s.getLayerName() for s in actual.stages] == [
        s.getLayerName() for s in expected.stages
    ]
    for e, a in zip(expected.stages, actual.stages):
        assert not isinstance(a, BaseEstimator)
        if e.hasParam("inputCol") and e.isDefined("inputCol"):
            assert e.getInputCol() == a.getInputCol()
        if e.hasParam("inputCols") and e.isDefined("inputCols"):
            assert e.getInputCols() == a.getInputCols()
        if e.hasParam("labelsArray") and e.isDefined("labelsArray"):
            assert e.getLabelsArray() == a.getLabelsArray()
        if e.hasParam("mean") and e.isDefined("mean"):
            np.testing.assert_allclose(e.getMean(), a.getMean(), rtol=MOMENT_RTOL)
            np.testing.assert_allclose(e.getStddev(), a.getStddev(), rtol=MOMENT_RTOL)

    # Same transformed output.
    out_cols = [s.getOutputCol() for s in expected.stages]
    order = ["a", "c", "rel", "s", "n"]
    expected_rows = (
        expected.transform(layers_dataframe).select(*order, *out_cols).collect()
    )
    actual_rows = actual.transform(layers_dataframe).select(*order, *out_cols).collect()
    assert len(expected_rows) == len(actual_rows)
    for e_row, a_row in zip(
        sorted(expected_rows, key=str), sorted(actual_rows, key=str)
    ):
        assert [e_row[c] for c in order] == [a_row[c] for c in order]
        for c in out_cols:
            np.testing.assert_allclose(
                np.array(e_row[c], dtype=float),
                np.array(a_row[c], dtype=float),
                rtol=1e-6,
            )
