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

import numpy as np
import pyspark.sql.functions as F
import pytest

from kamae.spark.estimators import (
    ConditionalStandardScaleEstimator,
    ImputeEstimator,
    MinMaxScaleEstimator,
    OneHotEncodeEstimator,
    SharedOneHotEncodeEstimator,
    SharedStringIndexEstimator,
    SingleFeatureArrayStandardScaleEstimator,
    StandardScaleEstimator,
    StringIndexEstimator,
)
from kamae.spark.pipeline import KamaeSparkPipeline
from kamae.spark.transformers import LogTransformer
from kamae.spark.utils import collect_labels_array, labels_from_distinct_values

# Moments fused into one aggregation may be summed in a different order.
MOMENT_RTOL = 1e-12


@pytest.fixture(scope="module")
def fit_dataframe(spark_session):
    rng = np.random.default_rng(0)
    words = ["b", "a", "", "é", "Z", "ä", "mask", "c"]
    rows = []
    for _ in range(300):
        rows.append(
            (
                [float(x) for x in rng.normal(0, 1, 3)],
                [float(x) if rng.random() > 0.1 else None for x in rng.normal(5, 2, 4)],
                float(rng.choice([0.0, rng.normal(10, 3)])),
                [[float(x) for x in rng.normal(0, 1, 2)] for _ in range(2)],
                int(rng.integers(0, 3)),
                int(rng.integers(0, 2)),
                None if rng.random() < 0.1 else str(rng.choice(words)),
                int(rng.integers(0, 7)),
                None if rng.random() < 0.1 else str(rng.choice(words[:5])),
            )
        )
    return spark_session.createDataFrame(
        rows,
        "a array<double>, b array<double>, c double, d array<array<double>>, "
        "rel int, m int, s string, n int, t string",
    ).repartition(4)


def _scaler(name, col, **kwargs):
    return ConditionalStandardScaleEstimator(
        inputCol=col, outputCol=f"{name}_out", layerName=name, **kwargs
    )


def _fusable_estimators():
    return [
        _scaler("a_rel", "a", relevanceCol="rel"),
        _scaler("b_plain", "b"),
        _scaler("b_nanfill", "b", relevanceCol="rel", nanFillValue=0.0),
        _scaler("c_skip_rel", "c", relevanceCol="rel", skipZeros=True, epsilon=1e-6),
        _scaler("c_skip", "c", skipZeros=True),
        _scaler(
            "a_mask_rel",
            "a",
            relevanceCol="rel",
            maskCols=["m"],
            maskOperators=["eq"],
            maskValues=[1],
        ),
        StringIndexEstimator(
            inputCol="s",
            outputCol="s_idx",
            layerName="s_idx",
            stringOrderType="alphabeticalAsc",
        ),
        StringIndexEstimator(
            inputCol="s",
            outputCol="s_idx_desc_mask",
            layerName="s_idx_desc_mask",
            stringOrderType="alphabeticalDesc",
            maskToken="mask",
        ),
        StringIndexEstimator(
            inputCol="n",
            outputCol="n_idx",
            layerName="n_idx",
            inputDtype="string",
            stringOrderType="alphabeticalAsc",
        ),
        OneHotEncodeEstimator(
            inputCol="s",
            outputCol="s_ohe",
            layerName="s_ohe",
            stringOrderType="alphabeticalAsc",
            maskToken="mask",
        ),
        _scaler(
            "m_binary",
            "m",
            inputDtype="double",
            relevanceCol="rel",
            scalingFunction="binary",
        ),
        _scaler(
            "m_binary_masked",
            "m",
            inputDtype="double",
            relevanceCol="rel",
            scalingFunction="binary",
            maskCols=["n"],
            maskOperators=["gt"],
            maskValues=[2],
        ),
        StandardScaleEstimator(
            inputCol="b", outputCol="b_std", layerName="b_std", maskValue=-1.0
        ),
        MinMaxScaleEstimator(
            inputCol="a", outputCol="a_minmax", layerName="a_minmax", maskValue=0.0
        ),
        ImputeEstimator(
            inputCol="c",
            outputCol="c_mean",
            layerName="c_mean",
            imputeMethod="mean",
            maskValue=-999.0,
        ),
        ImputeEstimator(
            inputCol="c",
            outputCol="c_median",
            layerName="c_median",
            imputeMethod="median",
            maskValue=0.0,
        ),
        SharedStringIndexEstimator(
            inputCols=["s", "t"],
            outputCols=["s_shared", "t_shared"],
            layerName="st_shared",
            stringOrderType="alphabeticalAsc",
            maskToken="mask",
        ),
        SharedOneHotEncodeEstimator(
            inputCols=["s", "t"],
            outputCols=["s_shared_ohe", "t_shared_ohe"],
            layerName="st_shared_ohe",
            stringOrderType="alphabeticalDesc",
        ),
    ]


def _assert_same_fit(expected, actual):
    assert type(expected) is type(actual)
    for param in ["inputCol", "inputCols"]:
        if expected.hasParam(param) and expected.isDefined(param):
            assert expected.getOrDefault(param) == actual.getOrDefault(param)
    if expected.hasParam("labelsArray"):
        assert expected.getLabelsArray() == actual.getLabelsArray()
    for param in ["mean", "stddev", "min", "max", "imputeValue"]:
        if expected.hasParam(param) and expected.isDefined(param):
            np.testing.assert_allclose(
                np.array(expected.getOrDefault(param), dtype=float),
                np.array(actual.getOrDefault(param), dtype=float),
                rtol=MOMENT_RTOL,
            )


@pytest.mark.parametrize("string_order_type", ["alphabeticalAsc", "alphabeticalDesc"])
@pytest.mark.parametrize("mask_token", [None, "mask", ""])
@pytest.mark.parametrize("max_num_labels", [None, 1, 3, 100])
def test_labels_from_distinct_values_matches_collect_labels_array(
    fit_dataframe, string_order_type, mask_token, max_num_labels
):
    for col in ["s", "n"]:
        expected = collect_labels_array(
            dataset=fit_dataframe,
            column=F.col(col),
            column_datatype=fit_dataframe.schema[col].dataType,
            string_order_type=string_order_type,
            mask_token=mask_token,
            max_num_labels=max_num_labels,
        )
        distinct = fit_dataframe.agg(
            F.collect_set(F.col(col).cast("string")).alias("v")
        ).first()["v"]
        actual = labels_from_distinct_values(
            values=distinct,
            string_order_type=string_order_type,
            mask_token=mask_token,
            max_num_labels=max_num_labels,
        )
        assert actual == expected


def test_fit_estimators_fused_matches_fit(fit_dataframe):
    expected = [e.fit(fit_dataframe) for e in _fusable_estimators()]
    estimators = _fusable_estimators()
    assert all(e.can_fit_by_aggregation(fit_dataframe) for e in estimators)
    actual = KamaeSparkPipeline.fit_estimators_fused(estimators, fit_dataframe)
    assert len(actual) == len(expected)
    for e, a in zip(expected, actual):
        _assert_same_fit(e, a)

    # Estimators are left pointing at their original input columns.
    def input_columns(e):
        if e.hasParam("inputCols") and e.isDefined("inputCols"):
            return e.getInputCols()
        return e.getInputCol()

    assert [input_columns(e) for e in estimators] == [
        input_columns(e) for e in _fusable_estimators()
    ]


def test_fit_estimators_fused_falls_back_for_unsupported(fit_dataframe):
    estimators = [
        _scaler(
            "m_binary",
            "m",
            inputDtype="double",
            relevanceCol="rel",
            scalingFunction="binary",
        ),
        StringIndexEstimator(
            inputCol="s",
            outputCol="s_freq",
            layerName="s_freq",
            stringOrderType="frequencyDesc",
        ),
        _scaler("a_rel", "a", relevanceCol="rel"),
        # These change the rows (explode) before aggregating, so they fit normally.
        _scaler("d_nested", "d", relevanceCol="rel"),
        StandardScaleEstimator(inputCol="d", outputCol="d_std", layerName="d_std"),
        MinMaxScaleEstimator(inputCol="d", outputCol="d_minmax", layerName="d_minmax"),
        SingleFeatureArrayStandardScaleEstimator(
            inputCol="a", outputCol="a_pooled", layerName="a_pooled"
        ),
        ImputeEstimator(
            inputCol="a",
            outputCol="a_impute",
            layerName="a_impute",
            imputeMethod="mean",
            maskValue=-999.0,
        ),
        # maxNumLabels: the distributed label fit only brings the first labels to
        # the driver, collect_set would gather every distinct label first.
        StringIndexEstimator(
            inputCol="s",
            outputCol="s_idx_max",
            layerName="s_idx_max",
            stringOrderType="alphabeticalDesc",
            maskToken="mask",
            maxNumLabels=4,
        ),
        SharedOneHotEncodeEstimator(
            inputCols=["s", "t"],
            outputCols=["s_shared_max", "t_shared_max"],
            layerName="st_shared_max",
            stringOrderType="alphabeticalDesc",
            maxNumLabels=3,
        ),
    ]
    assert [e.can_fit_by_aggregation(fit_dataframe) for e in estimators] == [
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    actual = KamaeSparkPipeline.fit_estimators_fused(estimators, fit_dataframe)
    for estimator, transformer in zip(estimators, actual):
        _assert_same_fit(estimator.fit(fit_dataframe), transformer)


def _layered_stages():
    # Layer 0: scaler on log(c) and an index of s. Layer 1: a scaler on the index.
    return [
        LogTransformer(inputCol="c", outputCol="c_log", layerName="c_log", alpha=1.0),
        _scaler("c_log_scaled", "c_log", relevanceCol="rel"),
        StringIndexEstimator(
            inputCol="s",
            outputCol="s_idx",
            layerName="s_idx",
            stringOrderType="alphabeticalAsc",
        ),
        _scaler("s_idx_scaled", "s_idx", inputDtype="double"),
        StringIndexEstimator(
            inputCol="s",
            outputCol="s_freq",
            layerName="s_freq",
            stringOrderType="frequencyDesc",
        ),
        _scaler("a_scaled", "a", relevanceCol="rel", skipZeros=True),
    ]


def test_estimator_layers():
    layers = KamaeSparkPipeline.estimator_layers(_layered_stages())
    assert [[e.getLayerName() for e in layer] for layer in layers] == [
        ["c_log_scaled", "s_idx", "s_freq", "a_scaled"],
        ["s_idx_scaled"],
    ]


def test_fused_pipeline_fit_matches_pipeline_fit(fit_dataframe):
    expected = KamaeSparkPipeline(stages=_layered_stages()).fit(fit_dataframe)
    actual = KamaeSparkPipeline(stages=_layered_stages(), fuseEstimatorFits=True).fit(
        fit_dataframe
    )
    assert [s.getLayerName() for s in actual.stages] == [
        s.getLayerName() for s in expected.stages
    ]
    for e, a in zip(expected.stages, actual.stages):
        if isinstance(e, LogTransformer):
            assert isinstance(a, LogTransformer)
            assert (e.getInputCol(), e.getOutputCol(), e.getAlpha()) == (
                a.getInputCol(),
                a.getOutputCol(),
                a.getAlpha(),
            )
        else:
            _assert_same_fit(e, a)
    out_cols = ["c_log_scaled_out", "s_idx_scaled_out", "s_freq", "a_scaled_out"]
    in_cols = ["a", "b", "c", "rel", "m", "s", "n", "t"]

    def sorted_rows(model):
        rows = model.transform(fit_dataframe).select(*in_cols, *out_cols).collect()
        return sorted(rows, key=lambda r: str([r[c] for c in in_cols]))

    expected_rows = sorted_rows(expected)
    actual_rows = sorted_rows(actual)
    assert len(expected_rows) == len(actual_rows)
    for e_row, a_row in zip(expected_rows, actual_rows):
        assert [e_row[c] for c in in_cols] == [a_row[c] for c in in_cols]
        for c in out_cols:
            np.testing.assert_allclose(
                np.array(e_row[c], dtype=float),
                np.array(a_row[c], dtype=float),
                rtol=1e-6,
            )


def test_fuse_estimator_fits_ignored_with_fit_sample_fraction(fit_dataframe):
    with pytest.warns(UserWarning, match="fuseEstimatorFits is ignored"):
        KamaeSparkPipeline(
            stages=_layered_stages()[:2],
            fuseEstimatorFits=True,
            fitSampleFraction=0.5,
        ).fit(fit_dataframe)


def test_nested_array_scaler_does_not_change_rows_of_other_estimators(fit_dataframe):
    # The helper column of a nested-array scaler explodes rows; fused with it, a
    # binary scaler (whose stddev depends on the row count) would be wrong.
    def estimators():
        return [
            _scaler(
                "m_binary",
                "m",
                inputDtype="double",
                relevanceCol="rel",
                scalingFunction="binary",
            ),
            _scaler("d_nested", "d", relevanceCol="rel"),
        ]

    expected = [e.fit(fit_dataframe) for e in estimators()]
    actual = KamaeSparkPipeline.fit_estimators_fused(estimators(), fit_dataframe)
    for e, a in zip(expected, actual):
        _assert_same_fit(e, a)


def test_estimator_whose_aggregations_add_rows_is_fitted_on_its_own(fit_dataframe):
    class NestedFusingStandardScaleEstimator(StandardScaleEstimator):
        """Wrongly claims it fits by aggregation for nested arrays."""

        def can_fit_by_aggregation(self, dataset):
            return True

    def estimators():
        return [
            _scaler(
                "m_binary",
                "m",
                inputDtype="double",
                relevanceCol="rel",
                scalingFunction="binary",
            ),
            NestedFusingStandardScaleEstimator(
                inputCol="d", outputCol="d_std", layerName="d_std"
            ),
        ]

    expected = [e.fit(fit_dataframe) for e in estimators()]
    with pytest.warns(UserWarning, match="row generator"):
        actual = KamaeSparkPipeline.fit_estimators_fused(estimators(), fit_dataframe)
    for e, a in zip(expected, actual):
        _assert_same_fit(e, a)


def test_label_estimators_fit_normally_with_distributed_collection(
    fit_dataframe, monkeypatch
):
    # The normal fit keeps collect_labels_array (distributed distinct + sort);
    # only the fused fit collects labels with F.collect_set.
    import kamae.spark.estimators.string_index as string_index_module

    calls = []
    original = string_index_module.collect_labels_array

    def counting_collect_labels_array(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        string_index_module, "collect_labels_array", counting_collect_labels_array
    )

    def estimator():
        return StringIndexEstimator(
            inputCol="s",
            outputCol="s_idx",
            layerName="s_idx",
            stringOrderType="alphabeticalAsc",
        )

    expected = estimator().fit(fit_dataframe)
    assert len(calls) == 1
    actual = KamaeSparkPipeline.fit_estimators_fused([estimator()], fit_dataframe)
    assert len(calls) == 1
    assert actual[0].getLabelsArray() == expected.getLabelsArray()
