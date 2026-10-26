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

"""Transformers must give the same results whether or not ANSI mode is enabled, since
it is enabled by default from Spark 4 and raises on division by zero and out of
bounds array access, where non-ANSI mode returns null."""

import pytest

from kamae.spark.transformers import (
    CosineSimilarityTransformer,
    DivideTransformer,
    ListMedianTransformer,
    ModuloTransformer,
    PairwiseCosineSimilarityTransformer,
    SubStringDelimAtIndexTransformer,
)


@pytest.fixture(params=["true", "false"], ids=["ansi", "non_ansi"])
def ansi_spark_session(request, spark_session):
    original = spark_session.conf.get("spark.sql.ansi.enabled")
    spark_session.conf.set("spark.sql.ansi.enabled", request.param)
    yield spark_session
    spark_session.conf.set("spark.sql.ansi.enabled", original)


class TestAnsiMode:
    def test_divide_by_zero(self, ansi_spark_session):
        df = ansi_spark_session.createDataFrame(
            [(1.0, 0.0), (0.0, 0.0), (6.0, 2.0)], ["col1", "col2"]
        )
        transformer = DivideTransformer(inputCols=["col1", "col2"], outputCol="out")
        actual = [r.out for r in transformer.transform(df).collect()]
        assert actual == [0.0, 0.0, 3.0]

    def test_modulo_by_zero(self, ansi_spark_session):
        df = ansi_spark_session.createDataFrame(
            [(10, 0), (-7, 3), (7, 3)], ["col1", "col2"]
        )
        transformer = ModuloTransformer(inputCols=["col1", "col2"], outputCol="out")
        actual = [r.out for r in transformer.transform(df).collect()]
        assert actual == [None, 2, 1]

    def test_cosine_similarity_zero_norm(self, ansi_spark_session):
        df = ansi_spark_session.createDataFrame(
            [([0.0, 0.0], [1.0, 2.0]), ([1.0, 0.0], [2.0, 0.0])], ["col1", "col2"]
        )
        transformer = CosineSimilarityTransformer(
            inputCols=["col1", "col2"], outputCol="out"
        )
        actual = [r.out for r in transformer.transform(df).collect()]
        assert actual == [0.0, 1.0]

    def test_pairwise_cosine_similarity_zero_norm(self, ansi_spark_session):
        df = ansi_spark_session.createDataFrame(
            [([1.0, 0.0], [0.0, 0.0, 2.0, 0.0])], ["query", "candidates"]
        )
        transformer = PairwiseCosineSimilarityTransformer(
            inputCols=["query", "candidates"], outputCol="out", embeddingDim=2
        )
        actual = transformer.transform(df).collect()[0].out
        assert actual == [0.0, 1.0]

    @pytest.mark.parametrize(
        "index, expected",
        [(1, ["b", "default"]), (-1, ["c", "a"]), (5, ["default", "default"])],
    )
    def test_sub_string_delim_at_index_out_of_bounds(
        self, ansi_spark_session, index, expected
    ):
        df = ansi_spark_session.createDataFrame([("a_b_c",), ("a",)], ["col1"])
        transformer = SubStringDelimAtIndexTransformer(
            inputCol="col1",
            outputCol="out",
            delimiter="_",
            index=index,
            defaultValue="default",
        )
        actual = [r.out for r in transformer.transform(df).collect()]
        assert actual == expected

    def test_list_median_all_values_filtered(self, ansi_spark_session):
        df = ansi_spark_session.createDataFrame(
            [(1, 5.0), (1, 3.0), (2, -1.0), (2, -1.0)], ["search_id", "value"]
        )
        transformer = ListMedianTransformer(
            inputCol="value",
            outputCol="out",
            queryIdCol="search_id",
            minFilterValue=0.0,
            nanFillValue=-99.0,
        )
        actual = {(r.search_id, r.out) for r in transformer.transform(df).collect()}
        assert actual == {(1, 4.0), (2, -99.0)}
