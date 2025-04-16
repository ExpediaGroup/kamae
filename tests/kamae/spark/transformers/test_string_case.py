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
import pytest
import tensorflow as tf

from kamae.spark.transformers import StringCaseTransformer


class TestStringCase:
    @pytest.fixture(scope="class")
    def example_dataframe_with_string_arrays_to_case(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        ["BasE", "Race", "Mace"],
                        ["TASTe", "cAsE", "Pace"],
                        ["g", "h", "HasTe"],
                        ["jeeeE", "keeeE", "meeeE"],
                    ],
                    [
                        ["heLLo", "woRLd"],
                        ["adVenture", "timE"],
                        ["begIns", "heRe"],
                        ["theRe", "theRe"],
                    ],
                )
            ],
            ["col1", "col2"],
        )

    @pytest.fixture(scope="class")
    def example_dataframe_with_strings_to_case(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("Hello World", "en_US", ["en_US", "en_US", "en_US"]),
                ("ADVENTURE Time", "en_CA", ["en_CA", "ch_CH", "es_US"]),
                ("Begins", "es_US", ["es_BR", "ch_CH", "es_US"]),
            ],
            ["col1", "col2", "col3"],
        )

    @pytest.fixture(scope="class")
    def string_case_transform_col1_upper_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("Hello World", "en_US", ["en_US", "en_US", "en_US"], "HELLO WORLD"),
                (
                    "ADVENTURE Time",
                    "en_CA",
                    ["en_CA", "ch_CH", "es_US"],
                    "ADVENTURE TIME",
                ),
                ("Begins", "es_US", ["es_BR", "ch_CH", "es_US"], "BEGINS"),
            ],
            ["col1", "col2", "col3", "upper_col1"],
        )

    @pytest.fixture(scope="class")
    def string_case_transform_col1_lower_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("Hello World", "en_US", ["en_US", "en_US", "en_US"], "hello world"),
                (
                    "ADVENTURE Time",
                    "en_CA",
                    ["en_CA", "ch_CH", "es_US"],
                    "adventure time",
                ),
                ("Begins", "es_US", ["es_BR", "ch_CH", "es_US"], "begins"),
            ],
            ["col1", "col2", "col3", "lower_col1"],
        )

    @pytest.fixture(scope="class")
    def string_case_transform_col2_lower_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("Hello World", "en_US", ["en_US", "en_US", "en_US"], "en_us"),
                ("ADVENTURE Time", "en_CA", ["en_CA", "ch_CH", "es_US"], "en_ca"),
                ("Begins", "es_US", ["es_BR", "ch_CH", "es_US"], "es_us"),
            ],
            ["col1", "col2", "col3", "lower_col2"],
        )

    @pytest.fixture(scope="class")
    def string_case_transform_col3_upper_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    "Hello World",
                    "en_US",
                    ["en_US", "en_US", "en_US"],
                    ["EN_US", "EN_US", "EN_US"],
                ),
                (
                    "ADVENTURE Time",
                    "en_CA",
                    ["en_CA", "ch_CH", "es_US"],
                    ["EN_CA", "CH_CH", "ES_US"],
                ),
                (
                    "Begins",
                    "es_US",
                    ["es_BR", "ch_CH", "es_US"],
                    ["ES_BR", "CH_CH", "ES_US"],
                ),
            ],
            ["col1", "col2", "col3", "upper_col3"],
        )

    @pytest.fixture(scope="class")
    def string_case_array_transform_col1_upper_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        ["BasE", "Race", "Mace"],
                        ["TASTe", "cAsE", "Pace"],
                        ["g", "h", "HasTe"],
                        ["jeeeE", "keeeE", "meeeE"],
                    ],
                    [
                        ["heLLo", "woRLd"],
                        ["adVenture", "timE"],
                        ["begIns", "heRe"],
                        ["theRe", "theRe"],
                    ],
                    [
                        ["BASE", "RACE", "MACE"],
                        ["TASTE", "CASE", "PACE"],
                        ["G", "H", "HASTE"],
                        ["JEEEE", "KEEEE", "MEEEE"],
                    ],
                )
            ],
            ["col1", "col2", "upper_col1"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, string_case_type, expected_dataframe",
        [
            (
                "example_dataframe_with_strings_to_case",
                "col1",
                "upper_col1",
                "upper",
                "string_case_transform_col1_upper_expected",
            ),
            (
                "example_dataframe_with_strings_to_case",
                "col1",
                "lower_col1",
                "lower",
                "string_case_transform_col1_lower_expected",
            ),
            (
                "example_dataframe_with_strings_to_case",
                "col2",
                "lower_col2",
                "lower",
                "string_case_transform_col2_lower_expected",
            ),
            (
                "example_dataframe_with_strings_to_case",
                "col3",
                "upper_col3",
                "upper",
                "string_case_transform_col3_upper_expected",
            ),
            (
                "example_dataframe_with_string_arrays_to_case",
                "col1",
                "upper_col1",
                "upper",
                "string_case_array_transform_col1_upper_expected",
            ),
        ],
    )
    def test_spark_string_case_transform(
        self,
        input_dataframe,
        input_col,
        output_col,
        string_case_type,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = StringCaseTransformer(
            inputCol=input_col,
            outputCol=output_col,
            stringCaseType=string_case_type,
        )
        actual = transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_string_case_transform_defaults(self):
        # when
        string_case_transform = StringCaseTransformer()
        # then
        assert string_case_transform.getLayerName() == string_case_transform.uid
        assert string_case_transform.getStringCaseType() == "lower"
        assert (
            string_case_transform.getOutputCol()
            == f"{string_case_transform.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, string_case_type",
        [
            (
                tf.constant(["HElLO wOrLd", "adVeNtuRE TIme", "BeGiNs"]),
                None,
                "string",
                "lower",
            ),
            (
                tf.constant(["hELlo WoRlD", "AdVenturE timE", "bEGIns"]),
                None,
                None,
                "upper",
            ),
            (
                tf.constant([True, False, False]),
                "string",
                "boolean",
                "upper",
            ),
        ],
    )
    def test_string_case_transform_spark_tf_parity(
        self, spark_session, input_tensor, input_dtype, output_dtype, string_case_type
    ):
        # given
        transformer = StringCaseTransformer(
            inputCol="input",
            outputCol="output",
            stringCaseType=string_case_type,
            inputDtype=input_dtype,
            outputDtype=output_dtype,
        )
        # when
        spark_df = spark_session.createDataFrame(
            [
                (v.decode("utf-8") if isinstance(v, bytes) else v,)
                for v in input_tensor.numpy().tolist()
            ],
            ["input"],
        )
        spark_values = (
            transformer.transform(spark_df)
            .select("output")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        tensorflow_values = [
            v.decode("utf-8") if isinstance(v, bytes) else v
            for v in transformer.get_tf_layer()(input_tensor).numpy().tolist()
        ]

        # then
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
