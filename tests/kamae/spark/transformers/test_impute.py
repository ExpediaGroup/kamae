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

from kamae.spark.estimators import ImputeEstimator
from kamae.spark.transformers import ImputeTransformer


class TestImpute:
    @pytest.fixture(scope="class")
    def impute_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.0, 2.0, 3.0, "a", "c", [1.0, 2.0, 3.0], "hello"),
                (4.0, 2.0, 6.0, "b", "c", [4.0, 2.0, 6.0], "b"),
                (7.0, 8.0, 3.0, "a", "a", [7.0, 8.0, 3.0], "hello"),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "col4_str_imputed",
            ],
        )

    @pytest.fixture(scope="class")
    def impute_nested_arrays_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        [1.0, -2.0, 3.0],
                        [1.0, 2.0, 3.0],
                        [1.0, 2.0, -3.0],
                        [4.0, 2.0, -6.0],
                    ],
                    [
                        [["a", "b"], ["c", "d"]],
                        [["a", "b"], ["c", "d"]],
                        [["a", "b"], ["c", "d"]],
                        [["a", "b"], ["c", "d"]],
                    ],
                    [
                        [1.0, -2.0, 3.0],
                        [1.0, 100.0, 3.0],
                        [1.0, 100.0, -3.0],
                        [4.0, 100.0, -6.0],
                    ],
                ),
                (
                    [
                        [4.0, -2.0, 6.0],
                        [4.0, -2.0, 6.0],
                        [4.0, 2.0, -6.0],
                        [7.0, 8.0, 3.0],
                    ],
                    [
                        [["e", "f"], ["g", "h"]],
                        [["e", "f"], ["g", "h"]],
                        [["e", "f"], ["g", "h"]],
                        [["e", "f"], ["g", "h"]],
                    ],
                    [
                        [4.0, -2.0, 6.0],
                        [4.0, -2.0, 6.0],
                        [4.0, 100.0, -6.0],
                        [7.0, 8.0, 3.0],
                    ],
                ),
                (
                    [
                        [7.0, 8.0, 3.0],
                        [7.0, -8.0, 3.0],
                        [7.0, 8.0, -3.0],
                        [-1.0, 2.0, -3.0],
                    ],
                    [
                        [["i", "j"], ["k", "l"]],
                        [["i", "j"], ["k", "l"]],
                        [["i", "j"], ["k", "l"]],
                        [["i", "j"], ["k", "l"]],
                    ],
                    [
                        [7.0, 8.0, 3.0],
                        [7.0, -8.0, 3.0],
                        [7.0, 8.0, -3.0],
                        [-1.0, 100.0, -3.0],
                    ],
                ),
            ],
            ["col1", "col2", "col1_mean_imputed"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, imputeValue, maskValue, expected_dataframe",
        [
            (
                "example_dataframe",
                "col4",
                "col4_str_imputed",
                "hello",
                "a",
                "impute_expected",
            ),
            (
                "example_dataframe_w_nested_arrays",
                "col1",
                "col1_mean_imputed",
                100.0,
                2.0,
                "impute_nested_arrays_expected",
            ),
        ],
    )
    def test_spark_impute_transform(
        self,
        input_dataframe,
        input_col,
        output_col,
        maskValue,
        imputeValue,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        impute_model = ImputeTransformer(
            inputCol=input_col,
            outputCol=output_col,
            maskValue=maskValue,
            imputeValue=imputeValue,
        )
        actual = impute_model.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, maskValue, imputeValue",
        [
            (
                tf.constant([-999.0, 6.0, 9.0, 100.0]),
                None,
                None,
                -999.0,
                300.0,
            ),
            (
                tf.constant(["hello", "cruel", "world"]),
                None,
                None,
                "goodbye",
                "hello",
            ),
        ],
    )
    def test_impute_spark_tf_parity(
        self,
        spark_session,
        input_tensor,
        input_dtype,
        output_dtype,
        maskValue,
        imputeValue,
    ):
        # given
        transformer = ImputeTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            maskValue=maskValue,
            imputeValue=imputeValue,
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
