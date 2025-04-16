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

from kamae.spark.transformers import ExpTransformer


class TestExp:
    @pytest.fixture(scope="class")
    def exp_transform_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 2.7182818284590455),
                (4, 2, 6, "b", "c", [4, 2, 6], 54.598150033144236),
                (7, 8, 3, "a", "a", [7, 8, 3], 1096.6331584284585),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "exp_col1"],
        )

    @pytest.fixture(scope="class")
    def exp_trans_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 7.3890560989306504),
                (4, 2, 6, "b", "c", [4, 2, 6], 7.3890560989306504),
                (7, 8, 3, "a", "a", [7, 8, 3], 2980.9579870417283),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "exp_col2"],
        )

    @pytest.fixture(scope="class")
    def exp_transform_expected_array(self, spark_session):
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
                        [2.7182818284590455, 0.1353352832366127, 20.085536923187668],
                        [2.7182818284590455, 7.3890560989306504, 20.085536923187668],
                        [2.7182818284590455, 7.3890560989306504, 0.049787068367863944],
                        [54.598150033144236, 7.3890560989306504, 0.0024787521766663585],
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
                        [54.598150033144236, 0.1353352832366127, 403.4287934927351],
                        [54.598150033144236, 0.1353352832366127, 403.4287934927351],
                        [54.598150033144236, 7.3890560989306504, 0.0024787521766663585],
                        [1096.6331584284585, 2980.9579870417283, 20.085536923187668],
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
                        [1096.6331584284585, 2980.9579870417283, 20.085536923187668],
                        [
                            1096.6331584284585,
                            0.00033546262790251185,
                            20.085536923187668,
                        ],
                        [1096.6331584284585, 2980.9579870417283, 0.049787068367863944],
                        [0.36787944117144233, 7.3890560989306504, 0.049787068367863944],
                    ],
                ),
            ],
            ["col1", "col2", "exp_col1"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, expected_dataframe",
        [
            ("example_dataframe", "col1", "exp_col1", "exp_transform_expected_1"),
            ("example_dataframe", "col2", "exp_col2", "exp_trans_expected_2"),
            (
                "example_dataframe_w_nested_arrays",
                "col1",
                "exp_col1",
                "exp_transform_expected_array",
            ),
        ],
    )
    def test_spark_exp_transform(
        self,
        input_dataframe,
        input_col,
        output_col,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = ExpTransformer(
            inputCol=input_col,
            outputCol=output_col,
        )
        actual = transformer.transform(input_dataframe)

        # then
        diff = actual.exceptAll(expected)

        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_exp_transform_defaults(self):
        # when
        exp_transformer = ExpTransformer()
        # then
        assert exp_transformer.getLayerName() == exp_transformer.uid
        assert exp_transformer.getOutputCol() == f"{exp_transformer.uid}__output"

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype",
        [
            (tf.constant([1.0, 2.0, 3.0]), None, None),
            (
                tf.constant(["1.0", "-5.0", "-6.0", "7.0", "-8.0", "9.0"]),
                "double",
                None,
            ),
            (tf.constant([-4.0, -5.0, -3.0, -47.0, -8.2, -11.0]), None, "double"),
        ],
    )
    def test_exp_spark_tf_parity(
        self, spark_session, input_tensor, input_dtype, output_dtype
    ):
        # given
        transformer = ExpTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
        )
        # when
        if input_tensor.dtype.name == "string":
            spark_df = spark_session.createDataFrame(
                [(v.decode("utf-8"),) for v in input_tensor.numpy().tolist()], ["input"]
            )
        else:
            spark_df = spark_session.createDataFrame(
                [(v,) for v in input_tensor.numpy().tolist()], ["input"]
            )
        spark_values = (
            transformer.transform(spark_df)
            .select("output")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        tensorflow_values = transformer.get_tf_layer()(input_tensor).numpy().tolist()

        # then
        np.testing.assert_almost_equal(
            spark_values,
            tensorflow_values,
            decimal=3,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
