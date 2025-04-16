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

from kamae.spark.transformers import LogTransformer


class TestLog:
    @pytest.fixture(scope="class")
    def log_transform_col1_alpha_1_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 0.6931471805599453),
                (4, 2, 6, "b", "c", [4, 2, 6], 1.6094379124341003),
                (7, 8, 3, "a", "a", [7, 8, 3], 2.0794415416798357),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "log_col1"],
        )

    @pytest.fixture(scope="class")
    def log_transform_col2_alpha_5_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 1.9459101490553132),
                (4, 2, 6, "b", "c", [4, 2, 6], 1.9459101490553132),
                (7, 8, 3, "a", "a", [7, 8, 3], 2.5649493574615367),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "log_col2"],
        )

    @pytest.fixture(scope="class")
    def log_transform_alpha_10_expected_array(self, spark_session):
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
                        [2.3978952727983707, 2.0794415416798357, 2.5649493574615367],
                        [2.3978952727983707, 2.4849066497880004, 2.5649493574615367],
                        [2.3978952727983707, 2.4849066497880004, 1.9459101490553132],
                        [2.6390573296152584, 2.4849066497880004, 1.3862943611198906],
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
                        [2.6390573296152584, 2.0794415416798357, 2.772588722239781],
                        [2.6390573296152584, 2.0794415416798357, 2.772588722239781],
                        [2.6390573296152584, 2.4849066497880004, 1.3862943611198906],
                        [2.833213344056216, 2.8903717578961645, 2.5649493574615367],
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
                        [2.833213344056216, 2.8903717578961645, 2.5649493574615367],
                        [2.833213344056216, 0.6931471805599453, 2.5649493574615367],
                        [2.833213344056216, 2.8903717578961645, 1.9459101490553132],
                        [2.1972245773362196, 2.4849066497880004, 1.9459101490553132],
                    ],
                ),
            ],
            ["col1", "col2", "log_col1"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, alpha, expected_dataframe",
        [
            (
                "example_dataframe",
                "col1",
                "log_col1",
                1,
                "log_transform_col1_alpha_1_expected",
            ),
            (
                "example_dataframe",
                "col2",
                "log_col2",
                5,
                "log_transform_col2_alpha_5_expected",
            ),
            (
                "example_dataframe_w_nested_arrays",
                "col1",
                "log_col1",
                10,
                "log_transform_alpha_10_expected_array",
            ),
        ],
    )
    def test_spark_log_transform(
        self,
        input_dataframe,
        input_col,
        output_col,
        alpha,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = LogTransformer(
            inputCol=input_col,
            outputCol=output_col,
            alpha=alpha,
        )
        actual = transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_log_transform_defaults(self):
        # when
        log_transform = LogTransformer()
        # then
        assert log_transform.getAlpha() == 0.0
        assert log_transform.getLayerName() == log_transform.uid
        assert log_transform.getOutputCol() == f"{log_transform.uid}__output"

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, alpha",
        [
            (tf.constant(["1.0", "4.0", "7.0", "8.0"]), "double", "float", 1),
            (tf.constant([2.0, 5.0, 1.0]), "float", "int", 2),
            (tf.constant([-1.0, 7.0]), None, None, 3),
            (tf.constant([0.0, 6.0, 3.0]), "float", "double", 4),
            (tf.constant([2.0, 5.0, 1.0, 5.0, 2.5]), None, "double", 10),
        ],
    )
    def test_log_transform_spark_tf_parity(
        self, spark_session, input_tensor, input_dtype, output_dtype, alpha
    ):
        # given
        transformer = LogTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            alpha=alpha,
        )
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
        tensorflow_values = [
            v.decode("utf-8") if isinstance(v, bytes) else v
            for v in transformer.get_tf_layer()(input_tensor).numpy().tolist()
        ]

        # then
        if isinstance(spark_values[0], str):
            np.testing.assert_equal(
                spark_values,
                tensorflow_values,
                err_msg="Spark and Tensorflow transform outputs are not equal",
            )
        else:
            np.testing.assert_almost_equal(
                spark_values,
                tensorflow_values,
                decimal=6,
                err_msg="Spark and Tensorflow transform outputs are not equal",
            )
