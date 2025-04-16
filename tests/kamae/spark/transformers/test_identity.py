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

from kamae.spark.transformers import IdentityTransformer


class TestIdentity:
    @pytest.fixture(scope="class")
    def identity_transform_col1_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 1),
                (4, 2, 6, "b", "c", [4, 2, 6], 4),
                (7, 8, 3, "a", "a", [7, 8, 3], 7),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "iden_col1"],
        )

    @pytest.fixture(scope="class")
    def identity_transform_col2_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 2),
                (4, 2, 6, "b", "c", [4, 2, 6], 2),
                (7, 8, 3, "a", "a", [7, 8, 3], 8),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "iden_col2"],
        )

    @pytest.fixture(scope="class")
    def identity_transform_col2_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 2),
                (4, 2, 6, "b", "c", [4, 2, 6], 2),
                (7, 8, 3, "a", "a", [7, 8, 3], 8),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "iden_col2"],
        )

    @pytest.fixture(scope="class")
    def identity_transform_col3_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], 3),
                (4, 2, 6, "b", "c", [4, 2, 6], 6),
                (7, 8, 3, "a", "a", [7, 8, 3], 3),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "iden_col3"],
        )

    @pytest.fixture(scope="class")
    def identity_transform_col4_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], "a"),
                (4, 2, 6, "b", "c", [4, 2, 6], "b"),
                (7, 8, 3, "a", "a", [7, 8, 3], "a"),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "iden_col4"],
        )

    @pytest.fixture(scope="class")
    def identity_transform_col5_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], "c"),
                (4, 2, 6, "b", "c", [4, 2, 6], "c"),
                (7, 8, 3, "a", "a", [7, 8, 3], "a"),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col1_col2_col3", "iden_col5"],
        )

    @pytest.mark.parametrize(
        "input_col, output_col, expected_dataframe",
        [
            ("col1", "iden_col1", "identity_transform_col1_expected"),
            ("col2", "iden_col2", "identity_transform_col2_expected"),
            ("col3", "iden_col3", "identity_transform_col3_expected"),
            ("col4", "iden_col4", "identity_transform_col4_expected"),
            ("col5", "iden_col5", "identity_transform_col5_expected"),
        ],
    )
    def test_spark_identity_transform(
        self,
        example_dataframe,
        input_col,
        output_col,
        expected_dataframe,
        request,
    ):
        # given
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = IdentityTransformer(
            inputCol=input_col,
            outputCol=output_col,
        )
        actual = transformer.transform(example_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_identity_transform_defaults(self):
        # when
        transformer = IdentityTransformer()
        # then
        assert transformer.getLayerName() == transformer.uid
        assert transformer.getOutputCol() == f"{transformer.uid}__output"

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype",
        [
            (tf.constant(["1.0", "4.0", "7.0", "8.0"]), "double", "string"),
            (tf.constant([2.0, 5.0, 1.0]), "int", "bigint"),
            (tf.constant([-1.0, 7.0]), "double", "smallint"),
            (tf.constant([0.0, 6.0, 3.0]), "float", None),
            (tf.constant([2.0, 5.0, 1.0, 5.0, 2.5]), None, None),
        ],
    )
    def test_identity_transform_spark_tf_parity(
        self, spark_session, input_tensor, input_dtype, output_dtype
    ):
        # given
        transformer = IdentityTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
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
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
