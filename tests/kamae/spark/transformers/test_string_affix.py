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

from kamae.spark.transformers import StringAffixTransformer


class TestStringAffix:
    @pytest.fixture(scope="class")
    def affix_transform_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.0, 2.0, 3.0, "a", "c", [1.0, 2.0, 3.0], "a"),
                (4.0, 2.0, 6.0, "b", "c", [4.0, 2.0, 6.0], "b"),
                (7.0, 8.0, 3.0, "a", "a", [7.0, 8.0, 3.0], "a"),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "col4_no_affix",
            ],
        )

    @pytest.fixture(scope="class")
    def affix_transform_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.0, 2.0, 3.0, "a", "c", [1.0, 2.0, 3.0], ">c<"),
                (4.0, 2.0, 6.0, "b", "c", [4.0, 2.0, 6.0], ">c<"),
                (7.0, 8.0, 3.0, "a", "a", [7.0, 8.0, 3.0], ">a<"),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                ">col5<",
            ],
        )

    @pytest.fixture(scope="class")
    def affix_transform_expected_4(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.0, 2.0, 3.0, "a", "c", [1.0, 2.0, 3.0], "www.a"),
                (4.0, 2.0, 6.0, "b", "c", [4.0, 2.0, 6.0], "www.b"),
                (7.0, 8.0, 3.0, "a", "a", [7.0, 8.0, 3.0], "www.a"),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "www_col4",
            ],
        )

    @pytest.fixture(scope="class")
    def affix_transform_expected_5(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1.0, 2.0, 3.0, "a", "c", [1.0, 2.0, 3.0], "c.com"),
                (4.0, 2.0, 6.0, "b", "c", [4.0, 2.0, 6.0], "c.com"),
                (7.0, 8.0, 3.0, "a", "a", [7.0, 8.0, 3.0], "a.com"),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "col5_com",
            ],
        )

    @pytest.fixture(scope="class")
    def string_affix_col1_array_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    [
                        ["a", "b", "c"],
                        ["d", "e", "f"],
                        ["g", "h", "i"],
                        ["j", "k", "l"],
                    ],
                    [
                        ["m", "n", "o"],
                        ["p", "q", "r"],
                        ["s", "t", "u"],
                        ["v", "w", "x"],
                    ],
                    [
                        ["pre_a_post", "pre_b_post", "pre_c_post"],
                        ["pre_d_post", "pre_e_post", "pre_f_post"],
                        ["pre_g_post", "pre_h_post", "pre_i_post"],
                        ["pre_j_post", "pre_k_post", "pre_l_post"],
                    ],
                )
            ],
            ["col1", "col2", "string_affix_col1"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, prefix, suffix, expected_dataframe",
        [
            (
                "example_dataframe",
                "col5",
                ">col5<",
                ">",
                "<",
                "affix_transform_expected_2",
            ),
            (
                "example_dataframe",
                "col4",
                "www_col4",
                "www.",
                None,
                "affix_transform_expected_4",
            ),
            # Should be equivalent to above
            (
                "example_dataframe",
                "col5",
                "col5_com",
                None,
                ".com",
                "affix_transform_expected_5",
            ),
            (
                "example_dataframe_w_multiple_string_nested_arrays",
                "col1",
                "string_affix_col1",
                "pre_",
                "_post",
                "string_affix_col1_array_expected",
            ),
        ],
    )
    def test_spark_string_affix_transform(
        self,
        input_dataframe,
        input_col,
        output_col,
        prefix,
        suffix,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = StringAffixTransformer(
            outputCol=output_col,
        )
        transformer = transformer.setInputCol(input_col)
        if prefix is not None:
            transformer = transformer.setPrefix(prefix)
        if suffix is not None:
            transformer = transformer.setSuffix(suffix)
        actual = transformer.transform(input_dataframe)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_string_affix_transform_defaults(self):
        # when
        string_affix_transformer = StringAffixTransformer()
        # then
        assert string_affix_transformer.getLayerName() == string_affix_transformer.uid
        assert (
            string_affix_transformer.getOutputCol()
            == f"{string_affix_transformer.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, prefix, suffix",
        [
            (tf.constant(["a", "b"]), None, None, None, "<"),
            (tf.constant(["hello", "world"]), "string", None, ">", None),
            (tf.constant([1, 2]), "string", None, ">", None),
            (
                tf.constant(["123", "A", "one"]),
                "string",
                "string",
                ">",
                "<",
            ),
        ],
    )
    def test_string_affix_transform_spark_tf_parity(
        self, spark_session, input_tensor, input_dtype, output_dtype, prefix, suffix
    ):
        # given
        transformer = StringAffixTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            prefix=prefix,
            suffix=suffix,
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

    @pytest.mark.parametrize(
        "input_list, prefix, suffix, expected_error",
        [
            # should fail for with both prefix and suffix not set
            (["a", "b"], None, None, ValueError),
        ],
    )
    def test_fail_string_affix_transform(
        self, spark_session, input_list, prefix, suffix, expected_error
    ):
        from pyspark.sql.types import StringType

        input_tensor = tf.constant(input_list, dtype=tf.string)
        # given
        transformer = (
            StringAffixTransformer().setInputCol("input").setOutputCol("output")
        )
        if prefix is not None:
            transformer = transformer.setPrefix(prefix)
        if suffix is not None:
            transformer = transformer.setSuffix(suffix)

        # when
        spark_cols = np.atleast_2d(input_list).T.tolist()
        spark_df = spark_session.createDataFrame(
            spark_cols,
            ["input"],
            StringType(),
        )
        # then
        with pytest.raises(expected_error):
            transformer.transform(spark_df)
        with pytest.raises(expected_error):
            transformer.get_tf_layer()(input_tensor)
