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

# pylint: disable=unused-argument
# pylint: disable=invalid-name
# pylint: disable=too-many-ancestors
# pylint: disable=no-member
import numpy as np
import pytest
import tensorflow as tf

from kamae.spark.transformers import StringMapTransformer

from ..test_helpers import tensor_to_python_type


class TestStringMap:
    @pytest.mark.parametrize(
        "input_values, string_match_values, string_replace_values, default_replace_value, expected_output",
        [
            (
                [
                    ("abc",),
                    ("def",),
                    ("hij",),
                ],
                ["abc", "def"],
                ["xyz1", "xyz2"],
                None,
                [
                    ("xyz1",),
                    ("xyz2",),
                    ("hij",),
                ],
            ),
            # Test replace default value
            (
                [
                    ("abc",),
                    ("def",),
                    ("hij",),
                ],
                ["abc", "def"],
                ["xyz1", "xyz2"],
                "default",
                [
                    ("xyz1",),
                    ("xyz2",),
                    ("default",),
                ],
            ),
        ],
    )
    def test_string_map_transform_layer(
        self,
        spark_session,
        input_values,
        string_match_values,
        string_replace_values,
        default_replace_value,
        expected_output,
    ):
        input_dataframe = spark_session.createDataFrame(input_values, ["input"])
        expected_dataframe = spark_session.createDataFrame(expected_output, ["output"])
        layer = StringMapTransformer(
            inputCol="input",
            outputCol="output",
            stringMatchValues=string_match_values,
            stringReplaceValues=string_replace_values,
            defaultReplaceValue=default_replace_value,
        )
        actual = layer.transform(input_dataframe).select("output")
        diff = actual.exceptAll(expected_dataframe)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    @pytest.mark.parametrize(
        "input_tensors, input_dtype, output_dtype, string_match_values, string_replace_values, default_replace_value",
        [
            (
                [tf.constant(["hello", "there", "friend"])],
                None,
                None,
                ["hello", "friend"],
                ["hi", "pal"],
                None,
            ),
            (
                [tf.constant(["hello", "there", "friend"])],
                None,
                None,
                ["hello", "friend"],
                ["hi", "pal"],
                "default",
            ),
            # test int map value through string casting
            (
                [tf.constant([100, 200])],
                "string",
                "bigint",
                ["100", "200"],
                ["0", "0"],
                None,
            ),
        ],
    )
    def test_string_map_spark_tf_parity_no_constants(
        self,
        spark_session,
        input_tensors,
        input_dtype,
        output_dtype,
        string_match_values,
        string_replace_values,
        default_replace_value,
    ):
        # given
        transformer = StringMapTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            stringMatchValues=string_match_values,
            stringReplaceValues=string_replace_values,
            defaultReplaceValue=default_replace_value,
        )

        # when
        spark_df = spark_session.createDataFrame(
            [
                tuple([tensor_to_python_type(ti) for ti in t])
                for t in zip(*input_tensors)
            ],
            ["input"],
        )
        spark_values = (
            transformer.transform(spark_df)
            .select("output")
            .rdd.map(lambda x: x[0])
            .collect()
        )
        tensorflow_values = [
            v.decode("utf-8") if isinstance(v, bytes) else v
            for v in transformer.get_tf_layer()(input_tensors).numpy().tolist()
        ]

        # then
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
