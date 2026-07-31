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

from kamae.spark.transformers import ArrayContainsTransformer


class TestArrayContains:
    @pytest.fixture(scope="class")
    def example_dataframe_with_arrays(self, spark_session):
        return spark_session.createDataFrame(
            [
                ([1, 2, 3], 2, 4),
                ([1, 2, 3], 5, 1),
                ([4, 5, 6], 4, 9),
            ],
            ["array_col", "value_col", "other_value_col"],
        )

    @pytest.fixture(scope="class")
    def array_contains_transform_array_value_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                ([1, 2, 3], 2, 4, 1.0),
                ([1, 2, 3], 5, 1, 0.0),
                ([4, 5, 6], 4, 9, 1.0),
            ],
            ["array_col", "value_col", "other_value_col", "array_contains_value"],
        )

    @pytest.fixture(scope="class")
    def array_contains_transform_array_other_value_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                ([1, 2, 3], 2, 4, 0.0),
                ([1, 2, 3], 5, 1, 1.0),
                ([4, 5, 6], 4, 9, 0.0),
            ],
            [
                "array_col",
                "value_col",
                "other_value_col",
                "array_contains_other_value",
            ],
        )

    @pytest.mark.parametrize(
        "input_cols, output_col, expected_dataframe",
        [
            (
                ["array_col", "value_col"],
                "array_contains_value",
                "array_contains_transform_array_value_expected",
            ),
            (
                ["array_col", "other_value_col"],
                "array_contains_other_value",
                "array_contains_transform_array_other_value_expected",
            ),
        ],
    )
    def test_spark_array_contains_transform(
        self,
        example_dataframe_with_arrays,
        input_cols,
        output_col,
        expected_dataframe,
        request,
    ):
        # given
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = ArrayContainsTransformer(
            inputCols=input_cols,
            outputCol=output_col,
        )
        actual = transformer.transform(example_dataframe_with_arrays)
        # then
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_array_contains_transform_defaults(self):
        # when
        array_contains_transform = ArrayContainsTransformer()
        # then
        assert array_contains_transform.getLayerName() == array_contains_transform.uid
        assert (
            array_contains_transform.getOutputCol()
            == f"{array_contains_transform.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_cols",
        [
            ["array_col"],
            ["array_col", "value_col", "other_value_col"],
        ],
    )
    def test_array_contains_transform_wrong_number_of_inputs_raises_error(
        self, input_cols
    ):
        # then
        with pytest.raises(ValueError):
            ArrayContainsTransformer(inputCols=input_cols)

    def test_array_contains_transform_non_array_input_raises_error(
        self, example_dataframe_with_arrays
    ):
        # given
        transformer = ArrayContainsTransformer(
            inputCols=["value_col", "other_value_col"],
            outputCol="array_contains_output",
        )
        # then
        with pytest.raises(TypeError):
            transformer.transform(example_dataframe_with_arrays).collect()

    @pytest.mark.parametrize(
        "input_arrays, input_values, input_dtype, output_dtype",
        [
            (
                [[1, 2, 3], [1, 2, 3], [4, 5, 6]],
                [2, 5, 4],
                None,
                None,
            ),
            (
                [[10, 20, 30, 40], [5, 6, 7, 8], [0, 0, 0, 0]],
                [30, 100, 0],
                "bigint",
                "double",
            ),
        ],
    )
    def test_array_contains_transform_spark_tf_parity(
        self,
        spark_session,
        input_arrays,
        input_values,
        input_dtype,
        output_dtype,
    ):
        # given
        transformer = ArrayContainsTransformer(
            inputCols=["array_col", "value_col"],
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
        )
        # when
        spark_df = spark_session.createDataFrame(
            zip(input_arrays, input_values),
            ["array_col", "value_col"],
        )
        spark_values = (
            transformer.transform(spark_df)
            .select("output")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        array_tensor = tf.constant(input_arrays)
        value_tensor = tf.constant([[v] for v in input_values])
        tensorflow_values = (
            transformer.get_keras_layer()([array_tensor, value_tensor])
            .numpy()
            .flatten()
            .tolist()
        )

        # then
        np.testing.assert_almost_equal(
            spark_values,
            tensorflow_values,
        )
