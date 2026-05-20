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
from pyspark.sql.types import ArrayType, FloatType, StructField, StructType

from kamae.spark.transformers import ArrayReduceMaxTransformer


class TestArrayReduceMaxTransformer:
    @pytest.fixture(scope="class")
    def input_df(self, spark_session):
        return spark_session.createDataFrame(
            [
                ([3.0, 1.0, 2.0],),
                ([0.0, 5.0, 4.0],),
                ([-3.0, -1.0, -2.0],),
            ],
            ["values"],
        )

    def test_returns_maximum_of_each_row(self, input_df):
        transformer = ArrayReduceMaxTransformer(inputCol="values", outputCol="result")
        result = transformer.transform(input_df).select("result").collect()

        assert [row.result for row in result] == pytest.approx([3.0, 5.0, -1.0])

    def test_default_value_for_empty_array(self, spark_session):
        schema = StructType(
            [StructField("values", ArrayType(FloatType()), nullable=True)]
        )

        df = spark_session.createDataFrame(
            [([],)],
            schema=schema,
        )

        transformer = ArrayReduceMaxTransformer(
            inputCol="values", outputCol="result", defaultValue=-99.0
        )
        result = transformer.transform(df).select("result").collect()

        assert result[0].result == pytest.approx(-99.0)

    @pytest.mark.parametrize(
        "rows, input_dtype, output_dtype, default_value",
        [
            # default dtypes
            (
                [[3.0, 1.0, 2.0], [0.0, 5.0, 4.0], [-3.0, -1.0, -2.0]],
                None,
                None,
                0.0,
            ),
            # float input, double output
            (
                [[3.0, 1.0, 2.0], [0.0, 5.0, 4.0], [-3.0, -1.0, -2.0]],
                "float",
                "double",
                0.0,
            ),
            # double input, float output
            (
                [[3.0, 1.0, 2.0], [0.0, 5.0, 4.0], [-3.0, -1.0, -2.0]],
                "double",
                "float",
                0.0,
            ),
            # different array length (5 elements)
            (
                [[5.0, 3.0, 1.0, 4.0, 2.0], [-1.0, -3.0, -2.0, -5.0, -4.0]],
                None,
                None,
                0.0,
            ),
            # non-default defaultValue is forwarded correctly to TF layer
            (
                [[1.0, 2.0, 3.0], [-5.0, -4.0, -6.0]],
                None,
                None,
                -99.0,
            ),
        ],
    )
    def test_spark_tf_parity(
        self, spark_session, rows, input_dtype, output_dtype, default_value
    ):
        transformer = ArrayReduceMaxTransformer(
            inputCol="values",
            outputCol="result",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            defaultValue=default_value,
        )

        spark_df = spark_session.createDataFrame([(row,) for row in rows], ["values"])
        spark_values = (
            transformer.transform(spark_df)
            .select("result")
            .rdd.map(lambda r: r[0])
            .collect()
        )

        inputs = tf.constant(rows, dtype=tf.float32)
        keras_values = transformer.get_keras_layer()(inputs).numpy().tolist()

        np.testing.assert_almost_equal(
            spark_values,
            keras_values,
            decimal=4,
            err_msg="Spark and TensorFlow outputs are not equal",
        )
