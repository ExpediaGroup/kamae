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
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, FloatType

from kamae.spark.transformers import BearingAngleTransformer


class TestBearingAngle:
    @pytest.fixture(scope="class")
    def example_dataframe_with_lat_lons(self, spark_session):
        return spark_session.createDataFrame(
            [
                (-12.05, -77.04, 37.77, -122.42),  # Lima - San Francisco
                (39.90, 116.41, -33.87, 151.21),  # Beijing - Sidney
                (39.90, 116.41, -12.05, -77.04),  # Beijing - Lima
            ],
            ["lat1", "lon1", "lat2", "lon2"],
        )

    @pytest.fixture(scope="class")
    def bearing_angle_transform_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (-12.05, -77.04, 37.77, -122.42, 321.7967),  # Expected bearing angles
                (39.90, 116.41, -33.87, 151.21, 151.2820),
                (39.90, 116.41, -12.05, -77.04, 26.8186),
            ],
            ["lat1", "lon1", "lat2", "lon2", "bearing_angle"],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_cols, output_col, lat_lon_constant, expected_dataframe",
        [
            (
                "example_dataframe_with_lat_lons",
                ["lat1", "lon1", "lat2", "lon2"],
                "bearing_angle",
                None,
                "bearing_angle_transform_expected",
            ),
        ],
    )
    def test_spark_bearing_angle_transform(
        self,
        input_dataframe,
        input_cols,
        output_col,
        lat_lon_constant,
        expected_dataframe,
        request,
    ):
        # given
        input_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)

        # when
        transformer = (
            BearingAngleTransformer(
                inputCols=input_cols,
                outputCol=output_col,
                inputDtype="double",  # Ensure dtype matches expected input types
                outputDtype="double",
                latLonConstant=lat_lon_constant,
            )
            if lat_lon_constant is not None
            else BearingAngleTransformer(
                inputCols=input_cols,
                outputCol=output_col,
                inputDtype="double",  # Ensure dtype matches expected input types
                outputDtype="double",
            )
        )
        actual = transformer.transform(input_dataframe)

        # Round all numeric columns to 2 decimals in both DataFrames
        for col_name in actual.columns:
            if isinstance(actual.schema[col_name].dataType, (DoubleType, FloatType)):
                actual = actual.withColumn(col_name, F.round(F.col(col_name), 2))
                expected = expected.withColumn(col_name, F.round(F.col(col_name), 2))

        # then
        # Perform the comparison
        diff = actual.subtract(expected).union(expected.subtract(actual))
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_bearing_angle_transform_defaults(self):
        # when
        bearing_angle_transform = BearingAngleTransformer()

        # then
        assert bearing_angle_transform.getLayerName() == bearing_angle_transform.uid
        assert (
            bearing_angle_transform.getOutputCol()
            == f"{bearing_angle_transform.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensors, input_dtype, output_dtype, lat_lon_constant",
        [
            (
                [
                    tf.constant([45.78, 23.09, -45.90, -67.78, -90.0, 78.0]),
                    tf.constant([67.89, 12.34, -0.12, 91.07, 90.0, -180.0]),
                ],
                None,
                "double",
                [85.0, 76.0],
            ),
        ],
    )
    def test_bearing_angle_transform_spark_tf_parity(
        self, spark_session, input_tensors, input_dtype, output_dtype, lat_lon_constant
    ):
        # given
        transformer = BearingAngleTransformer(
            inputCols=[f"input_{i}" for i in range(len(input_tensors))],
            outputCol="output",
            latLonConstant=lat_lon_constant,
            inputDtype=input_dtype,
            outputDtype=output_dtype,
        )
        # when
        spark_df = spark_session.createDataFrame(
            zip(*[input_tensor.numpy().tolist() for input_tensor in input_tensors]),
            [f"input_{i}" for i in range(len(input_tensors))],
        )
        spark_values = (
            transformer.transform(spark_df)
            .select("output")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        tensorflow_values = transformer.get_tf_layer()(input_tensors).numpy().tolist()

        # then
        np.testing.assert_almost_equal(
            spark_values,
            tensorflow_values,
            decimal=2,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
