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

from kamae.spark.transformers import HaversineDistanceTransformer


class TestHaversineDistance:
    @pytest.fixture(scope="class")
    def example_dataframe_with_lat_lons(self, spark_session):
        return spark_session.createDataFrame(
            [
                (45.78, 23.09, 67.89, 12.34),
                (-45.90, -167.78, -0.12, 91.07),
                (-90.0, 180.0, 90.0, -180.0),
            ],
            ["lat1", "lon1", "lat2", "lon2"],
        )

    @pytest.fixture(scope="class")
    def example_dataframe_with_lat_lon_arrays(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    45.78,
                    23.09,
                    67.89,
                    12.34,
                    [[[45.78, 23.09]], [[67.89, 12.34]]],
                    [[[-45.90, -167.78]], [[-0.12, 91.07]]],
                    [[[-90.0, -12.0]], [[90.0, -18.0]]],
                    [[[-34.89, 12.09]], [[-61.9, 0.34]]],
                ),
                (
                    45.78,
                    23.09,
                    67.89,
                    12.34,
                    [[[45.78, 23.09]], [[67.89, 12.34]]],
                    [[[-45.90, -167.78]], [[-0.12, 91.07]]],
                    [[[-90.0, 12.0]], [[90.0, -18.0]]],
                    [[[-34.89, 12.09]], [[-61.9, 0.34]]],
                ),
            ],
            [
                "lat1_scalar",
                "lon1_scalar",
                "lat2_scalar",
                "lon2_scalar",
                "lat1_array",
                "lon1_array",
                "lat2_array",
                "lon2_array",
            ],
        )

    @pytest.fixture(scope="class")
    def example_dataframe_with_invalid_lat_lons(self, spark_session):
        return spark_session.createDataFrame(
            [
                (180.9, 23.09, 67.89, 12.34),
                (-45.90, -167.78, -0.12, 91.07),
                (-90.0, 180.0, 90.0, -270.0),
            ],
            ["lat1", "lon1", "lat2", "lon2"],
        )

    @pytest.fixture(scope="class")
    def haversine_distance_transform_lat_lon_1_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (45.78, 23.09, 67.89, 12.34, 4598.788027037987),
                (-45.90, -167.78, -0.12, 91.07, 15335.826999933153),
                (-90.0, 180.0, 90.0, -180.0, 19459.112162797792),
            ],
            ["lat1", "lon1", "lat2", "lon2", "haversine_distance_lat_lon_1"],
        )

    @pytest.fixture(scope="class")
    def haversine_distance_transform_lat_lon_2_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (45.78, 23.09, 67.89, 12.34, 9195.820595615063),
                (-45.90, -167.78, -0.12, 91.07, 15420.239622812023),
                (-90.0, 180.0, 90.0, -180.0, 7351.096600471779),
            ],
            ["lat1", "lon1", "lat2", "lon2", "haversine_distance_lat_lon_2"],
        )

    @pytest.fixture(scope="class")
    def haversine_distance_transform_lat_lon_1_2_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (45.78, 23.09, 67.89, 12.34, 2535.3581302297216),
                (-45.90, -167.78, -0.12, 91.07, 10857.854909388832),
                (-90.0, 180.0, 90.0, -180.0, 20015.086796020572),
            ],
            ["lat1", "lon1", "lat2", "lon2", "haversine_distance_lat_lon_1_2"],
        )

    @pytest.fixture(scope="class")
    def haversine_distance_transform_invalid_lat_lon_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (180.9, 23.09, 67.89, 12.34, None),
                (-45.90, -167.78, -0.12, 91.07, 10857.854909388832),
                (-90.0, 180.0, 90.0, -270.0, None),
            ],
            ["lat1", "lon1", "lat2", "lon2", "haversine_invalid_lat_lon"],
        )

    @pytest.fixture(scope="class")
    def haversine_distance_transform_lat_lon_array_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    45.78,
                    23.09,
                    67.89,
                    12.34,
                    [[[45.78, 23.09]], [[67.89, 12.34]]],
                    [[[-45.90, -167.78]], [[-0.12, 91.07]]],
                    [[[-90.0, -12.0]], [[90.0, -18.0]]],
                    [[[-34.89, 12.09]], [[-61.9, 0.34]]],
                    [
                        [[15098.047139798186, 18781.858349523165]],
                        [[2458.519828111194, 10504.20591993853]],
                    ],
                ),
                (
                    45.78,
                    23.09,
                    67.89,
                    12.34,
                    [[[45.78, 23.09]], [[67.89, 12.34]]],
                    [[[-45.90, -167.78]], [[-0.12, 91.07]]],
                    [[[-90.0, 12.0]], [[90.0, -18.0]]],
                    [[[-34.89, 12.09]], [[-61.9, 0.34]]],
                    [
                        [[15098.047139798186, 16113.23115194962]],
                        [[2458.519828111194, 10504.20591993853]],
                    ],
                ),
            ],
            [
                "lat1_scalar",
                "lon1_scalar",
                "lat2_scalar",
                "lon2_scalar",
                "lat1_array",
                "lon1_array",
                "lat2_array",
                "lon2_array",
                "haversine_distance_arrays",
            ],
        )

    @pytest.fixture(scope="class")
    def haversine_distance_transform_lat_lon_array_w_scalar_expected(
        self, spark_session
    ):
        return spark_session.createDataFrame(
            [
                (
                    45.78,
                    23.09,
                    67.89,
                    12.34,
                    [[[45.78, 23.09]], [[67.89, 12.34]]],
                    [[[-45.90, -167.78]], [[-0.12, 91.07]]],
                    [[[-90.0, -12.0]], [[90.0, -18.0]]],
                    [[[-34.89, 12.09]], [[-61.9, 0.34]]],
                    [
                        [[15098.047139798186, 16258.905030029802]],
                        [[4917.039656222387, 10593.45289733345]],
                    ],
                ),
                (
                    45.78,
                    23.09,
                    67.89,
                    12.34,
                    [[[45.78, 23.09]], [[67.89, 12.34]]],
                    [[[-45.90, -167.78]], [[-0.12, 91.07]]],
                    [[[-90.0, 12.0]], [[90.0, -18.0]]],
                    [[[-34.89, 12.09]], [[-61.9, 0.34]]],
                    [
                        [[15098.047139798186, 13590.232667290386]],
                        [[4917.039656222387, 10593.45289733345]],
                    ],
                ),
            ],
            [
                "lat1_scalar",
                "lon1_scalar",
                "lat2_scalar",
                "lon2_scalar",
                "lat1_array",
                "lon1_array",
                "lat2_array",
                "lon2_array",
                "haversine_distance_array_w_scalar",
            ],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_cols, output_col, lat_lon_constant, expected_dataframe",
        [
            (
                "example_dataframe_with_lat_lons",
                ["lat1", "lon1"],
                "haversine_distance_lat_lon_1",
                [85.0, 76.0],
                "haversine_distance_transform_lat_lon_1_expected",
            ),
        ],
    )
    def test_spark_haversine_distance_1_transform(
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
            HaversineDistanceTransformer(
                inputCols=input_cols,
                outputCol=output_col,
                latLonConstant=lat_lon_constant,
            )
            if lat_lon_constant is not None
            else HaversineDistanceTransformer(
                inputCols=input_cols,
                outputCol=output_col,
            )
        )
        actual = transformer.transform(input_dataframe)
        actual_values = (
            actual.select("haversine_distance_lat_lon_1")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        # Convert expected DataFrame to a list (assuming 'expected' is a PySpark DataFrame)
        expected_values = (
            expected.select("haversine_distance_lat_lon_1")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        actual_values = np.where(actual_values == None, np.nan, actual_values).astype(
            float
        )
        expected_values = np.where(
            expected_values == None, np.nan, expected_values
        ).astype(float)

        # Then compare using np.testing.assert_almost_equal
        np.testing.assert_almost_equal(
            actual_values,
            expected_values,
            decimal=2,
            err_msg="Expected and actual dataframes are not equal",
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_cols, output_col, lat_lon_constant, expected_dataframe",
        [
            (
                "example_dataframe_with_lat_lons",
                ["lat2", "lon2"],
                "haversine_distance_lat_lon_2",
                [23.89, -123.8],
                "haversine_distance_transform_lat_lon_2_expected",
            ),
        ],
    )
    def test_spark_haversine_distance_2_transform(
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
            HaversineDistanceTransformer(
                inputCols=input_cols,
                outputCol=output_col,
                latLonConstant=lat_lon_constant,
            )
            if lat_lon_constant is not None
            else HaversineDistanceTransformer(
                inputCols=input_cols,
                outputCol=output_col,
            )
        )
        actual = transformer.transform(input_dataframe)
        actual_values = (
            actual.select("haversine_distance_lat_lon_2")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        # Convert expected DataFrame to a list (assuming 'expected' is a PySpark DataFrame)
        expected_values = (
            expected.select("haversine_distance_lat_lon_2")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        actual_values = np.where(actual_values == None, np.nan, actual_values).astype(
            float
        )
        expected_values = np.where(
            expected_values == None, np.nan, expected_values
        ).astype(float)

        # Then compare using np.testing.assert_almost_equal
        np.testing.assert_almost_equal(
            actual_values,
            expected_values,
            decimal=2,
            err_msg="Expected and actual dataframes are not equal",
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_cols, output_col, lat_lon_constant, expected_dataframe",
        [
            (
                "example_dataframe_with_lat_lons",
                ["lat1", "lon1", "lat2", "lon2"],
                "haversine_distance_lat_lon_1_2",
                None,
                "haversine_distance_transform_lat_lon_1_2_expected",
            ),
        ],
    )
    def test_spark_haversine_distance_1_2_transform(
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
            HaversineDistanceTransformer(
                inputCols=input_cols,
                outputCol=output_col,
                latLonConstant=lat_lon_constant,
            )
            if lat_lon_constant is not None
            else HaversineDistanceTransformer(
                inputCols=input_cols,
                outputCol=output_col,
            )
        )
        actual = transformer.transform(input_dataframe)
        actual_values = (
            actual.select("haversine_distance_lat_lon_1_2")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        # Convert expected DataFrame to a list (assuming 'expected' is a PySpark DataFrame)
        expected_values = (
            expected.select("haversine_distance_lat_lon_1_2")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        actual_values = np.where(actual_values == None, np.nan, actual_values).astype(
            float
        )
        expected_values = np.where(
            expected_values == None, np.nan, expected_values
        ).astype(float)

        # Then compare using np.testing.assert_almost_equal
        np.testing.assert_almost_equal(
            actual_values,
            expected_values,
            decimal=2,
            err_msg="Expected and actual dataframes are not equal",
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_cols, output_col, lat_lon_constant, expected_dataframe",
        [
            (
                "example_dataframe_with_lat_lon_arrays",
                ["lat1_array", "lon1_array", "lat2_array", "lon2_array"],
                "haversine_distance_arrays",
                None,
                "haversine_distance_transform_lat_lon_array_expected",
            ),
        ],
    )
    def test_spark_haversine_distance_arrays_transform(
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
            HaversineDistanceTransformer(
                inputCols=input_cols,
                outputCol=output_col,
                latLonConstant=lat_lon_constant,
            )
            if lat_lon_constant is not None
            else HaversineDistanceTransformer(
                inputCols=input_cols,
                outputCol=output_col,
            )
        )
        actual = transformer.transform(input_dataframe)
        actual_values = (
            actual.select("haversine_distance_arrays").rdd.map(lambda r: r[0]).collect()
        )

        # Convert expected DataFrame to a list (assuming 'expected' is a PySpark DataFrame)
        expected_values = (
            expected.select("haversine_distance_arrays")
            .rdd.map(lambda r: r[0])
            .collect()
        )

        actual_values = np.where(actual_values == None, np.nan, actual_values).astype(
            float
        )
        expected_values = np.where(
            expected_values == None, np.nan, expected_values
        ).astype(float)

        # Then compare using np.testing.assert_almost_equal
        np.testing.assert_almost_equal(
            actual_values,
            expected_values,
            decimal=2,
            err_msg="Expected and actual dataframes are not equal",
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_cols, output_col, lat_lon_constant, expected_dataframe",
        [
            (
                "example_dataframe_with_lat_lon_arrays",
                ["lat1_scalar", "lon1_array", "lat2_array", "lon2_scalar"],
                "haversine_distance_array_w_scalar",
                None,
                "haversine_distance_transform_lat_lon_array_w_scalar_expected",
            ),
        ],
    )
    def test_spark_haversine_distance_arrays_w_scalar_transform(
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
            HaversineDistanceTransformer(
                inputCols=input_cols,
                outputCol=output_col,
                latLonConstant=lat_lon_constant,
            )
            if lat_lon_constant is not None
            else HaversineDistanceTransformer(
                inputCols=input_cols,
                outputCol=output_col,
            )
        )
        actual = transformer.transform(input_dataframe)
        actual_values = (
            actual.select("haversine_distance_array_w_scalar")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        # Convert expected DataFrame to a list (assuming 'expected' is a PySpark DataFrame)
        expected_values = (
            expected.select("haversine_distance_array_w_scalar")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        actual_values = np.where(actual_values == None, np.nan, actual_values).astype(
            float
        )
        expected_values = np.where(
            expected_values == None, np.nan, expected_values
        ).astype(float)

        # Then compare using np.testing.assert_almost_equal
        np.testing.assert_almost_equal(
            actual_values,
            expected_values,
            decimal=2,
            err_msg="Expected and actual dataframes are not equal",
        )

    @pytest.mark.parametrize(
        "input_cols, output_col, lat_lon_constant, expected_dataframe",
        [
            (
                ["lat1", "lon1", "lat2", "lon2"],
                "haversine_invalid_lat_lon",
                None,
                "haversine_distance_transform_invalid_lat_lon_expected",
            ),
        ],
    )
    def test_spark_haversine_distance_invalid_transform(
        self,
        example_dataframe_with_invalid_lat_lons,
        input_cols,
        output_col,
        lat_lon_constant,
        expected_dataframe,
        request,
    ):
        # given
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = (
            HaversineDistanceTransformer(
                inputCols=input_cols,
                outputCol=output_col,
                latLonConstant=lat_lon_constant,
            )
            if lat_lon_constant is not None
            else HaversineDistanceTransformer(
                inputCols=input_cols,
                outputCol=output_col,
            )
        )
        actual = transformer.transform(example_dataframe_with_invalid_lat_lons)
        actual_values = (
            actual.select("haversine_invalid_lat_lon").rdd.map(lambda r: r[0]).collect()
        )
        # Convert expected DataFrame to a list (assuming 'expected' is a PySpark DataFrame)
        expected_values = (
            expected.select("haversine_invalid_lat_lon")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        actual_values = np.where(actual_values == None, np.nan, actual_values).astype(
            float
        )
        expected_values = np.where(
            expected_values == None, np.nan, expected_values
        ).astype(float)

        # Then compare using np.testing.assert_almost_equal
        np.testing.assert_almost_equal(
            actual_values,
            expected_values,
            decimal=2,
            err_msg="Expected and actual dataframes are not equal",
        )

    def test_haversine_distance_transform_defaults(self):
        # when
        haversine_distance_transform = HaversineDistanceTransformer()
        # then
        assert (
            haversine_distance_transform.getLayerName()
            == haversine_distance_transform.uid
        )
        assert (
            haversine_distance_transform.getOutputCol()
            == f"{haversine_distance_transform.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensors, input_dtype, output_dtype, lat_lon_constant, unit",
        [
            (
                [
                    tf.constant([45.78, 23.09, -45.90, -67.78, -90.0, 78.0]),
                    tf.constant([67.89, 12.34, -0.12, 91.07, 90.0, -180.0]),
                ],
                None,
                "double",
                [85.0, 76.0],
                "km",
            ),
            (
                [
                    tf.constant([45.78, 23.09, -45.90, -67.78, -90.0, 12.0]),
                    tf.constant([67.89, 12.34, -0.12, 91.07, 90.0, -180.0]),
                    tf.constant([23.45, 76.89, -89.0, 88.07, 9.87, -18.0]),
                    tf.constant([120.0, 120.34, -12.98, 9.07, 9.0, -180.0]),
                ],
                "double",
                "float",
                None,
                "miles",
            ),
            (
                [
                    tf.constant([45.78, -45.9, -90]),
                    tf.constant([23.09, -167.78, 180.0]),
                    tf.constant([67.89, -0.12, 90.0]),
                    tf.constant([12.34, 91.07, -180.0]),
                ],
                None,
                None,
                None,
                "miles",
            ),
        ],
    )
    def test_haversine_distance_transform_spark_tf_parity(
        self,
        spark_session,
        input_tensors,
        input_dtype,
        output_dtype,
        lat_lon_constant,
        unit,
    ):
        # given
        transformer = HaversineDistanceTransformer(
            inputCols=[f"input_{i}" for i in range(len(input_tensors))],
            outputCol="output",
            latLonConstant=lat_lon_constant,
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            unit=unit,
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
            decimal=6,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
