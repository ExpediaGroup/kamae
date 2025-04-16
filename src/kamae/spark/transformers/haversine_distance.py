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
import math
from typing import List, Optional

import pyspark.sql.functions as F
import tensorflow as tf
from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import DataType, DoubleType, FloatType

from kamae.spark.params import LatLonConstantParams, MultiInputSingleOutputParams
from kamae.spark.utils import multi_input_single_output_scalar_transform
from kamae.tensorflow.layers import HaversineDistanceLayer

from .base import BaseTransformer


class HaversineDistanceParams(LatLonConstantParams, MultiInputSingleOutputParams):
    """
    Mixin class containing unit parameters.
    """

    unit = Param(
        Params._dummy(),
        "unit",
        """The unit to use for the distance calculation.
        Must be either "km" or "miles".""",
        typeConverter=TypeConverters.toString,
    )

    def setUnit(self, value: str) -> "HaversineDistanceParams":
        """
        Sets the unit parameter.
        :param value: The unit to use for the distance calculation.
        :returns: Instance of class mixed in.
        """
        if value not in ["km", "miles"]:
            raise ValueError("unit must be either 'km' or 'miles'")
        return self._set(unit=value)

    def getUnit(self) -> str:
        """
        Gets the unit parameter.
        :returns: The unit to use for the distance calculation.
        """
        return self.getOrDefault(self.unit)

    def setInputCols(self, value: List[str]) -> "HaversineDistanceParams":
        """
        Overrides setting the input columns for the transformer.
        Throws an error if we do not have either two or four input columns depending
        on whether latLonConstant is provided.
        :param value: List of input columns.
        :returns: Class instance.
        """
        if self.getLatLonConstant() is not None and len(value) != 2:
            raise ValueError(
                """When setting inputCols for HaversineDistanceTransformer,
                if the latLonConstant is not None,
                there must be exactly two input columns."""
            )
        elif len(value) not in [2, 4]:
            raise ValueError(
                """When setting inputCols for HaversineDistanceTransformer,
                there must be either two or four input columns."""
            )
        return self._set(inputCols=value)


class HaversineDistanceTransformer(
    BaseTransformer,
    HaversineDistanceParams,
):
    """
    Haversine Distance Spark Transformer for use in Spark pipelines.
    This transformer computes the haversine distance between two lat/lon pairs.
    This can be between four columns (one for each lat/lon) or between two columns
    and a constant.

    The transformer will return null distance if any of the lat/lon values
    are out of bounds. For lat, this is [-90, 90] and for lon, this is [-180, 180].
    """

    @keyword_only
    def __init__(
        self,
        inputCols: Optional[List[str]] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        latLonConstant: Optional[List[float]] = None,
        unit: str = "km",
    ) -> None:
        """
        Initializes an HaversineDistanceTransformer transformer.

        :param inputCols: Input column names. If latLonConstant is provided, then two
        input columns are required. These must be in the order [lat, lon].
        If latLonConstant is not provided, then four input columns are required.
        These must be in the order [lat1, lon1, lat2, lon2].
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column(s) to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param latLonConstant: Optional list of lat/lon constant to use.
        Must be in the order [lat, lon].
        If not provided, then four input columns are required.
        :param unit: The unit to use for the distance calculation.
        Must be either "km" or "miles".
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(latLonConstant=None, unit="km")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        self.earth_radius = 6371.0 if unit == "km" else 3958.8

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return [FloatType(), DoubleType()]

    @staticmethod
    def validate_lat_lon_column(x: Column, lat: bool = True) -> Column:
        """
        Validates that the input column is a valid lat or lon column.
        If not, then casts the column to null.

        :param x: Input column.
        :param lat: Whether the column is a lat column or not.
        :returns: Column.
        """
        cond = x.between(-90.0, 90.0) if lat else x.between(-180.0, 180.0)
        return F.when(cond, x)

    def _get_input_cols(self) -> List[Column]:
        """
        Gets the input columns as a list of pyspark.sql.Column. Also checks if the
        lat and lons are out of bounds and if so casts them to null.
        This will make the output null.

        If the latLonConstant is provided, then returns the input columns and the
        constant split into lat and lon. Otherwise, returns the input columns.

        :returns: List of input columns.
        """
        input_cols = self.getInputCols()

        lat_lon_constant = self.getLatLonConstant()

        if lat_lon_constant is not None:
            return [
                F.col(input_cols[0]).alias(self.uid + "_lat1"),
                F.col(input_cols[1]).alias(self.uid + "_lon1"),
                F.lit(lat_lon_constant[0]).alias(self.uid + "_lat2"),
                F.lit(lat_lon_constant[1]).alias(self.uid + "_lon2"),
            ]
        else:
            return [
                F.col(input_cols[0]).alias(self.uid + "_lat1"),
                F.col(input_cols[1]).alias(self.uid + "_lon1"),
                F.col(input_cols[2]).alias(self.uid + "_lat2"),
                F.col(input_cols[3]).alias(self.uid + "_lon2"),
            ]

    @staticmethod
    def _to_radians_col(x: Column) -> Column:
        """
        Converts a column of degrees to radians.

        :param x: Column of degrees.
        :returns: Column of radians in double precision.
        """
        return x.cast(DoubleType()) * F.lit(math.pi / 180)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset. Creates a new column with name `outputCol`,
        which is the haversine distance between the input lat/lon columns.

        Returns null if any of the lat/lon values are out of bounds.

        :param dataset: Pyspark dataframe to transform.
        :returns: Transformed pyspark dataframe.
        """
        input_cols = self._get_input_cols()
        # input_cols can contain either actual columns or lit(constants). In order to
        # determine the datatype of the input columns, we select them from the dataset
        # first.
        input_col_names = dataset.select(input_cols).columns
        input_col_datatypes = [
            self.get_column_datatype(dataset=dataset.select(input_cols), column_name=c)
            for c in input_col_names
        ]

        def haversine_distance_transform(
            x: Column, input_col_names: List[str]
        ) -> Column:
            lat1_radians = self._to_radians_col(
                self.validate_lat_lon_column(x[input_col_names[0]])
            )
            lon1_radians = self._to_radians_col(
                self.validate_lat_lon_column(x[input_col_names[1]], lat=False)
            )
            lat2_radians = self._to_radians_col(
                self.validate_lat_lon_column(x[input_col_names[2]])
            )
            lon2_radians = self._to_radians_col(
                self.validate_lat_lon_column(x[input_col_names[3]], lat=False)
            )

            lat_diff = lat2_radians - lat1_radians
            lon_diff = lon2_radians - lon1_radians

            a = F.pow(F.sin(lat_diff / 2.0), 2.0) + F.cos(lat1_radians) * F.cos(
                lat2_radians
            ) * F.pow(F.sin(lon_diff / 2.0), 2.0)
            c = 2.0 * F.asin(F.pow(a, 0.5))
            r = F.lit(self.earth_radius)

            return c * r

        output_col = multi_input_single_output_scalar_transform(
            input_cols=input_cols,
            input_col_names=input_col_names,
            input_col_datatypes=input_col_datatypes,
            func=lambda x: haversine_distance_transform(x, input_col_names),
        )

        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer for the haversine distance transformer.

        :returns: Tensorflow keras layer with name equal to the layerName parameter that
         computes the haversine distance between two lat/lon pairs.
        """
        return HaversineDistanceLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            lat_lon_constant=self.getLatLonConstant(),
            unit=self.getUnit(),
        )
