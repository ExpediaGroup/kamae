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
from typing import List, Optional

import pyspark.sql.functions as F
import tensorflow as tf
from pyspark import keyword_only
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql.types import DataType, DoubleType, LongType

from kamae.spark.params import (
    DateTimeParams,
    SingleInputSingleOutputParams,
    UnixTimestampParams,
)
from kamae.spark.transformers.base import BaseTransformer
from kamae.spark.utils import single_input_single_output_scalar_transform
from kamae.tensorflow.layers import UnixTimestampToDateTimeLayer


class UnixTimestampToDateTimeTransformer(
    BaseTransformer, SingleInputSingleOutputParams, UnixTimestampParams, DateTimeParams
):
    """
    Transformer that converts a unix timestamp to a datetime.

    The unix timestamp can be in milliseconds or seconds, set by the `unit` parameter.
    If the `includeTime` parameter is set to True, the output will be in
    yyyy-MM-dd HH:mm:ss.SSS format. If set to False, the output will be in
    yyyy-MM-dd format.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        unit: str = "s",
        includeTime: bool = True,
        layerName: Optional[str] = None,
    ) -> None:
        """
        Initialises the UnixTimestampToDateTimeTransformer.

        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param unit: Unit of the timestamp. Can be `milliseconds` (shorthand `ms`)
         or `seconds` (shorthand `s`). Default is `s` (seconds).
        :param includeTime: Whether to include the time in the output. Default is True.
        :param layerName: Layer name. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(unit="s", includeTime=True)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        # TODO: Remove this when we only support PySpark 3.5+. It is only used to get
        #  the timezone set by the user for datetime operations. In 3.5+ we can use the
        #  current_timezone() function. Also is there a better way to access this than
        #  inside a class attribute? Setting it at the top of the file causes issues
        #  in tests as we import all transformers when the package is loaded.
        self.spark = SparkSession.builder.getOrCreate()

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return [
            DoubleType(),
            LongType(),
        ]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input integer timestamp to the date string with format
        yyyy-MM-dd HH:mm:ss.SSS.

        :param dataset: Input dataframe.
        :returns: Transformed dataframe.
        """

        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )

        def unix_timestamp_to_datetime(
            unix_timestamp: Column, include_time: bool
        ) -> Column:
            """
            Returns the date in yyyy-MM-dd HH:mm:ss.SSS format from a Unix timestamp
            in seconds.

            :param unix_timestamp: Unix timestamp in seconds.
            :param include_time: Whether to include the time in the output.
            :returns: Column of the date in yyyy-MM-dd HH:mm:ss.SSS format if
            include_time is True, otherwise in yyyy-MM-dd format.
            """
            # from_unixtime throws away milliseconds, so we have to calculate them
            # separately
            milliseconds_3dp = (
                (unix_timestamp - F.floor(unix_timestamp)) * 1000.0
            ).cast("int")
            local_datetime_str_wo_millis = F.from_unixtime(
                unix_timestamp, format="yyyy-MM-dd HH:mm:ss"
            )
            local_datetime_str_w_millis = F.concat(
                local_datetime_str_wo_millis,
                F.lit("."),
                F.lpad(milliseconds_3dp.cast("string"), 3, "0"),
            )
            local_datetime = F.to_timestamp(
                local_datetime_str_w_millis, format="yyyy-MM-dd HH:mm:ss.SSS"
            )
            utc_datetime = F.to_utc_timestamp(
                local_datetime, self.spark.conf.get("spark.sql.session.timeZone")
            )
            date_fmt = "yyyy-MM-dd HH:mm:ss.SSS" if include_time else "yyyy-MM-dd"
            return F.date_format(utc_datetime, date_fmt)

        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: unix_timestamp_to_datetime(x, self.getIncludeTime())
            if self.getUnit() == "s"
            else unix_timestamp_to_datetime(x / F.lit(1000), self.getIncludeTime()),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer that performs the unix timestamp to date transform.

        :returns: Tensorflow layer that performs the unix timestamp to date transform.
        """
        return UnixTimestampToDateTimeLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            unit=self.getUnit(),
            include_time=self.getIncludeTime(),
        )
