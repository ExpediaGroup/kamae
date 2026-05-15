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

import pyspark.sql.functions as F
from pyspark.ml.param import TypeConverters
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql.types import DoubleType, LongType

from kamae.keras.tensorflow.layers import UnixTimestampToDateTimeLayer
from kamae.params import ParamSpec
from kamae.params.shared_specs import UNIX_TIMESTAMP_PARAMS
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.transformers.base import BaseTransformer
from kamae.spark.utils import single_input_single_output_scalar_transform


class UnixTimestampToDateTimeTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    Transformer that converts a unix timestamp to a datetime.

    The unix timestamp can be in milliseconds or seconds, set by the `unit` parameter.
    If the `includeTime` parameter is set to True, the output will be in
    yyyy-MM-dd HH:mm:ss.SSS format. If set to False, the output will be in
    yyyy-MM-dd format.
    """

    _compatible_dtypes = [DoubleType(), LongType()]
    _keras_layer_class = UnixTimestampToDateTimeLayer
    _params = {
        **UNIX_TIMESTAMP_PARAMS,
        "includeTime": ParamSpec(
            spark_typeconverter=TypeConverters.toBoolean,
            default=True,
            doc="Whether to include the time in the output.",
        ),
    }

    @property
    def spark(self):
        """
        Returns the current Spark session.

        TODO: Remove this when we only support PySpark 3.5+. It is only used to get
        the timezone set by the user for datetime operations. In 3.5+ we can use the
        current_timezone() function.
        """
        return SparkSession.builder.getOrCreate()

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
