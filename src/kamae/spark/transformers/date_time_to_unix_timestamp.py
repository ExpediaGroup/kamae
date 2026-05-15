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
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import StringType

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.tensorflow.layers import DateTimeToUnixTimestampLayer
from kamae.params.shared_specs import UNIX_TIMESTAMP_PARAMS
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.transformers.base import BaseTransformer
from kamae.spark.utils import single_input_single_output_scalar_transform


class DateTimeToUnixTimestampTransformer(
    BaseTransformer, SingleInputSingleOutputParams
):
    """
    Transformer that converts a datetime string to a unix timestamp.

    The unix timestamp can be in milliseconds or seconds, set by the `unit` parameter.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = [StringType()]
    _keras_layer_class = DateTimeToUnixTimestampLayer
    _params = {**UNIX_TIMESTAMP_PARAMS}

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

        def datetime_to_unix_timestamp(datetime: Column) -> Column:
            """
            Converts a datetime string to a unix timestamp in seconds.

            :param datetime: Column of datetime strings.
            :returns: Column of unix timestamps.
            """
            # Check if we have a date only or datetime. This is quite a crude way to
            # check if we have a datetime string.
            split_datetime = F.split(datetime, " ")
            is_datetime = F.size(split_datetime) > 1
            # If we have a date only, add 00:00:00.000 UTC. Otherwise, add UTC suffix
            # This is to ensure that the datetime string is in the correct format for
            # PySpark
            datetime_w_utc_tz = F.when(
                is_datetime, F.concat(datetime, F.lit(" UTC"))
            ).otherwise(F.concat(datetime, F.lit(" 00:00:00.000 UTC")))
            # Convert datetime string to timestamp
            datetime_timestamp = F.to_timestamp(datetime_w_utc_tz)
            # Convert timestamp to unix timestamp
            unix_timestamp_wo_milliseconds = F.unix_timestamp(datetime_timestamp)
            # Extract milliseconds from the datetime string
            milliseconds_str = F.date_format(datetime_timestamp, "SSS")
            # Convert milliseconds to float
            milliseconds_float = milliseconds_str.cast("float") / 1000.0
            # Add milliseconds to the unix timestamp if we have them
            return unix_timestamp_wo_milliseconds + milliseconds_float

        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: datetime_to_unix_timestamp(x)
            if self.getUnit() == "s"
            else datetime_to_unix_timestamp(x) * 1000.0,
        )
        return dataset.withColumn(self.getOutputCol(), output_col)
