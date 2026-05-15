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

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.tensorflow.layers import CurrentUnixTimestampLayer
from kamae.params.shared_specs import UNIX_TIMESTAMP_PARAMS
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.transformers.base import BaseTransformer
from kamae.spark.utils import single_input_single_output_scalar_transform


class CurrentUnixTimestampTransformer(BaseTransformer, SingleInputSingleOutputParams):
    """
    Returns the current unix timestamp in either seconds or milliseconds.

    NOTE: Parity between this and its TensorFlow counterpart is very difficult at the
    millisecond level. TensorFlow provides much more precision of the timestamp,
    and has floating 64-bit precision of the unix timestamp in seconds.
    Whereas Spark 3.4.0 only supports millisecond precision (3 decimal places of unix
    timestamp in seconds). Therefore, parity is not guaranteed at this precision.

    It is recommended not to rely on parity at the millisecond level.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = None
    _keras_layer_class = CurrentUnixTimestampLayer
    _params = {**UNIX_TIMESTAMP_PARAMS}

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Returns a column of the current unix timestamp. If an array column is provided,
        we return an array column of identical structure with elements populated by
        the current unix timestamp.

        :param dataset: Input dataframe.
        :returns: Transformed dataframe.
        """
        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )

        def current_unix_timestamp() -> Column:
            """
            Returns the current unix timestamp in either seconds or milliseconds.

            :returns: Column of the current unix timestamp.
            """
            # TODO: For PySpark 3.5+ we can use unix_millis. For now, we use
            #  unix_timestamp that returns seconds (truncated so no milliseconds).
            #  In order to get milliseconds, we get the milliseconds from the current
            #  timestamp and add it to the truncated seconds.
            current_ts = F.current_timestamp()
            unix_timestamp_in_trucated_seconds = F.unix_timestamp(current_ts)
            milliseconds_str = F.date_format(current_ts, "SSS")
            milliseconds_float = milliseconds_str.cast("float") / 1000.0
            unix_timestamp_in_seconds = (
                unix_timestamp_in_trucated_seconds + milliseconds_float
            )
            return (
                unix_timestamp_in_seconds
                if self.getUnit() == "s"
                else unix_timestamp_in_seconds * 1000.0
            )

        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: current_unix_timestamp(),
        )

        return dataset.withColumn(self.getOutputCol(), output_col)
