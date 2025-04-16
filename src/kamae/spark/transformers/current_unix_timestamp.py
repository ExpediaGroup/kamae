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
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import DataType

from kamae.spark.params import SingleInputSingleOutputParams, UnixTimestampParams
from kamae.spark.transformers.base import BaseTransformer
from kamae.spark.utils import single_input_single_output_scalar_transform
from kamae.tensorflow.layers import CurrentUnixTimestampLayer


class CurrentUnixTimestampTransformer(
    BaseTransformer, SingleInputSingleOutputParams, UnixTimestampParams
):
    """
    Returns the current unix timestamp in either seconds or milliseconds.

    NOTE: Parity between this and its TensorFlow counterpart is very difficult at the
    millisecond level. TensorFlow provides much more precision of the timestamp,
    and has floating 64-bit precision of the unix timestamp in seconds.
    Whereas Spark 3.4.0 only supports millisecond precision (3 decimal places of unix
    timestamp in seconds). Therefore, parity is not guaranteed at this precision.

    It is recommended not to rely on parity at the millisecond level.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        unit: str = "s",
    ) -> None:
        """
        Initialises the CurrentUnixTimestamp layer.

        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :param unit: Unit of the output timestamp. Can be either "s" (or "seconds")
        for seconds or "ms" (or "milliseconds") for milliseconds. Defaults to "s".
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(unit="s")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return None

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

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer.

        :returns: CurrentUnixTimestampLayer Tensorflow layer.
        """
        return CurrentUnixTimestampLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            unit=self.getUnit(),
        )
