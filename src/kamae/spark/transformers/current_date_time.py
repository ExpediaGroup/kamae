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
from pyspark.sql.types import DataType

from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.transformers.base import BaseTransformer
from kamae.spark.utils import single_input_single_output_scalar_transform
from kamae.tensorflow.layers import CurrentDateTimeLayer


class CurrentDateTimeTransformer(BaseTransformer, SingleInputSingleOutputParams):
    """
    Returns the current UTC datetime in yyyy-MM-dd HH:mm:ss.SSS format.

    NOTE: Parity between this and its TensorFlow counterpart is very difficult at the
    millisecond level. We have to round the TensorFlow timestamp to the 3rd decimal
    place for milliseconds, because  Spark already truncates to 3 decimal places.
    Therefore, parity is not guaranteed at this precision.

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
    ) -> None:
        """
        Initialises the CurrentDateTime layer.

        :param inputCol: Input column name.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
        in the keras model. If not set, we use the uid of the Spark transformer.
        :returns: None - class instantiated.
        """
        super().__init__()
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
        return None

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Returns a column of the current datetime. If an array column is provided,
        we return an array column of identical structure with elements populated by
        the current datetime

        :param dataset: Input dataframe.
        :returns: Transformed dataframe.
        """

        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )

        def current_utc_timestamp() -> Column:
            """
            Returns the current UTC timestamp. Spark respects the timezone
            set in the Spark session for datetime operations, so we need to be agnostic
            to the timezone set by the user.

            :returns: Column of the current UTC timestamp.
            """
            local_timestamp = F.localtimestamp()
            # TODO: Replace this with current_timezone() once we only support PySpark
            #  3.5+
            local_timezone = self.spark.conf.get("spark.sql.session.timeZone")
            return F.date_format(
                F.to_utc_timestamp(local_timestamp, local_timezone),
                "yyyy-MM-dd HH:mm:ss.SSS",
            )

        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: current_utc_timestamp(),
        )

        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        """
        Gets the tensorflow layer.

        :returns: CurrentDateTimeLayer Tensorflow layer.
        """
        return CurrentDateTimeLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
        )
