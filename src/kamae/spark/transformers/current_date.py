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
from typing import Optional

import pyspark.sql.functions as F
from pyspark import keyword_only
from pyspark.sql import Column, DataFrame, SparkSession

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.tensorflow.layers import CurrentDateLayer
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.transformers.base import BaseTransformer
from kamae.spark.utils import single_input_single_output_scalar_transform


class CurrentDateTransformer(BaseTransformer, SingleInputSingleOutputParams):
    """
    Returns the current UTC date in yyyy-MM-dd format.
    """

    supported_backends = TENSORFLOW_ONLY

    _compatible_dtypes = None
    _keras_layer_class = CurrentDateLayer

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
        Returns a column of the current date. If an array column is provided,
        we return an array column of identical structure with elements populated by
        the current date.

        :param dataset: Input dataframe.
        :returns: Transformed dataframe.
        """

        input_datatype = self.get_column_datatype(
            dataset=dataset, column_name=self.getInputCol()
        )

        def current_utc_date() -> Column:
            """
            Returns the current UTC date. Spark respects the timezone set in the Spark
            session so we need to convert the local timestamp to UTC before extracting
            the date.

            :returns: Column of the current UTC date.
            """
            local_timestamp = F.localtimestamp()
            # TODO: Replace this with current_timezone() once we only support PySpark
            #  3.5+
            local_timezone = self.spark.conf.get("spark.sql.session.timeZone")
            return F.to_date(F.to_utc_timestamp(local_timestamp, local_timezone))

        output_col = single_input_single_output_scalar_transform(
            input_col=F.col(self.getInputCol()),
            input_col_datatype=input_datatype,
            func=lambda x: current_utc_date().cast("string"),
        )

        return dataset.withColumn(self.getOutputCol(), output_col)
