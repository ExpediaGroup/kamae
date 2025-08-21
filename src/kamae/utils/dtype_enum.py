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

from enum import Enum
from typing import Any, Dict

import tensorflow as tf
from pyspark.sql.types import (
    BooleanType,
    ByteType,
    DataType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
)


class DType(Enum):
    """
    Enum class for supported data types in Kamae.
    Contains a string name, the corresponding Spark data type, the corresponding
    TensorFlow data type, and the number of bytes the data type takes up.
    String is a special case, as it can be of any length, so the number of bytes
    is set to 0.
    """

    STRING = (
        "string",
        StringType(),
        tf.string,
        0,
        False,
        False,
    )  # String can be of any length
    BIGINT = ("bigint", LongType(), tf.int64, 8, False, True)
    INT = ("int", IntegerType(), tf.int32, 4, False, True)
    SMALLINT = ("smallint", ShortType(), tf.int16, 2, False, True)
    TINYINT = ("tinyint", ByteType(), tf.int8, 1, False, True)
    FLOAT = ("float", FloatType(), tf.float32, 4, True, False)
    DOUBLE = ("double", DoubleType(), tf.float64, 8, True, False)
    BOOLEAN = ("boolean", BooleanType(), tf.bool, 1, False, False)

    def __init__(
        self,
        dtype_name: str,
        spark_dtype: DataType,
        tf_dtype: tf.dtypes.DType,
        bytes: int,
        is_floating: bool = False,
        is_integer: bool = False,
    ) -> None:
        self.dtype_name = dtype_name
        self.spark_dtype = spark_dtype
        self.tf_dtype = tf_dtype
        self.bytes = bytes
        self.is_floating = is_floating
        self.is_integer = is_integer

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dtype_name": self.dtype_name,
            "spark_dtype": self.spark_dtype,
            "tf_dtype": self.tf_dtype,
            "bytes": self.bytes,
            "is_floating": self.is_floating,
            "is_integer": self.is_integer,
        }
