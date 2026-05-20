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

"""
TensorFlow-specific Keras layers.

These layers use TensorFlow-specific operations and are the canonical location
for TF-only layers. All layers use the unified BaseLayer from kamae.keras.core.base.
"""

from .bloom_encode import BloomEncodeLayer  # noqa: F401
from .bucketize import BucketizeLayer  # noqa: F401
from .current_date import CurrentDateLayer  # noqa: F401
from .current_date_time import CurrentDateTimeLayer  # noqa: F401
from .current_unix_timestamp import CurrentUnixTimestampLayer  # noqa: F401
from .date_add import DateAddLayer  # noqa: F401
from .date_diff import DateDiffLayer  # noqa: F401
from .date_parse import DateParseLayer  # noqa: F401
from .date_time_to_unix_timestamp import DateTimeToUnixTimestampLayer  # noqa: F401
from .hash_index import HashIndexLayer  # noqa: F401
from .if_statement import IfStatementLayer  # noqa: F401
from .lambda_function import LambdaFunctionLayer  # noqa: F401
from .list_max import ListMaxLayer  # noqa: F401
from .list_mean import ListMeanLayer  # noqa: F401
from .list_median import ListMedianLayer  # noqa: F401
from .list_min import ListMinLayer  # noqa: F401
from .list_rank import ListRankLayer  # noqa: F401
from .list_std_dev import ListStdDevLayer  # noqa: F401
from .min_hash_index import MinHashIndexLayer  # noqa: F401
from .one_hot_encode import OneHotEncodeLayer  # noqa: F401
from .ordinal_array_encode import OrdinalArrayEncodeLayer  # noqa: F401
from .string_affix import StringAffixLayer  # noqa: F401
from .string_array_constant import StringArrayConstantLayer  # noqa: F401
from .string_case import StringCaseLayer  # noqa: F401
from .string_concatenate import StringConcatenateLayer  # noqa: F401
from .string_contains import StringContainsLayer  # noqa: F401
from .string_contains_list import StringContainsListLayer  # noqa: F401
from .string_index import StringIndexLayer  # noqa: F401
from .string_isin_list import StringIsInListLayer  # noqa: F401
from .string_list_to_string import StringListToStringLayer  # noqa: F401
from .string_map import StringMapLayer  # noqa: F401
from .string_replace import StringReplaceLayer  # noqa: F401
from .string_to_string_list import StringToStringListLayer  # noqa: F401
from .sub_string_delim_at_index import SubStringDelimAtIndexLayer  # noqa: F401
from .unix_timestamp_to_date_time import UnixTimestampToDateTimeLayer  # noqa: F401

__all__ = [
    "BloomEncodeLayer",
    "BucketizeLayer",
    "CurrentDateLayer",
    "CurrentDateTimeLayer",
    "CurrentUnixTimestampLayer",
    "DateAddLayer",
    "DateDiffLayer",
    "DateParseLayer",
    "DateTimeToUnixTimestampLayer",
    "HashIndexLayer",
    "IfStatementLayer",
    "LambdaFunctionLayer",
    "ListMaxLayer",
    "ListMeanLayer",
    "ListMedianLayer",
    "ListMinLayer",
    "ListRankLayer",
    "ListStdDevLayer",
    "MinHashIndexLayer",
    "OneHotEncodeLayer",
    "OrdinalArrayEncodeLayer",
    "StringAffixLayer",
    "StringArrayConstantLayer",
    "StringCaseLayer",
    "StringConcatenateLayer",
    "StringContainsLayer",
    "StringContainsListLayer",
    "StringIndexLayer",
    "StringIsInListLayer",
    "StringListToStringLayer",
    "StringMapLayer",
    "StringReplaceLayer",
    "StringToStringListLayer",
    "SubStringDelimAtIndexLayer",
    "UnixTimestampToDateTimeLayer",
]
