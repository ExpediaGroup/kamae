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
TensorFlow-specific utilities for TF-only layers.

These utilities use TensorFlow-specific operations and are only available
when using the TensorFlow backend.
"""

from .date_utils import (  # noqa: F401
    datetime_add_days,
    datetime_day,
    datetime_day_of_year,
    datetime_hour,
    datetime_is_weekend,
    datetime_millisecond,
    datetime_minute,
    datetime_month,
    datetime_second,
    datetime_to_unix_timestamp,
    datetime_total_days,
    datetime_total_milliseconds,
    datetime_total_seconds,
    datetime_weekday,
    datetime_year,
    unix_timestamp_to_datetime,
)
from .list_utils import get_top_n, listify_tensors, segmented_operation  # noqa: F401
from .transform_utils import map_fn_w_axis  # noqa: F401
