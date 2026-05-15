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

from dataclasses import dataclass
from typing import Any, Callable, Optional

# Sentinel: "this param is required — must be passed to __init__"
_REQUIRED = object()

# Sentinel: "optional param with no sensible default"
_UNSET = object()


@dataclass
class ParamSpec:
    """Unified declarative specification for a parameter shared across Spark
    transformers and Keras layers.

    Attributes:
        default: Default value. Use ``_REQUIRED`` for params that must be
            provided at construction time, ``_UNSET`` for optional params
            with no sensible default (Spark-side only).
        doc: Description string.
        spark_typeconverter: Explicit PySpark TypeConverter. Only used by
            Spark codegen; ignored on the Keras side.
        validator: Optional callable ``(value) -> value`` or
            ``(self, value) -> value`` invoked in setters (Spark) and
            during ``__init__`` (Keras).
    """

    default: Any
    doc: str = ""
    spark_typeconverter: Optional[Callable] = None
    validator: Optional[Callable] = None
