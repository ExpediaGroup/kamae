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

from operator import eq, ge, gt, le, lt, ne
from typing import Any, Callable


def get_condition_operator(cond_op_string: str) -> Callable[[Any, Any], Any]:
    """
    Translates a string condition operator to a function operator.

    :returns: Function operator.
    """
    allowed_cond_ops = {
        "eq": eq,
        "neq": ne,
        "lt": lt,
        "leq": le,
        "gt": gt,
        "geq": ge,
    }
    try:
        return allowed_cond_ops[cond_op_string]
    except KeyError:
        raise ValueError(
            f"""Unknown condition operator: {cond_op_string}.
            Allowed condition operators are: {allowed_cond_ops.keys()}"""
        )
