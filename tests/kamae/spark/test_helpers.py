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


def tensor_to_python_type(x):
    if x.dtype.name == "string":
        return x.numpy().decode("utf-8")
    elif x.dtype.name == "bool":
        return bool(x.numpy())
    elif x.dtype.is_floating:
        return float(x.numpy())
    elif x.dtype.is_integer:
        return int(x.numpy())
    else:
        raise ValueError(f"Unknown dtype: {x.dtype}")
