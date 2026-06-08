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
kamae

kamae is a Python package comprising a set of reusable Keras
transformation layers.
"""

__version__ = "2.40.1"
__name__ = "kamae"

from .discovery import (  # noqa: F401
    get_compatible_layers,
    get_compatible_transformers,
    get_jit_compatible_layers,
    get_jit_compatible_transformers,
)
