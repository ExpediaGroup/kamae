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
Backend-agnostic Keras layers.

Portable layers that work across TensorFlow, JAX, and PyTorch backends.
"""

from .absolute_value import AbsoluteValueLayer
from .base import BaseLayer
from .exp import ExpLayer
from .identity import IdentityLayer
from .log import LogLayer
from .multiply import MultiplyLayer

__all__ = [
    "BaseLayer",
    "IdentityLayer",
    "AbsoluteValueLayer",
    "MultiplyLayer",
    "ExpLayer",
    "LogLayer",
]
