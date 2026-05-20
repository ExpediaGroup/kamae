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
Backend detection and enforcement utilities for Keras 3 multi-backend support.
"""

from typing import FrozenSet

import keras

ALL_BACKENDS: FrozenSet[str] = frozenset({"tensorflow", "jax", "torch"})
TENSORFLOW_ONLY: FrozenSet[str] = frozenset({"tensorflow"})


def current_backend() -> str:
    """
    Returns the current Keras backend.

    :returns: Backend name: 'tensorflow', 'jax', or 'torch'
    """
    return keras.backend.backend()


def require_tensorflow() -> None:
    """
    Raises RuntimeError if not running on TensorFlow backend.

    This should be called in the __init__ of TensorFlow-only layers
    to fail fast with a clear error message.

    :raises RuntimeError: If current backend is not TensorFlow
    """
    backend = current_backend()
    if backend != "tensorflow":
        raise RuntimeError(
            f"This layer requires TensorFlow backend. "
            f"Current backend: {backend}. "
            f"Set KERAS_BACKEND=tensorflow before importing keras."
        )


def validate_backend(class_name: str, supported_backends: FrozenSet[str]) -> None:
    """
    Validates that the current backend is supported by the layer/operation.

    :param class_name: Name of the class being validated
    :param supported_backends: Frozenset of supported backend names
    :raises RuntimeError: If current backend is not in supported_backends
    """
    backend = current_backend()
    if backend not in supported_backends:
        raise RuntimeError(
            f"{class_name} requires one of {sorted(supported_backends)} backends. "
            f"Current backend: '{backend}'. "
            f"Set KERAS_BACKEND=tensorflow before importing keras."
        )
