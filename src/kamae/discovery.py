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
Discovery utilities for finding backend-compatible layers and transformers.
"""

import inspect
from typing import Any, Callable, Dict, Union

import kamae.keras.core.layers as core_layers
import kamae.keras.tensorflow.layers as tf_layers
import kamae.spark.estimators as estimators
import kamae.spark.transformers as transformers
from kamae.keras.core.backend import ALL_BACKENDS, current_backend
from kamae.keras.core.base import BaseLayer
from kamae.spark.estimators.base import BaseEstimator
from kamae.spark.transformers.base import BaseTransformer


def _inspect_modules(
    modules: list[Any], attribute: str, condition: Callable[[Any], bool]
) -> Dict[str, type]:
    """
    Helper to inspect multiple modules for classes matching a condition.

    :param modules: List of modules to inspect
    :param attribute: Attribute name to check on each class
    :param condition: Function that returns True if the attribute value matches
    :returns: Dict mapping class names to class objects
    """
    compatible = {}
    for module in modules:
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if hasattr(obj, attribute) and condition(getattr(obj, attribute)):
                compatible[name] = obj
    return compatible


def get_compatible_layers(backend: str = None) -> Dict[str, type[BaseLayer]]:
    """
    Returns a dict of Keras layer classes compatible with the specified backend.

    :param backend: Backend name ('tensorflow', 'jax', or 'torch'). If None, uses
        the current backend.
    :returns: Dict mapping layer names to layer class objects that work on the
        specified backend.
    :raises ValueError: If backend name is invalid.

    Example:
        >>> from kamae.discovery import get_compatible_layers
        >>> # Get layers that work on JAX
        >>> jax_layers = get_compatible_layers('jax')
        >>> # Instantiate a layer by name
        >>> layer = jax_layers['MultiplyLayer'](multiplier=2.0)
        >>> # List available layer names
        >>> print(list(jax_layers.keys()))
    """
    if backend is None:
        backend = current_backend()

    if backend not in ALL_BACKENDS:
        raise ValueError(
            f"Invalid backend '{backend}'. Must be one of {sorted(ALL_BACKENDS)}"
        )

    return _inspect_modules(
        modules=[core_layers, tf_layers],
        attribute="supported_backends",
        condition=lambda backends: backend in backends,
    )


def get_compatible_transformers(
    backend: str = None,
) -> Dict[str, Union[type[BaseTransformer], type[BaseEstimator]]]:
    """
    Returns a dict of Spark transformer/estimator classes compatible with the
    specified backend.

    :param backend: Backend name ('tensorflow', 'jax', or 'torch'). If None, uses
        the current backend.
    :returns: Dict mapping transformer/estimator names to class objects that work
        on the specified backend.
    :raises ValueError: If backend name is invalid.

    Example:
        >>> from kamae.discovery import get_compatible_transformers
        >>> # Get transformers that work on PyTorch
        >>> torch_transformers = get_compatible_transformers('torch')
        >>> # Instantiate a transformer by name
        >>> transformer = torch_transformers['LogTransformer'](inputCol="x", outputCol="y")
        >>> # List available transformer names
        >>> print(list(torch_transformers.keys()))
    """
    if backend is None:
        backend = current_backend()

    if backend not in ALL_BACKENDS:
        raise ValueError(
            f"Invalid backend '{backend}'. Must be one of {sorted(ALL_BACKENDS)}"
        )

    return _inspect_modules(
        modules=[transformers, estimators],
        attribute="supported_backends",
        condition=lambda backends: backend in backends,
    )


def get_jit_compatible_layers() -> Dict[str, type[BaseLayer]]:
    """
    Returns a dict of Keras layer classes that are JIT-compatible.

    JIT-compatible layers can be compiled with @tf.function or jax.jit for improved
    performance.

    :returns: Dict mapping layer names to JIT-compatible layer class objects.

    Example:
        >>> from kamae.discovery import get_jit_compatible_layers
        >>> jit_layers = get_jit_compatible_layers()
        >>> # Instantiate a JIT-compatible layer by name
        >>> layer = jit_layers['MultiplyLayer'](multiplier=2.0)
        >>> # See how many JIT-compatible layers exist
        >>> print(f"Found {len(jit_layers)} JIT-compatible layers")
    """
    return _inspect_modules(
        modules=[core_layers, tf_layers],
        attribute="jit_compatible",
        condition=lambda jit: jit is True,
    )


def get_jit_compatible_transformers() -> (
    Dict[str, Union[type[BaseTransformer], type[BaseEstimator]]]
):
    """
    Returns a dict of Spark transformer/estimator classes that are JIT-compatible.

    JIT-compatible transformers generate Keras layers that can be compiled with
    @tf.function or jax.jit for improved performance.

    :returns: Dict mapping transformer/estimator names to JIT-compatible class objects.

    Example:
        >>> from kamae.discovery import get_jit_compatible_transformers
        >>> jit_transformers = get_jit_compatible_transformers()
        >>> # Instantiate a JIT-compatible transformer by name
        >>> transformer = jit_transformers['LogTransformer'](inputCol="x", outputCol="y")
        >>> # See all JIT-compatible transformer names
        >>> print(list(jit_transformers.keys()))
    """
    return _inspect_modules(
        modules=[transformers, estimators],
        attribute="jit_compatible",
        condition=lambda jit: jit is True,
    )
