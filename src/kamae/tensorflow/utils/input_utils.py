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

"""Provides utilities for tensorflow layer inputs"""
from typing import Callable, Iterable, List, Union

import tensorflow as tf

from kamae.tensorflow.typing import Tensor


def iter_values(x) -> Iterable:
    """
    Returns an iterator over the values of a generic iterator.
    Will be used to construct lists from iterables such as lists, tuples, dicts, etc.

    :param x: An iterable
    :returns: An iterator over the values of the iterable.
    """
    if hasattr(x, "itervalues"):
        return x.itervalues()
    if hasattr(x, "values"):
        return iter(x.values())
    return iter(x)


def enforce_single_tensor_input(layer_call_method: Callable) -> Callable:
    """
    Enforces that the inputs to a layer are a single tensor. If the inputs are an
    iterable, then we check it has a single element and that the element is a tensor.
    If the inputs are a tensor, then we return the tensor.

    :param layer_call_method: The layer's call method to decorate.
    :raises TypeError: If the inputs are an iterable with more than one element.
    :returns: The function called with a single tensor.
    """

    def _enforce_single_tensor_input(
        self,
        inputs: Union[Tensor, Iterable[Tensor]],
        **kwargs,
    ) -> Tensor:
        if tf.is_tensor(inputs):
            # If the inputs are a tensor, then we return the tensor.
            processed_inputs = inputs
        else:
            input_list = list(iter_values(inputs))
            if len(input_list) == 1 and tf.is_tensor(input_list[0]):
                # If the inputs are an iterable with a single tensor,
                # then we return the tensor.
                processed_inputs = input_list[0]
            else:
                # Otherwise, we raise an error.
                raise ValueError(
                    f"""Expected inputs to be a single tensor, but got a list of
                    {len(input_list)} tensors."""
                )
        return layer_call_method(self, processed_inputs, **kwargs)

    return _enforce_single_tensor_input


def enforce_multiple_tensor_input(layer_call_method: Callable) -> Callable:
    """
    Enforces that the inputs to a layer are an iterable of tensors.
    We check that all elements are tensors. If the inputs are a single tensor, rather
    than an iterable we raise an error.

    :param layer_call_method: The layer's call method to decorate.
    :raises TypeError: If the inputs are a single tensor, an iterable of length 1
    or an iterable of non-tensors.
    :returns: The function called with a list of tensors.
    """

    def _enforce_multiple_tensor_input(
        self,
        inputs: Union[Tensor, Iterable[Tensor]],
        **kwargs,
    ) -> List[Tensor]:
        if tf.is_tensor(inputs):
            raise ValueError(
                """Expected inputs to be a iterable of tensors,
                but got a single tensor."""
            )
        else:
            input_list = list(iter_values(inputs))
            if len(input_list) > 1 and all(
                [tf.is_tensor(input_tensor) for input_tensor in input_list]
            ):
                processed_inputs = input_list
            else:
                raise ValueError(
                    """Invalid inputs. Expected inputs to be an iterable of tensors,
                    but either got an iterable of non-tensors or a single tensor."""
                )
        return layer_call_method(self, processed_inputs, **kwargs)

    return _enforce_multiple_tensor_input


def allow_single_or_multiple_tensor_input(layer_call_method: Callable) -> Callable:
    """
    Enforces that the inputs to a layer are either a single tensor or a list of tensors.
    If the inputs are an iterable, then we check that all elements are tensors. If the
    inputs are a tensor, then we return a list containing the tensor.

    :param layer_call_method: The layer's call method to decorate.
    :returns: The function called with a list of tensors.
    """

    def _allow_single_or_multiple_tensor_input(
        self,
        inputs: Union[Tensor, Iterable[Tensor]],
        **kwargs,
    ) -> List[Tensor]:
        if tf.is_tensor(inputs):
            processed_inputs = [inputs]
        else:
            input_list = list(iter_values(inputs))
            if all([tf.is_tensor(input_tensor) for input_tensor in input_list]):
                processed_inputs = input_list
            else:
                raise ValueError(
                    """All elements of the inputs must be tensors, but got an iterable
                    containing non-tensors."""
                )
        return layer_call_method(self, processed_inputs, **kwargs)

    return _allow_single_or_multiple_tensor_input
