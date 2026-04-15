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
Multi-backend shape utility functions for backend-agnostic operations.
"""

from typing import Iterable, List

from keras import ops

from kamae.keras.core.typing import Tensor


def reshape_to_equal_rank(inputs: Iterable[Tensor]) -> List[Tensor]:
    """
    Reshapes the input tensors to match the rank of the largest tensor.

    This is a backend-agnostic version using keras.ops.

    :param inputs: The input tensors to reshape.
    :return: The reshaped input tensors.
    """
    max_rank = max([len(tensor.shape) for tensor in inputs])
    reshaped_inputs = []
    for x in inputs:
        rank_diff = max_rank - len(x.shape)
        if rank_diff > 0:
            # Get shape as tensor (handles both static and dynamic shapes)
            shape_tensor = ops.convert_to_tensor(ops.shape(x))
            reshape_dim = ops.concatenate(
                [
                    shape_tensor[:-1],
                    ops.ones(rank_diff, dtype="int32"),
                    shape_tensor[-1:],
                ],
                axis=0,
            )
            x = ops.reshape(x, reshape_dim)
        reshaped_inputs.append(x)
    return reshaped_inputs
