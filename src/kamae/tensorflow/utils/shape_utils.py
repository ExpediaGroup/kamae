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

from typing import Iterable, List

import tensorflow as tf

from kamae.tensorflow.typing import Tensor


def reshape_to_equal_rank(inputs: Iterable[Tensor]) -> List[Tensor]:
    """
    Reshapes the input tensors to match the rank of the largest tensor.

    :param inputs: The input tensors to reshape.
    :return: The reshaped input tensors.
    """
    max_rank = max([len(tensor.shape) for tensor in inputs])
    reshaped_inputs = []
    for x in inputs:
        rank_diff = max_rank - len(x.shape)
        if rank_diff > 0:
            reshape_dim = tf.concat(
                [
                    tf.shape(x)[:-1],
                    tf.ones(rank_diff, dtype=tf.int32),
                    tf.shape(x)[-1:],
                ],
                axis=0,
            )
            x = tf.reshape(x, reshape_dim)
        reshaped_inputs.append(x)
    return reshaped_inputs
