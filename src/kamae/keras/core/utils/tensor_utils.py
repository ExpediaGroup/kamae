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
Portable tensor utility functions for backend-agnostic operations.
"""

from typing import Any, List, Union

import numpy as np
from keras import ops


def listify_tensors(x: Union[Any, np.ndarray, List[Any]]) -> List[Any]:
    """
    Converts any tensors or numpy arrays to lists for config serialization.

    Works with any backend (TensorFlow, JAX, PyTorch).

    :param x: The input tensor or numpy array.
    :returns: The input as a list.
    """
    # Check if it's a tensor using ops.is_tensor (works across backends)
    if hasattr(x, "numpy"):
        # Most backend tensors have a .numpy() method
        x = x.numpy()
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return x
