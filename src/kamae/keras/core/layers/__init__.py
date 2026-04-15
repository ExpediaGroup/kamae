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

Multi-backend layers that work across TensorFlow, JAX, and PyTorch backends.
"""

from .absolute_value import AbsoluteValueLayer
from .array_concatenate import ArrayConcatenateLayer
from .array_crop import ArrayCropLayer
from .array_split import ArraySplitLayer
from .array_subtract_minimum import ArraySubtractMinimumLayer
from .bearing_angle import BearingAngleLayer
from .bin import BinLayer
from .conditional_standard_scale import ConditionalStandardScaleLayer
from .cosine_similarity import CosineSimilarityLayer
from .divide import DivideLayer
from .exp import ExpLayer
from .exponent import ExponentLayer
from .haversine_distance import HaversineDistanceLayer
from .identity import IdentityLayer
from .impute import ImputeLayer
from .log import LogLayer
from .logical_and import LogicalAndLayer
from .logical_not import LogicalNotLayer
from .logical_or import LogicalOrLayer
from .max import MaxLayer
from .mean import MeanLayer
from .min import MinLayer
from .min_max_scale import MinMaxScaleLayer
from .modulo import ModuloLayer
from .multiply import MultiplyLayer
from .numerical_if_statement import NumericalIfStatementLayer
from .round import RoundLayer
from .round_to_decimal import RoundToDecimalLayer
from .standard_scale import StandardScaleLayer
from .subtract import SubtractLayer
from .sum import SumLayer

__all__ = [
    "IdentityLayer",
    "AbsoluteValueLayer",
    "MultiplyLayer",
    "ExpLayer",
    "LogLayer",
    "DivideLayer",
    "SubtractLayer",
    "RoundLayer",
    "RoundToDecimalLayer",
    "ModuloLayer",
    "SumLayer",
    "MaxLayer",
    "MinLayer",
    "MeanLayer",
    "ExponentLayer",
    "LogicalAndLayer",
    "LogicalOrLayer",
    "LogicalNotLayer",
    "NumericalIfStatementLayer",
    "ArrayConcatenateLayer",
    "ArraySplitLayer",
    "ArrayCropLayer",
    "ArraySubtractMinimumLayer",
    "StandardScaleLayer",
    "ConditionalStandardScaleLayer",
    "MinMaxScaleLayer",
    "ImputeLayer",
    "BinLayer",
    "BearingAngleLayer",
    "CosineSimilarityLayer",
    "HaversineDistanceLayer",
]
