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

"""Tests for JIT compatibility attributes on Spark estimators and transformers."""

import inspect

from pyspark.ml import Estimator, Transformer

import kamae.spark.estimators as estimators_mod
import kamae.spark.transformers as transformers_mod


def test_all_spark_operations_have_jit_compatible_attribute():
    """Test that all Spark transformers and estimators have jit_compatible attribute."""
    # Get all transformer classes
    transformers = [
        obj
        for name, obj in vars(transformers_mod).items()
        if isinstance(obj, type)
        and issubclass(obj, Transformer)
        and obj is not Transformer
        and name != "BaseTransformer"  # Exclude base class
    ]

    # Get all estimator classes
    estimators = [
        obj
        for name, obj in vars(estimators_mod).items()
        if isinstance(obj, type)
        and issubclass(obj, Estimator)
        and obj is not Estimator
        and name != "BaseEstimator"  # Exclude base class
    ]

    all_operations = transformers + estimators

    for op_cls in all_operations:
        assert hasattr(
            op_cls, "jit_compatible"
        ), f"{op_cls.__name__} missing jit_compatible attribute"
        assert isinstance(
            op_cls.jit_compatible, bool
        ), f"{op_cls.__name__}.jit_compatible must be bool, got {type(op_cls.jit_compatible)}"
