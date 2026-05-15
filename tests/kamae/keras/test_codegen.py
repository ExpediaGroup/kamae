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

import keras
import pytest

import kamae
from kamae.keras.core.base import BaseLayer
from kamae.params import _REQUIRED, ParamSpec


@keras.saving.register_keras_serializable(package=kamae.__name__)
class _StubLayer(BaseLayer):
    _compatible_dtypes = ["float32"]
    _params = {
        "alpha": ParamSpec(default=0.0),
        "beta": ParamSpec(default=_REQUIRED),
        "gamma": ParamSpec(default=None),
    }

    def _call(self, inputs, **kwargs):
        return inputs[0]


@keras.saving.register_keras_serializable(package=kamae.__name__)
class _ValidatedLayer(BaseLayer):
    _compatible_dtypes = None
    _params = {
        "lo": ParamSpec(default=0.0),
        "hi": ParamSpec(default=1.0),
    }

    @staticmethod
    def _post_init(self):
        if self.lo >= self.hi:
            raise ValueError("lo must be less than hi")
        self.span = self.hi - self.lo

    def _call(self, inputs, **kwargs):
        return inputs[0]


@keras.saving.register_keras_serializable(package=kamae.__name__)
class _ChildLayer(_StubLayer):
    """Subclass that inherits codegen __init__/get_config."""

    pass


class TestKerasCodegen:
    def test_params_set_as_attributes(self):
        layer = _StubLayer(alpha=1.5, beta="required_val")
        assert layer.alpha == 1.5
        assert layer.beta == "required_val"
        assert layer.gamma is None

    def test_required_param_raises(self):
        with pytest.raises(TypeError, match="beta"):
            _StubLayer(alpha=1.0)

    def test_default_values(self):
        layer = _StubLayer(beta="x")
        assert layer.alpha == 0.0
        assert layer.gamma is None

    def test_compatible_dtypes_property(self):
        layer = _StubLayer(beta="x")
        assert layer.compatible_dtypes == ["float32"]

    def test_get_config_roundtrip(self):
        layer = _StubLayer(name="test", alpha=2.5, beta=[1, 2, 3], gamma="g")
        config = layer.get_config()
        assert config["alpha"] == 2.5
        assert config["beta"] == [1, 2, 3]
        assert config["gamma"] == "g"
        restored = _StubLayer.from_config(config)
        assert restored.alpha == layer.alpha
        assert restored.beta == layer.beta

    def test_post_init_fires(self):
        with pytest.raises(ValueError, match="lo must be less than hi"):
            _ValidatedLayer(lo=5.0, hi=3.0)

    def test_post_init_sets_derived_attr(self):
        layer = _ValidatedLayer(lo=1.0, hi=4.0)
        assert layer.span == 3.0

    def test_subclass_no_recursion(self):
        layer = _ChildLayer(alpha=1.0, beta="val")
        config = layer.get_config()
        assert config["alpha"] == 1.0
        restored = _ChildLayer.from_config(config)
        assert restored.beta == "val"
