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

import pytest
from pyspark.ml.param import TypeConverters
from pyspark.sql.types import FloatType, StringType

from kamae.keras.core.base import BaseLayer
from kamae.params import _UNSET, ParamSpec
from kamae.params.shared_specs import MASK_VALUE_PARAMS
from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.params.param_spec import _camel_to_snake, build_keras_layer_from_specs
from kamae.spark.transformers.base import BaseTransformer


class _StubKerasLayer(BaseLayer):
    _compatible_dtypes = None
    _params = {
        "my_value": ParamSpec(default=None),
        "mask_value": ParamSpec(default=None),
    }

    def _call(self, inputs, **kwargs):
        return inputs[0]


def _validate_positive(value):
    if value is not None and value <= 0:
        raise ValueError("must be positive")
    return value


def _validate_with_self(self, value):
    if self.isDefined("inputCol") and value is not None and value < 0:
        raise ValueError("negative not allowed when inputCol set")
    return value


class _BasicTransformer(BaseTransformer, SingleInputSingleOutputParams):
    _compatible_dtypes = [FloatType()]
    _keras_layer_class = _StubKerasLayer
    _params = {
        "myValue": ParamSpec(
            spark_typeconverter=TypeConverters.toFloat, default=1.0, doc="A value"
        ),
    }

    def _transform(self, dataset):
        return dataset


class _ValidatedTransformer(BaseTransformer, SingleInputSingleOutputParams):
    _compatible_dtypes = None
    _keras_layer_class = _StubKerasLayer
    _params = {
        "threshold": ParamSpec(
            default=_UNSET,
            doc="Threshold",
            spark_typeconverter=TypeConverters.toFloat,
            validator=_validate_positive,
        ),
    }

    def _transform(self, dataset):
        return dataset


class _SelfValidatedTransformer(BaseTransformer, SingleInputSingleOutputParams):
    _compatible_dtypes = None
    _keras_layer_class = _StubKerasLayer
    _params = {
        "offset": ParamSpec(
            default=_UNSET,
            doc="Offset",
            spark_typeconverter=TypeConverters.toFloat,
            validator=_validate_with_self,
        ),
    }

    def _transform(self, dataset):
        return dataset


class _SharedDictTransformer(BaseTransformer, SingleInputSingleOutputParams):
    _compatible_dtypes = [StringType()]
    _keras_layer_class = _StubKerasLayer
    _params = {
        **MASK_VALUE_PARAMS,
        "extra": ParamSpec(
            spark_typeconverter=TypeConverters.toString,
            default="default",
            doc="Extra param",
        ),
    }

    def _transform(self, dataset):
        return dataset


class _NoParamsTransformer(BaseTransformer, SingleInputSingleOutputParams):
    _compatible_dtypes = [FloatType()]
    _keras_layer_class = _StubKerasLayer

    def _transform(self, dataset):
        return dataset


class TestSparkCodegenParams:
    def test_getter_setter_generated(self):
        t = _BasicTransformer()
        assert hasattr(t, "getMyValue")
        assert hasattr(t, "setMyValue")
        t.setMyValue(42.0)
        assert t.getMyValue() == 42.0

    def test_default_value(self):
        t = _BasicTransformer()
        assert t.getMyValue() == 1.0

    def test_unset_default_is_none(self):
        t = _ValidatedTransformer()
        assert t.getThreshold() is None

    def test_compatible_dtypes_property(self):
        t = _BasicTransformer()
        assert t.compatible_dtypes == [FloatType()]

    def test_no_params_still_constructs(self):
        t = _NoParamsTransformer()
        assert t.compatible_dtypes == [FloatType()]


class TestSparkCodegenValidators:
    def test_single_arg_validator_rejects(self):
        t = _ValidatedTransformer()
        with pytest.raises(ValueError, match="must be positive"):
            t.setThreshold(-1.0)

    def test_single_arg_validator_accepts(self):
        t = _ValidatedTransformer()
        t.setThreshold(5.0)
        assert t.getThreshold() == 5.0

    def test_two_arg_validator_uses_self(self):
        t = _SelfValidatedTransformer(inputCol="x")
        with pytest.raises(ValueError, match="negative not allowed"):
            t.setOffset(-1.0)

    def test_two_arg_validator_passes_without_inputcol(self):
        t = _SelfValidatedTransformer()
        t.setOffset(-1.0)
        assert t.getOffset() == -1.0


class TestSparkCodegenInit:
    def test_init_kwargs(self):
        t = _BasicTransformer(inputCol="x", outputCol="y", myValue=3.14)
        assert t.getInputCol() == "x"
        assert t.getOutputCol() == "y"
        assert t.getMyValue() == 3.14

    def test_init_defaults(self):
        t = _BasicTransformer()
        assert t.getMyValue() == 1.0
        assert t.getLayerName() == t.uid


class TestSparkCodegenKerasLayer:
    def test_keras_layer_class_generates_get_keras_layer(self):
        t = _BasicTransformer(
            inputCol="x",
            outputCol="y",
            layerName="test_layer",
            myValue=99.0,
        )
        layer = t.get_keras_layer()
        assert isinstance(layer, _StubKerasLayer)
        assert layer.name == "test_layer"
        assert layer.my_value == 99.0


class TestSparkCodegenSharedDicts:
    def test_shared_dict_params_accessible(self):
        t = _SharedDictTransformer()
        assert hasattr(t, "getMaskValue")
        assert hasattr(t, "setMaskValue")
        t.setMaskValue(0.0)
        assert t.getMaskValue() == 0.0

    def test_shared_and_custom_coexist(self):
        t = _SharedDictTransformer(maskValue=1.5, extra="custom")
        assert t.getMaskValue() == 1.5
        assert t.getExtra() == "custom"


class TestCamelToSnake:
    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("myValue", "my_value"),
            ("numOOVIndices", "num_oov_indices"),
            ("maskToken", "mask_token"),
            ("simple", "simple"),
            ("HTMLParser", "html_parser"),
        ],
    )
    def test_conversion(self, input_val, expected):
        assert _camel_to_snake(input_val) == expected
