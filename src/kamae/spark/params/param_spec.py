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

import inspect
from typing import Any, Callable, Dict, Optional, Type

from pyspark.ml.param import Param, Params, TypeConverters

from kamae.params import _UNSET, ParamSpec, _camel_to_snake  # noqa: F401


def install_param_specs(
    cls: type,
    param_specs: Dict[str, ParamSpec],
) -> None:
    """Attach ``Param`` objects, getters, and setters to *cls* for each
    entry in *param_specs*.

    This is called by ``BaseTransformer.__init_subclass__`` for every
    concrete transformer that defines ``_params``.
    """
    for name, spec in param_specs.items():
        converter = spec.spark_typeconverter or TypeConverters.identity

        param_obj = Param(Params._dummy(), name, spec.doc, typeConverter=converter)
        setattr(cls, name, param_obj)

        getter_name = f"get{name[0].upper()}{name[1:]}"
        setter_name = f"set{name[0].upper()}{name[1:]}"

        def _make_getter(param_name: str) -> Callable:
            def getter(self: Any) -> Any:
                return self.getOrDefault(param_name)

            getter.__name__ = getter_name
            return getter

        def _make_setter(param_name: str, validator: Optional[Callable]) -> Callable:
            if validator is not None:
                _needs_self = len(inspect.signature(validator).parameters) > 1
            else:
                _needs_self = False

            def setter(self: Any, value: Any) -> Any:
                if validator is not None:
                    if _needs_self:
                        value = validator(self, value)
                    else:
                        value = validator(value)
                return self._set(**{param_name: value})

            setter.__name__ = setter_name
            return setter

        if not hasattr(cls, getter_name):
            setattr(cls, getter_name, _make_getter(name))
        if not hasattr(cls, setter_name):
            setattr(cls, setter_name, _make_setter(name, spec.validator))


def build_keras_layer_from_specs(
    self: Any,
    keras_layer_class: Type,
    param_specs: Dict[str, ParamSpec],
) -> Any:
    """Construct a Keras layer instance from the transformer's current param
    values.

    Always passes ``name``, ``input_dtype``, ``output_dtype``. Then for each
    entry in *param_specs*, reads the current value via the generated getter
    method (e.g. ``getAlpha()``) so that any custom validation logic in the
    getter is respected.
    """
    layer_params = getattr(keras_layer_class, "_params", {})
    kwargs: Dict[str, Any] = {
        "name": self.getLayerName(),
        "input_dtype": self.getInputKerasDtype(),
        "output_dtype": self.getOutputKerasDtype(),
    }
    for name, spec in param_specs.items():
        keras_name = _camel_to_snake(name)
        if layer_params and keras_name not in layer_params:
            continue
        getter_name = f"get{name[0].upper()}{name[1:]}"
        getter = getattr(self, getter_name)
        kwargs[keras_name] = getter()
    return keras_layer_class(**kwargs)
