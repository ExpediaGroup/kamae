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

from inspect import Parameter, Signature
from typing import Any, Dict, List

from pyspark import keyword_only

from kamae.params import ParamSpec
from kamae.params.param_spec import _UNSET


def collect_io_params(cls: type) -> List[str]:
    """Return the base I/O parameter names contributed by the IO mixins
    (e.g. SingleInputSingleOutputParams).  We inspect the MRO for well-known
    PySpark shared params and our own HasInputDtype / HasOutputDtype."""
    names: List[str] = []
    has = set()
    for klass in cls.__mro__:
        for attr_name in ("inputCol", "inputCols", "outputCol", "outputCols"):
            if attr_name not in has and hasattr(klass, attr_name):
                has.add(attr_name)
                names.append(attr_name)
    for extra in ("inputDtype", "outputDtype", "layerName"):
        if extra not in has and hasattr(cls, extra):
            names.append(extra)
    return names


def collect_mixin_params(cls: type) -> Dict[str, Any]:
    """Collect param names from mixin classes that need default values set.

    Returns a dict mapping param names to None (the default for optional params).
    """
    mixin_defaults = {}
    for klass in cls.__mro__[1:]:  # Skip cls itself
        if klass.__name__.endswith("Params") and hasattr(klass, "__dict__"):
            for attr_name, attr_value in klass.__dict__.items():
                # Check if it's a PySpark Param
                if (
                    hasattr(attr_value, "__class__")
                    and attr_value.__class__.__name__ == "Param"
                ):
                    mixin_defaults[attr_name] = None
    return mixin_defaults


def build_init(
    io_param_names: List[str],
    custom_param_specs: Dict[str, ParamSpec],
    cls: type,
) -> Any:
    """Generate a ``@keyword_only`` ``__init__`` method for a transformer.

    The generated ``__init__`` accepts all I/O params (defaulting to ``None``)
    plus any custom params from ``_params``, calls ``super().__init__()``,
    sets defaults, and delegates to ``setParams``.
    """

    defaults = {}
    for spec_name, spec in custom_param_specs.items():
        defaults[spec_name] = None if spec.default is _UNSET else spec.default

    # Also include defaults for mixin params
    mixin_defaults = collect_mixin_params(cls)
    defaults.update(mixin_defaults)

    @keyword_only
    def __init__(self, **kwargs):
        super(type(self), self).__init__()
        if defaults:
            self._setDefault(**defaults)
        init_kwargs = self._input_kwargs
        self.setParams(**init_kwargs)

    sig_params = [Parameter("self", Parameter.POSITIONAL_OR_KEYWORD)]
    for name in io_param_names:
        sig_params.append(Parameter(name, Parameter.KEYWORD_ONLY, default=None))
    for name, spec in custom_param_specs.items():
        sig_default = None if spec.default is _UNSET else spec.default
        sig_params.append(Parameter(name, Parameter.KEYWORD_ONLY, default=sig_default))
    # Also add mixin params to signature
    for name in mixin_defaults.keys():
        if name not in io_param_names and name not in custom_param_specs:
            sig_default = defaults.get(name, None)
            sig_params.append(
                Parameter(name, Parameter.KEYWORD_ONLY, default=sig_default)
            )
    __init__.__signature__ = Signature(sig_params)
    __init__.__wrapped__ = True

    return __init__
