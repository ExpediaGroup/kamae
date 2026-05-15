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

from typing import Dict

from kamae.params import _REQUIRED, ParamSpec  # noqa: F401
from kamae.params.param_spec import _UNSET


def install_params(
    cls: type,
    param_specs: Dict[str, ParamSpec],
) -> None:
    """Generate ``__init__`` and ``get_config`` methods for a Keras layer
    based on its ``_params`` declaration.

    - ``__init__`` accepts ``name``, ``input_dtype``, ``output_dtype``,
      ``**kwargs`` (from BaseLayer), plus every key in *param_specs*. Each
      param is stored as ``self.<name>``.
    - ``get_config`` calls ``super().get_config()`` and adds every param.
    """

    if "__init__" not in cls.__dict__:
        specs = param_specs
        post_init = cls.__dict__.get("_post_init")
        _super = super

        def __init__(self, name=None, input_dtype=None, output_dtype=None, **kwargs):
            for param_name, spec in specs.items():
                value = kwargs.pop(param_name, spec.default)
                if value is _REQUIRED:
                    raise TypeError(
                        f"{type(self).__name__}.__init__() missing required "
                        f"keyword argument: '{param_name}'"
                    )
                if value is _UNSET:
                    value = None
                if spec.validator is not None:
                    value = spec.validator(value)
                setattr(self, param_name, value)
            _super(cls, self).__init__(
                name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
            )
            if post_init is not None:
                post_init(self)

        cls.__init__ = __init__

    if "get_config" not in cls.__dict__:
        specs = param_specs
        _super = super

        def get_config(self):
            config = _super(cls, self).get_config()
            for param_name in specs:
                config[param_name] = getattr(self, param_name)
            return config

        cls.get_config = get_config
