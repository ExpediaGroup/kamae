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
from typing import Union

from kamae.spark.estimators import BaseEstimator
from kamae.spark.transformers import BaseTransformer

from .pipeline_model import (  # noqa: F401 # isort: skip
    KamaeSparkPipelineModel,
    KamaeSparkPipelineModelReader,
    KamaeSparkPipelineModelWriter,
)
from .pipeline import (  # noqa: F401 # isort: skip
    KamaeSparkPipeline,
    KamaeSparkPipelineReader,
    KamaeSparkPipelineWriter,
)


KamaePipelineStage = Union[BaseEstimator, BaseTransformer]
