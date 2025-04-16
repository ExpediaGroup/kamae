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

from typing import TYPE_CHECKING, List, Optional, Type

import networkx as nx
from pyspark import keyword_only
from pyspark.ml import Pipeline
from pyspark.ml.param import Params
from pyspark.ml.pipeline import PipelineReader, PipelineSharedReadWrite, PipelineWriter
from pyspark.ml.util import DefaultParamsReader, MLWriter
from pyspark.sql import DataFrame

from kamae.graph import PipelineGraph
from kamae.spark.estimators import BaseEstimator
from kamae.spark.pipeline import KamaeSparkPipelineModel
from kamae.spark.transformers import BaseTransformer

if TYPE_CHECKING:
    from pyspark.ml._typing import ParamMap

    from kamae.spark.pipeline import KamaePipelineStage


class KamaeSparkPipeline(Pipeline):
    """
    KamaeSparkPipeline is a subclass of pyspark.ml.Pipeline that is used to chain
    together BaseTransformers.
    It maintains the same functionality as pyspark.ml.Pipeline e.g. serialisation.
    """

    @keyword_only
    def __init__(self, *, stages: Optional[List["KamaePipelineStage"]] = None) -> None:
        """
        Initialises the KamaeSparkPipeline object.

        :param stages: List of LayerTransformers to chain together.
        :returns: None - class instantiated.
        """
        super().__init__(stages=stages)

    def setStages(self, value: List["KamaePipelineStage"]) -> "KamaeSparkPipeline":
        """
        Sets the stages of the pipeline.

        :param value: List of pipeline stages.
        :returns: KamaeSparkPipeline object with stages set.
        """
        return self._set(stages=value)

    def getStages(self) -> List["KamaePipelineStage"]:
        """
        Gets the stages of the pipeline.

        :returns: List of pipeline stages.
        """
        return self.getOrDefault("stages")

    @keyword_only
    def setParams(
        self, *, stages: Optional["KamaePipelineStage"] = None
    ) -> "KamaeSparkPipeline":
        """
        Sets the keyword arguments of the pipeline.

        :param stages: List of pipeline stages.
        :returns: KamaeSparkPipeline object with stages set.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def expand_pipeline_stages(self) -> List["KamaePipelineStage"]:
        """
        Expands the pipeline stages to include all nested pipeline stages.
        If the pipeline stage is itself a pipeline model, it will be expanded
        recursively.

        :returns: List of all pipeline stages flattened to transformer level.
        """
        expanded_stages = []
        for stage in self.getStages():
            if isinstance(stage, (KamaeSparkPipelineModel, KamaeSparkPipeline)):
                # Recursively expand the pipeline stages.
                expanded_stages.extend(stage.expand_pipeline_stages())
            else:
                expanded_stages.append(stage)
        return expanded_stages

    @staticmethod
    def collect_estimator_parents(
        stages: List["KamaePipelineStage"],
    ) -> List["KamaePipelineStage"]:
        """
        Collects the parent stages of the estimators in the pipeline.

        Used to determine which transformers to execute before the estimators in the
        pipeline.

        :param stages: List of pipeline stages.
        :returns: List of names of the ancestors of the estimators in the pipeline.
        """
        stage_dict = {
            stage.getOrDefault("layerName"): stage.construct_layer_info()
            for stage in stages
        }
        pipeline_graph = PipelineGraph(stage_dict=stage_dict)
        estimator_stages = [
            stage for stage in stages if isinstance(stage, BaseEstimator)
        ]
        estimator_parents = []
        for estimator in estimator_stages:
            layer_name = estimator.getLayerName()
            specific_estimator_parents = nx.ancestors(pipeline_graph.graph, layer_name)
            estimator_parents.extend(specific_estimator_parents)

        distinct_estimator_parents = list(set(estimator_parents))
        estimator_parent_stages = [
            stage
            for stage in stages
            if stage.getLayerName() in distinct_estimator_parents
        ]
        return estimator_parent_stages

    def _fit(self, dataset: DataFrame) -> "KamaeSparkPipelineModel":
        """
        Fits the pipeline to the dataset. Returns a KamaeSparkPipelineModel object.

        Calls the super fit method of the pyspark.ml.Pipeline class and
        then constructs a KamaeSparkPipelineModel uses the stages from the fit pipeline.

        :param dataset: PySpark DataFrame to fit the pipeline to.
        :returns: KamaeSparkPipelineModel object.
        """
        expanded_pipeline_stages = self.expand_pipeline_stages()

        for stage in expanded_pipeline_stages:
            if not (
                isinstance(stage, BaseEstimator) or isinstance(stage, BaseTransformer)
            ):
                raise TypeError(
                    "Cannot recognize a pipeline stage of type %s." % type(stage)
                )

        # Native Spark checks for the last estimator and executes all transformers
        # before it, regardless whether there is a dependency between them. See here:
        # https://github.com/apache/spark/blob/master/python/pyspark/ml/pipeline.py#L120
        # We can be clever, since we have built a proper DAG, by only executing
        # transformers that are required by the estimator.

        # Collect the parents of the estimators in the pipeline
        estimator_parent_stages = self.collect_estimator_parents(
            expanded_pipeline_stages
        )
        # Fit each stage, appending the transformer to the list of transformers
        # If the stage is a parent of an estimator, transform the dataset.
        transformers: List[BaseTransformer] = []
        for stage in expanded_pipeline_stages:
            if isinstance(stage, BaseTransformer):
                transformers.append(stage)
                if stage in estimator_parent_stages:
                    dataset = stage.transform(dataset)
            else:
                model = stage.fit(dataset)
                transformers.append(model)
                if stage in estimator_parent_stages:
                    dataset = model.transform(dataset)
        return KamaeSparkPipelineModel(transformers)

    def copy(self, extra: Optional["ParamMap"] = None) -> "KamaeSparkPipeline":
        """
        Creates a copy of the KamaeSparkPipeline object.

        :param extra: Additional optional params to copy to new pipeline.
        :returns: KamaeSparkPipeline object.
        """
        if extra is None:
            extra = dict()
        that = Params.copy(self, extra)
        stages = [stage.copy(extra) for stage in that.getStages()]
        return that.setStages(stages)

    def write(self) -> MLWriter:
        """
        Uses the KamaeSparkPipelineWriter class to write the pipeline to a
        persistent storage path.

        :returns: KamaeSparkPipelineWriter object.
        """
        return KamaeSparkPipelineWriter(self)

    @classmethod
    def read(cls) -> "KamaeSparkPipelineReader":
        """
        Uses the KamaeSparkPipelineReader class to read a pipeline from a
        persistent storage path.

        :returns: KamaeSparkPipelineReader object.
        """
        return KamaeSparkPipelineReader(cls)


class KamaeSparkPipelineReader(PipelineReader):
    """
    Util class for reading a pipeline from a persistent storage path.
    """

    def __init__(self, cls: Type[KamaeSparkPipeline]):
        super().__init__(cls=cls)

    def load(self, path: str) -> KamaeSparkPipeline:
        """
        Loads a pipeline from a given path.

        :param path: Path to stored pipeline.
        :returns: KamaeSparkPipeline object.
        """
        metadata = DefaultParamsReader.loadMetadata(path, self.sc)
        uid, stages = PipelineSharedReadWrite.load(metadata, self.sc, path)
        return KamaeSparkPipeline(stages=stages)._resetUid(uid)


class KamaeSparkPipelineWriter(PipelineWriter):
    """
    Util class for writing a pipeline to a persistent storage path.
    """

    def __init__(self, instance: KamaeSparkPipeline):
        super().__init__(instance=instance)
