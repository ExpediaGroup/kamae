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

from typing import TYPE_CHECKING, List, Optional, Set, Type

import networkx as nx
from pyspark import keyword_only
from pyspark.ml import Pipeline
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.pipeline import PipelineReader, PipelineSharedReadWrite, PipelineWriter
from pyspark.ml.util import DefaultParamsReader, MLWriter
from pyspark.sql import DataFrame
from pyspark.storagelevel import StorageLevel

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

    Three opt-in fit optimisations are available, all defaulting off (fit behaviour
    unchanged): `checkpointInterval` reliably checkpoints every N stages to bound
    logical-plan depth (requires a checkpoint dir); `cacheIntermediateData` persists
    the working DataFrame at each estimator-fit boundary to avoid re-scanning the
    upstream lineage; `pruneInputColumns` drops input columns no stage consumes.
    """

    checkpointInterval = Param(
        Params._dummy(),
        "checkpointInterval",
        "Stages between reliable checkpoint(eager=True) calls during fit, to bound "
        "logical-plan depth. Requires a checkpoint dir. 0 (default) disables it.",
        typeConverter=TypeConverters.toInt,
    )

    cacheIntermediateData = Param(
        Params._dummy(),
        "cacheIntermediateData",
        "If True, persist the working DataFrame (MEMORY_AND_DISK) at each "
        "estimator-fit boundary to avoid re-scanning the upstream lineage. False "
        "(default) disables it.",
        typeConverter=TypeConverters.toBoolean,
    )

    pruneInputColumns = Param(
        Params._dummy(),
        "pruneInputColumns",
        "If True, drop input columns no stage consumes before fitting. False "
        "(default) disables it.",
        typeConverter=TypeConverters.toBoolean,
    )

    @keyword_only
    def __init__(
        self,
        *,
        stages: Optional[List["KamaePipelineStage"]] = None,
        checkpointInterval: int = 0,
        cacheIntermediateData: bool = False,
        pruneInputColumns: bool = False,
    ) -> None:
        """
        Initialises the KamaeSparkPipeline object.

        :param stages: List of LayerTransformers to chain together.
        :param checkpointInterval: Number of stages between reliable
        checkpoint(eager=True) calls during fit. 0 (default) disables it.
        :param cacheIntermediateData: If True, persist the working DataFrame at
        each estimator-fit boundary to avoid re-scanning the upstream lineage.
        False (default) disables it.
        :param pruneInputColumns: If True, drop input columns no stage consumes
        before fitting. False (default) disables it.
        :returns: None - class instantiated.
        """
        kwargs = self._input_kwargs
        super().__init__()
        self._setDefault(
            checkpointInterval=0,
            cacheIntermediateData=False,
            pruneInputColumns=False,
        )
        self.setParams(**kwargs)

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

    def setCheckpointInterval(self, value: int) -> "KamaeSparkPipeline":
        """
        Sets the `checkpointInterval` parameter.

        :param value: Number of stages between reliable checkpoint calls during
        fit. 0 (or None) disables checkpointing.
        :returns: KamaeSparkPipeline object with checkpointInterval set.
        """
        return self._set(checkpointInterval=value)

    def getCheckpointInterval(self) -> int:
        """
        Gets the value of the `checkpointInterval` parameter.

        :returns: The checkpointInterval value.
        """
        return self.getOrDefault(self.checkpointInterval)

    def setCacheIntermediateData(self, value: bool) -> "KamaeSparkPipeline":
        """
        Sets the `cacheIntermediateData` parameter.

        :param value: Whether to persist the working DataFrame at each
        estimator-fit boundary during fit.
        :returns: KamaeSparkPipeline object with cacheIntermediateData set.
        """
        return self._set(cacheIntermediateData=value)

    def getCacheIntermediateData(self) -> bool:
        """
        Gets the value of the `cacheIntermediateData` parameter.

        :returns: The cacheIntermediateData value.
        """
        return self.getOrDefault(self.cacheIntermediateData)

    def setPruneInputColumns(self, value: bool) -> "KamaeSparkPipeline":
        """
        Sets the `pruneInputColumns` parameter.

        :param value: Whether to drop input columns no stage consumes before fitting.
        :returns: KamaeSparkPipeline object with pruneInputColumns set.
        """
        return self._set(pruneInputColumns=value)

    def getPruneInputColumns(self) -> bool:
        """
        Gets the value of the `pruneInputColumns` parameter.

        :returns: The pruneInputColumns value.
        """
        return self.getOrDefault(self.pruneInputColumns)

    @keyword_only
    def setParams(
        self,
        *,
        stages: Optional["KamaePipelineStage"] = None,
        checkpointInterval: int = 0,
        cacheIntermediateData: bool = False,
        pruneInputColumns: bool = False,
    ) -> "KamaeSparkPipeline":
        """
        Sets the keyword arguments of the pipeline.

        :param stages: List of pipeline stages.
        :param checkpointInterval: Number of stages between reliable
        checkpoint(eager=True) calls during fit. 0 (default) disables it.
        :param cacheIntermediateData: If True, persist the working DataFrame at
        each estimator-fit boundary. False (default) disables it.
        :param pruneInputColumns: If True, drop input columns no stage consumes
        before fitting. False (default) disables it.
        :returns: KamaeSparkPipeline object with params set.
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

    @staticmethod
    def collect_required_input_columns(
        stages: List["KamaePipelineStage"],
    ) -> Set[str]:
        """
        Collects every column potentially read by any stage in the pipeline.

        Generous by design: unions canonical inputs with the value(s) of every param
        whose name ends in `Col`/`Cols`, so aux columns read during fit are not
        missed. Over-inclusion is harmless (names not matching `dataset.columns` are
        ignored); omission would wrongly drop data the pipeline needs.

        :param stages: List of pipeline stages.
        :returns: Set of column names potentially read by at least one stage.
        """
        required_input_columns: Set[str] = set()
        for stage in stages:
            inputs, _ = stage.get_layer_inputs_outputs()
            required_input_columns.update(inputs)
            for param in stage.params:
                if not (param.name.endswith("Col") or param.name.endswith("Cols")):
                    continue
                if not stage.isDefined(param):
                    continue
                value = stage.getOrDefault(param)
                if isinstance(value, str):
                    required_input_columns.add(value)
                elif isinstance(value, (list, tuple)):
                    required_input_columns.update(
                        item for item in value if isinstance(item, str)
                    )
        return required_input_columns

    def prune_unused_input_columns(
        self,
        dataset: DataFrame,
        stages: List["KamaePipelineStage"],
    ) -> DataFrame:
        """
        Projects the input DataFrame down to only the columns the pipeline reads.

        Returned unchanged if there are no unused columns to drop.

        :param dataset: Input DataFrame to prune.
        :param stages: Expanded pipeline stages.
        :returns: DataFrame projected to the columns the pipeline consumes.
        """
        required_input_columns = self.collect_required_input_columns(stages)
        columns_to_keep = [c for c in dataset.columns if c in required_input_columns]
        if columns_to_keep and len(columns_to_keep) < len(dataset.columns):
            return dataset.select(*columns_to_keep)
        return dataset

    @staticmethod
    def _validate_stage_types(stages: List["KamaePipelineStage"]) -> None:
        """
        Ensures every expanded stage is a recognised estimator or transformer.

        :param stages: Expanded pipeline stages.
        :raises TypeError: If any stage is not a BaseEstimator or BaseTransformer.
        """
        for stage in stages:
            if not isinstance(stage, (BaseEstimator, BaseTransformer)):
                raise TypeError(
                    "Cannot recognize a pipeline stage of type %s." % type(stage)
                )

    @staticmethod
    def _resolve_checkpoint_enabled(
        dataset: DataFrame, checkpoint_interval: Optional[int]
    ) -> bool:
        """
        Determines whether checkpointing is enabled and validates its prerequisites.

        Fails fast if enabled without a checkpoint dir, rather than raising mid-fit.

        :param dataset: DataFrame whose SparkContext is checked for a checkpoint dir.
        :param checkpoint_interval: Configured checkpoint interval (0/None disables).
        :returns: True if checkpointing is enabled, False otherwise.
        :raises ValueError: If enabled but no checkpoint directory has been set.
        """
        checkpoint_enabled = checkpoint_interval is not None and checkpoint_interval > 0
        if (
            checkpoint_enabled
            and dataset.sparkSession.sparkContext.getCheckpointDir() is None
        ):
            raise ValueError(
                "checkpointInterval > 0 requires a checkpoint directory. Set one via "
                "spark.sparkContext.setCheckpointDir(<path>) before fitting."
            )
        return checkpoint_enabled

    def _fit(self, dataset: DataFrame) -> "KamaeSparkPipelineModel":
        """
        Fits the pipeline to the dataset. Returns a KamaeSparkPipelineModel object.

        Calls the super fit method of the pyspark.ml.Pipeline class and
        then constructs a KamaeSparkPipelineModel uses the stages from the fit pipeline.

        Optionally applies the opt-in fit optimisations (`pruneInputColumns`,
        `checkpointInterval`, `cacheIntermediateData`); see the class docstring. All
        preserve data exactly, so fitted results match the defaults-off behaviour.

        :param dataset: PySpark DataFrame to fit the pipeline to.
        :returns: KamaeSparkPipelineModel object.
        :raises ValueError: If checkpointing is enabled but no checkpoint directory
        has been set on the SparkContext.
        """
        expanded_pipeline_stages = self.expand_pipeline_stages()
        self._validate_stage_types(expanded_pipeline_stages)

        # Opt-in: drop input columns no stage consumes. Default False = no change.
        if self.getPruneInputColumns():
            dataset = self.prune_unused_input_columns(dataset, expanded_pipeline_stages)

        # Native Spark checks for the last estimator and executes all transformers
        # before it, regardless whether there is a dependency between them. See here:
        # https://github.com/apache/spark/blob/master/python/pyspark/ml/pipeline.py#L120
        # We can be clever, since we have built a proper DAG, by only executing
        # transformers that are required by the estimator.

        # Collect the parents of the estimators in the pipeline
        estimator_parent_stages = self.collect_estimator_parents(
            expanded_pipeline_stages
        )
        # Opt-in plan-depth bounding. 0 (or None) = no change.
        checkpoint_interval = self.getCheckpointInterval()
        checkpoint_enabled = self._resolve_checkpoint_enabled(
            dataset, checkpoint_interval
        )
        cache_enabled = self.getCacheIntermediateData()
        last_checkpoint_index = 0
        # The single persisted frame (if any), unpersisted once superseded or done.
        cached_dataset: Optional[DataFrame] = None
        # Fit each stage, appending the transformer to the list of transformers
        # If the stage is a parent of an estimator, transform the dataset.
        transformers: List[BaseTransformer] = []
        for index, stage in enumerate(expanded_pipeline_stages):
            if isinstance(stage, BaseTransformer):
                transformers.append(stage)
                if stage in estimator_parent_stages:
                    dataset = stage.transform(dataset)
            else:
                # Truncate accumulated lineage before the fit action to bound plan
                # depth. eager=True materialises now.
                if (
                    checkpoint_enabled
                    and index - last_checkpoint_index >= checkpoint_interval
                ):
                    dataset = dataset.checkpoint(eager=True)
                    last_checkpoint_index = index
                # Persist so the fit action and downstream transforms reuse a
                # materialised frame instead of re-scanning. One frame held at a time.
                if cache_enabled:
                    new_cached = dataset.persist(StorageLevel.MEMORY_AND_DISK)
                    if cached_dataset is not None:
                        cached_dataset.unpersist()
                    cached_dataset = new_cached
                    dataset = new_cached
                model = stage.fit(dataset)
                transformers.append(model)
                if stage in estimator_parent_stages:
                    dataset = model.transform(dataset)
        if cached_dataset is not None:
            cached_dataset.unpersist()
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

    def __init__(self, cls: Type[KamaeSparkPipeline]) -> None:
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

    def __init__(self, instance: KamaeSparkPipeline) -> None:
        super().__init__(instance=instance)
