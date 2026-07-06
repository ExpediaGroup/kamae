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

    The `checkpointInterval` param optionally bounds the depth of the Spark
    logical plan built up while fitting a multi-estimator pipeline. When set to a
    positive integer it triggers a reliable `DataFrame.checkpoint(eager=True)`
    every `checkpointInterval` stages (evaluated at estimator-fit action
    boundaries), physically truncating the accumulated lineage. This is a
    depth-bounding / reliability feature: it guards against deep-plan failures such
    as "plan too large", 64KB codegen, and CodeCache-full errors, and avoids
    re-executing the full upstream lineage on every estimator fit.

    Reliable checkpointing writes the intermediate DataFrame to the checkpoint
    directory configured via `spark.sparkContext.setCheckpointDir(<path>)`, which
    must point at fault-tolerant storage (DFS/cloud storage). Unlike local
    checkpointing it survives executor loss (e.g. autoscaling, spot reclaim, OOM),
    at the cost of writing to remote storage rather than executor-local disk. A
    checkpoint directory MUST be set before fitting with a positive interval. Its
    throughput impact is data-dependent and NOT guaranteed positive (the full, wide
    intermediate DataFrame is persisted with no column pruning), so benchmark before
    relying on it for speed. The default of 0 disables checkpointing entirely,
    leaving fit behaviour byte-for-byte unchanged.

    The `cacheIntermediateData` param optionally persists the working DataFrame
    (MEMORY_AND_DISK) at each estimator-fit boundary so that the estimator's fit
    action - and any subsequent transforms - reuse a materialised result instead of
    re-executing (and re-reading from source) the full upstream lineage on every
    estimator. Only one intermediate frame is held at a time: each new persist
    unpersists the one it supersedes, and the final frame is released before
    returning. Unlike `checkpointInterval` it does not truncate the logical plan or
    require a checkpoint directory; it is purely a re-scan-avoidance optimisation.
    It preserves data exactly, so fitted results are identical to the default. The
    default of False leaves fit behaviour unchanged.
    """

    checkpointInterval = Param(
        Params._dummy(),
        "checkpointInterval",
        "Number of stages between reliable checkpoint(eager=True) calls during "
        "fit, used to bound logical-plan depth. Requires a checkpoint directory set "
        "via spark.sparkContext.setCheckpointDir. 0 (the default) disables "
        "checkpointing and leaves fit behaviour exactly unchanged.",
        typeConverter=TypeConverters.toInt,
    )

    cacheIntermediateData = Param(
        Params._dummy(),
        "cacheIntermediateData",
        "If True, persist the working DataFrame (MEMORY_AND_DISK) at each "
        "estimator-fit boundary so estimator fits reuse a materialised result "
        "rather than re-scanning the upstream lineage from source. False (the "
        "default) leaves fit behaviour exactly unchanged.",
        typeConverter=TypeConverters.toBoolean,
    )

    @keyword_only
    def __init__(
        self,
        *,
        stages: Optional[List["KamaePipelineStage"]] = None,
        checkpointInterval: int = 0,
        cacheIntermediateData: bool = False,
    ) -> None:
        """
        Initialises the KamaeSparkPipeline object.

        :param stages: List of LayerTransformers to chain together.
        :param checkpointInterval: Number of stages between reliable
        checkpoint(eager=True) calls during fit. 0 (default) disables it.
        :param cacheIntermediateData: If True, persist the working DataFrame at
        each estimator-fit boundary to avoid re-scanning the upstream lineage.
        False (default) disables it.
        :returns: None - class instantiated.
        """
        kwargs = self._input_kwargs
        super().__init__()
        self._setDefault(checkpointInterval=0, cacheIntermediateData=False)
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

    @keyword_only
    def setParams(
        self,
        *,
        stages: Optional["KamaePipelineStage"] = None,
        checkpointInterval: int = 0,
        cacheIntermediateData: bool = False,
    ) -> "KamaeSparkPipeline":
        """
        Sets the keyword arguments of the pipeline.

        :param stages: List of pipeline stages.
        :param checkpointInterval: Number of stages between reliable
        checkpoint(eager=True) calls during fit. 0 (default) disables it.
        :param cacheIntermediateData: If True, persist the working DataFrame at
        each estimator-fit boundary. False (default) disables it.
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
        Collects every column read as an input by any stage in the pipeline.

        A raw input-DataFrame column absent from this set is consumed by no stage
        and can be dropped before fitting, so it is not carried through every
        transform (and every materialisation) below.

        :param stages: List of pipeline stages.
        :returns: Set of column names read by at least one stage.
        """
        required_input_columns: Set[str] = set()
        for stage in stages:
            inputs, _ = stage.get_layer_inputs_outputs()
            required_input_columns.update(inputs)
        return required_input_columns

    def prune_unused_input_columns(
        self,
        dataset: DataFrame,
        stages: List["KamaePipelineStage"],
    ) -> DataFrame:
        """
        Projects the input DataFrame down to only the columns the pipeline reads.

        Columns produced by stages are created downstream via `withColumn`, so only
        the pipeline's source columns need to be present up front. Pruning here
        keeps the frame narrow before any expansion, reducing the cost of every
        subsequent transform and materialisation. If no unused columns are found
        (or the pipeline reads none of the DataFrame's columns) the DataFrame is
        returned unchanged.

        :param dataset: Input DataFrame to prune.
        :param stages: Expanded pipeline stages.
        :returns: DataFrame projected to the columns the pipeline consumes.
        """
        required_input_columns = self.collect_required_input_columns(stages)
        columns_to_keep = [c for c in dataset.columns if c in required_input_columns]
        if columns_to_keep and len(columns_to_keep) < len(dataset.columns):
            return dataset.select(*columns_to_keep)
        return dataset

    def _fit(self, dataset: DataFrame) -> "KamaeSparkPipelineModel":
        """
        Fits the pipeline to the dataset. Returns a KamaeSparkPipelineModel object.

        Calls the super fit method of the pyspark.ml.Pipeline class and
        then constructs a KamaeSparkPipelineModel uses the stages from the fit pipeline.

        Before fitting, the input DataFrame is projected down to only the columns
        the pipeline reads (see `prune_unused_input_columns`), so columns no stage
        consumes are not carried through every transform and materialisation.

        If `checkpointInterval` is a positive integer, the working DataFrame is
        reliably checkpointed via `checkpoint(eager=True)` roughly every
        `checkpointInterval` stages (at estimator-fit action boundaries) to bound
        logical-plan depth. checkpoint(eager=True) preserves the data exactly and
        only truncates lineage, so fitted results are numerically identical to the
        default (interval=0) behaviour. A checkpoint directory must be configured via
        `spark.sparkContext.setCheckpointDir` before fitting with a positive interval.
        The default of 0 (or None) disables checkpointing entirely.

        If `cacheIntermediateData` is True, the working DataFrame is persisted
        (MEMORY_AND_DISK) at each estimator-fit boundary so the fit action and any
        subsequent transform reuse a materialised result rather than re-scanning the
        upstream lineage. Persistence preserves data exactly, so fitted results are
        identical to the default (False) behaviour. The default of False disables it.

        :param dataset: PySpark DataFrame to fit the pipeline to.
        :returns: KamaeSparkPipelineModel object.
        :raises ValueError: If checkpointing is enabled but no checkpoint directory
        has been set on the SparkContext.
        """
        expanded_pipeline_stages = self.expand_pipeline_stages()

        for stage in expanded_pipeline_stages:
            if not (
                isinstance(stage, BaseEstimator) or isinstance(stage, BaseTransformer)
            ):
                raise TypeError(
                    "Cannot recognize a pipeline stage of type %s." % type(stage)
                )

        # Drop input columns no stage consumes before any expansion, so dead
        # columns are not carried through every transform and materialisation below.
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
        # Optional, opt-in plan-depth bounding. 0 (or None) keeps behaviour unchanged.
        checkpoint_interval = self.getCheckpointInterval()
        checkpoint_enabled = checkpoint_interval is not None and checkpoint_interval > 0
        # Reliable checkpoint() requires a checkpoint directory; fail fast with a clear
        # message rather than letting Spark raise mid-fit after work has been done.
        if (
            checkpoint_enabled
            and dataset.sparkSession.sparkContext.getCheckpointDir() is None
        ):
            raise ValueError(
                "checkpointInterval > 0 requires a checkpoint directory. Set one via "
                "spark.sparkContext.setCheckpointDir(<path>) before fitting."
            )
        cache_enabled = self.getCacheIntermediateData()
        last_checkpoint_index = 0
        # Holds the single intermediate frame currently persisted (if any) so it can
        # be unpersisted once superseded or once fitting completes.
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
                # Truncate the accumulated lineage just before the fit action so the
                # plan is physically bounded. eager=True forces materialisation now.
                if (
                    checkpoint_enabled
                    and index - last_checkpoint_index >= checkpoint_interval
                ):
                    dataset = dataset.checkpoint(eager=True)
                    last_checkpoint_index = index
                # Persist the working frame so the fit action and any subsequent
                # transform read a materialised result rather than re-scanning the
                # upstream lineage from source. Only one frame is held at a time.
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
