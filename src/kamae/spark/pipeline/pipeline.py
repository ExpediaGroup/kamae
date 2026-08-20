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

import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple, Type

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

    Five opt-in fit optimisations are available, all defaulting off (fit behaviour
    unchanged): `checkpointInterval` reliably checkpoints every N stages to bound
    logical-plan depth (requires a checkpoint dir); `cacheIntermediateData` persists
    the working DataFrame at each estimator-fit boundary to avoid re-scanning the
    upstream lineage; `pruneInputColumns` drops input columns no stage consumes;
    `cacheEstimatorInput` projects to the columns still read downstream at the first
    estimator boundary and persists that narrow frame once, so independent sibling
    estimators reuse it instead of re-scanning the wide input; `fitSampleFraction`
    draws a single persisted sample of the input up-front and fits every estimator
    from it (see its param docstring for the correctness caveat).
    """

    checkpointInterval = Param(
        Params._dummy(),
        "checkpointInterval",
        "Stages between reliable checkpoint(eager=True) calls during fit, to bound "
        "logical-plan depth. Requires a checkpoint dir. None (default) disables it.",
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

    cacheEstimatorInput = Param(
        Params._dummy(),
        "cacheEstimatorInput",
        "If True, at the first estimator-fit boundary project the working DataFrame "
        "to the columns still read downstream and persist (MEMORY_AND_DISK) that "
        "narrow frame once, reused by all subsequent estimators. Competes with "
        "cacheIntermediateData; enable at most one. False (default) disables it.",
        typeConverter=TypeConverters.toBoolean,
    )

    fitSampleFraction = Param(
        Params._dummy(),
        "fitSampleFraction",
        "If set to a float in (0, 1], draw a single sample of the input up-front, "
        "persist (MEMORY_AND_DISK) and materialise it once, and fit every estimator "
        "from that shared sample (each estimator's own sampleFraction is ignored for "
        "the fit). Avoids re-scanning or persisting the wide source once per "
        "estimator. ONLY correct when every estimator computes sample-robust "
        "statistics (mean/std/quantiles, e.g. ConditionalStandardScale); vocabulary "
        "builders (StringIndexer/OneHot), min/max scalers and distinct counts need "
        "exact/global statistics and will be inaccurate on a sample. Disables "
        "cacheIntermediateData and cacheEstimatorInput. None (default) disables it.",
        typeConverter=TypeConverters.toFloat,
    )

    fitSampleSeed = Param(
        Params._dummy(),
        "fitSampleSeed",
        "Optional integer seed passed to the fitSampleFraction sample for "
        "reproducibility. None (default) leaves the sample unseeded.",
        typeConverter=TypeConverters.toInt,
    )

    @keyword_only
    def __init__(
        self,
        *,
        stages: Optional[List["KamaePipelineStage"]] = None,
        checkpointInterval: Optional[int] = None,
        cacheIntermediateData: bool = False,
        pruneInputColumns: bool = False,
        cacheEstimatorInput: bool = False,
        fitSampleFraction: Optional[float] = None,
        fitSampleSeed: Optional[int] = None,
    ) -> None:
        """
        Initialises the KamaeSparkPipeline object.

        :param stages: List of LayerTransformers to chain together.
        :param checkpointInterval: Number of stages between reliable
        checkpoint(eager=True) calls during fit. None (default) disables it.
        :param cacheIntermediateData: If True, persist the working DataFrame at
        each estimator-fit boundary to avoid re-scanning the upstream lineage.
        False (default) disables it.
        :param pruneInputColumns: If True, drop input columns no stage consumes
        before fitting. False (default) disables it.
        :param cacheEstimatorInput: If True, project to the columns still read
        downstream at the first estimator boundary and persist that narrow frame
        once for reuse by subsequent estimators. False (default) disables it.
        :param fitSampleFraction: If set to a float in (0, 1], fit every estimator
        from a single up-front persisted sample of the input. Only correct when all
        estimators compute sample-robust statistics. None (default) disables it.
        :param fitSampleSeed: Optional integer seed for the fitSampleFraction
        sample. None (default) leaves it unseeded.
        :returns: None - class instantiated.
        """
        kwargs = self._input_kwargs
        super().__init__(stages=stages)
        self._setDefault(
            checkpointInterval=None,
            cacheIntermediateData=False,
            pruneInputColumns=False,
            cacheEstimatorInput=False,
            fitSampleFraction=None,
            fitSampleSeed=None,
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

    def setCheckpointInterval(self, value: Optional[int]) -> "KamaeSparkPipeline":
        """
        Sets the `checkpointInterval` parameter.

        :param value: Positive number of stages between reliable checkpoint calls
        during fit. None disables checkpointing.
        :returns: KamaeSparkPipeline object with checkpointInterval set.
        :raises ValueError: If value is not None and not a positive integer.
        """
        if value is not None and value < 1:
            raise ValueError(
                "checkpointInterval must be a positive integer or None, got "
                f"{value}."
            )
        return self._set(checkpointInterval=value)

    def getCheckpointInterval(self) -> Optional[int]:
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

    def setCacheEstimatorInput(self, value: bool) -> "KamaeSparkPipeline":
        """
        Sets the `cacheEstimatorInput` parameter.

        :param value: Whether to project and persist a narrow estimator-input
        frame once at the first estimator boundary during fit.
        :returns: KamaeSparkPipeline object with cacheEstimatorInput set.
        """
        return self._set(cacheEstimatorInput=value)

    def getCacheEstimatorInput(self) -> bool:
        """
        Gets the value of the `cacheEstimatorInput` parameter.

        :returns: The cacheEstimatorInput value.
        """
        return self.getOrDefault(self.cacheEstimatorInput)

    def setFitSampleFraction(self, value: Optional[float]) -> "KamaeSparkPipeline":
        """
        Sets the `fitSampleFraction` parameter.

        :param value: Fraction in (0, 1] of the input to sample once up-front and
        fit every estimator from. None disables it.
        :returns: KamaeSparkPipeline object with fitSampleFraction set.
        :raises ValueError: If value is not None and not in the range (0, 1].
        """
        if value is not None and not 0.0 < value <= 1.0:
            raise ValueError(
                f"fitSampleFraction must be in the range (0, 1] or None, got {value}."
            )
        return self._set(fitSampleFraction=value)

    def getFitSampleFraction(self) -> Optional[float]:
        """
        Gets the value of the `fitSampleFraction` parameter.

        :returns: The fitSampleFraction value.
        """
        return self.getOrDefault(self.fitSampleFraction)

    def setFitSampleSeed(self, value: Optional[int]) -> "KamaeSparkPipeline":
        """
        Sets the `fitSampleSeed` parameter.

        :param value: Integer seed for the fitSampleFraction sample, or None.
        :returns: KamaeSparkPipeline object with fitSampleSeed set.
        """
        return self._set(fitSampleSeed=value)

    def getFitSampleSeed(self) -> Optional[int]:
        """
        Gets the value of the `fitSampleSeed` parameter.

        :returns: The fitSampleSeed value.
        """
        return self.getOrDefault(self.fitSampleSeed)

    @keyword_only
    def setParams(
        self,
        *,
        stages: Optional["KamaePipelineStage"] = None,
        checkpointInterval: Optional[int] = None,
        cacheIntermediateData: bool = False,
        pruneInputColumns: bool = False,
        cacheEstimatorInput: bool = False,
        fitSampleFraction: Optional[float] = None,
        fitSampleSeed: Optional[int] = None,
    ) -> "KamaeSparkPipeline":
        """
        Sets the keyword arguments of the pipeline.

        Routes each supplied param through its setter so setter-level validation
        (e.g. checkpointInterval) runs.

        :param stages: List of pipeline stages.
        :param checkpointInterval: Number of stages between reliable
        checkpoint(eager=True) calls during fit. None (default) disables it.
        :param cacheIntermediateData: If True, persist the working DataFrame at
        each estimator-fit boundary. False (default) disables it.
        :param pruneInputColumns: If True, drop input columns no stage consumes
        before fitting. False (default) disables it.
        :param cacheEstimatorInput: If True, project to the columns still read
        downstream at the first estimator boundary and persist that narrow frame
        once for reuse by subsequent estimators. False (default) disables it.
        :param fitSampleFraction: If set to a float in (0, 1], fit every estimator
        from a single up-front persisted sample of the input. None (default)
        disables it.
        :param fitSampleSeed: Optional integer seed for the fitSampleFraction
        sample. None (default) leaves it unseeded.
        :returns: KamaeSparkPipeline object with params set.
        """
        for param_name, param_value in self._input_kwargs.items():
            setter = getattr(self, f"set{param_name[0].upper()}{param_name[1:]}")
            setter(param_value)
        return self

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
        whose name ends in `Col`/`Cols`, so aux columns read during fit (e.g.
        maskCols, relevanceCol, queryIdCol) are not missed. Over-inclusion is
        harmless (names not matching `dataset.columns` are ignored); omission would
        wrongly drop data the pipeline needs at fit time.

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
        if columns_to_keep:
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
        `checkpointInterval`, `cacheIntermediateData`, `cacheEstimatorInput`,
        `fitSampleFraction`); see the class docstring. With the exception of
        `fitSampleFraction`, all preserve data exactly, so fitted results match the
        defaults-off behaviour.

        If both `cacheIntermediateData` and `cacheEstimatorInput` are enabled,
        `cacheEstimatorInput` takes precedence (a warning is emitted) and
        `cacheIntermediateData` is ignored, since the narrow frame is a strictly
        smaller cache and the intermediate cache would evict it.

        If `fitSampleFraction` is set, the input is sampled once, persisted and
        materialised, and every estimator is fit from that shared sample with its
        own `sampleFraction` temporarily disabled; `cacheIntermediateData` and
        `cacheEstimatorInput` are disabled (with a warning) as they persist frames
        this option avoids. Only correct for sample-robust estimators (a runtime
        warning is emitted); see the `fitSampleFraction` param docstring.

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
        cache_estimator_input = self.getCacheEstimatorInput()
        # Competing caching strategies - both would persist the wide frame, and the
        # intermediate cache's per-boundary unpersist would evict the narrow frame.
        # cacheEstimatorInput wins: it persists a strictly narrower frame once.
        if cache_enabled and cache_estimator_input:
            warnings.warn(
                "cacheIntermediateData and cacheEstimatorInput are competing "
                "caching strategies; cacheEstimatorInput takes precedence and "
                "cacheIntermediateData is ignored.",
                stacklevel=2,
            )
            cache_enabled = False

        # Opt-in: fit every estimator from one shared sample drawn up-front, instead
        # of re-scanning or persisting the wide source once per estimator.
        fit_sample_fraction = self.getFitSampleFraction()
        sampled_dataset: Optional[DataFrame] = None
        # (stage, param, was_explicitly_set, original_value) for restoration.
        overridden_sample_fractions: List[Tuple[Any, Any, bool, Any]] = []
        if fit_sample_fraction is not None:
            warnings.warn(
                "fitSampleFraction fits every estimator on one shared sample of the "
                "input. This is only correct when all estimators compute "
                "sample-robust statistics (mean/std/quantiles, e.g. "
                "ConditionalStandardScale). Vocabulary builders "
                "(StringIndexer/OneHot), min/max scalers and distinct counts require "
                "exact/global statistics and will be inaccurate on a sample. It also "
                "replaces the independent per-estimator samples with one shared "
                "sample.",
                stacklevel=2,
            )
            # fitSampleFraction persists a tiny sample instead of the wide frame, so
            # the caching strategies it supersedes are turned off.
            if cache_enabled or cache_estimator_input:
                warnings.warn(
                    "fitSampleFraction is incompatible with cacheIntermediateData "
                    "and cacheEstimatorInput (which persist frames this option "
                    "avoids); the caching options are disabled.",
                    stacklevel=2,
                )
                cache_enabled = False
                cache_estimator_input = False
            sampled_dataset = dataset.sample(
                fraction=fit_sample_fraction, seed=self.getFitSampleSeed()
            ).persist(StorageLevel.MEMORY_AND_DISK)
            # Force one materialisation so the sample is computed exactly once and
            # every estimator fit reads the cached rows instead of re-scanning.
            sampled_dataset.count()
            dataset = sampled_dataset
            # The shared sample is already drawn, so each estimator must not sample
            # again (fraction-of-a-fraction would leave far too few rows). Disable
            # each estimator's sampleFraction for this fit, restoring it in finally.
            for stage in expanded_pipeline_stages:
                if isinstance(stage, BaseEstimator) and stage.hasParam(
                    "sampleFraction"
                ):
                    param = stage.getParam("sampleFraction")
                    was_set = stage.isSet(param)
                    original = stage.getOrDefault(param) if was_set else None
                    overridden_sample_fractions.append(
                        (stage, param, was_set, original)
                    )
                    stage.set(param, None)

        # Fit each stage, appending the transformer to the list of transformers.
        # If the stage is a parent of an estimator, transform the dataset.
        transformers: List[BaseTransformer] = []
        try:
            fitted_pipeline_model = self._run_fit_loop(
                expanded_pipeline_stages=expanded_pipeline_stages,
                dataset=dataset,
                estimator_parent_stages=estimator_parent_stages,
                transformers=transformers,
                checkpoint_enabled=checkpoint_enabled,
                checkpoint_interval=checkpoint_interval,
                cache_enabled=cache_enabled,
                cache_estimator_input=cache_estimator_input,
            )
        finally:
            # Restore each estimator's original sampleFraction and release the
            # shared sample, whether or not the fit succeeded.
            for stage, param, was_set, original in overridden_sample_fractions:
                if was_set:
                    stage.set(param, original)
                else:
                    stage.clear(param)
            if sampled_dataset is not None:
                sampled_dataset.unpersist()
        return fitted_pipeline_model

    def _run_fit_loop(
        self,
        *,
        expanded_pipeline_stages: List["KamaePipelineStage"],
        dataset: DataFrame,
        estimator_parent_stages: List["KamaePipelineStage"],
        transformers: List[BaseTransformer],
        checkpoint_enabled: bool,
        checkpoint_interval: Optional[int],
        cache_enabled: bool,
        cache_estimator_input: bool,
    ) -> "KamaeSparkPipelineModel":
        """
        Runs the stage-by-stage fit loop, applying the checkpoint/cache optimisations.

        Extracted from `_fit` so the loop can run inside a try/finally that restores
        estimator sampling and releases the shared sample when fitSampleFraction is
        used. Behaviour is identical to the previous inline loop.

        `cacheIntermediateData` and `cacheEstimatorInput` are mutually exclusive (see
        `_fit`), so a single `persisted_frame` handle tracks whichever frame is
        persisted. It is kept separate from `dataset` because `dataset` is reassigned
        by `model.transform(...)` between boundaries; the handle is what lets us
        unpersist the actual persisted frame at the end.

        :param expanded_pipeline_stages: Flattened pipeline stages to fit.
        :param dataset: DataFrame (possibly sampled) to fit the stages against.
        :param estimator_parent_stages: Stages whose output an estimator consumes.
        :param transformers: Accumulator list the fitted stages are appended to.
        :param checkpoint_enabled: Whether reliable checkpointing is enabled.
        :param checkpoint_interval: Stages between checkpoints (when enabled).
        :param cache_enabled: Whether to persist at each estimator boundary.
        :param cache_estimator_input: Whether to persist a narrow frame once.
        :returns: KamaeSparkPipelineModel object.
        """
        last_checkpoint_index = 0
        # The single persisted frame (if any), unpersisted once superseded or done.
        # Only one of the mutually-exclusive cache strategies ever populates it.
        persisted_frame: Optional[DataFrame] = None
        estimator_input_cached = False
        for index, stage in enumerate(expanded_pipeline_stages):
            if isinstance(stage, BaseTransformer):
                transformers.append(stage)
                if stage in estimator_parent_stages:
                    dataset = stage.transform(dataset)
            else:
                # cacheEstimatorInput: at the first estimator boundary, project to
                # the columns still read downstream and persist that narrow frame
                # once. Independent sibling estimators then fit against it instead of
                # re-scanning the wide input. The persist sits below each estimator's
                # in-fit sample, so sampling is unchanged.
                if cache_estimator_input and not estimator_input_cached:
                    estimator_input_cached = True
                    live_columns = self.collect_required_input_columns(
                        expanded_pipeline_stages[index:]
                    )
                    keep_columns = [c for c in dataset.columns if c in live_columns]
                    if keep_columns and len(keep_columns) < len(dataset.columns):
                        dataset = dataset.select(*keep_columns).persist(
                            StorageLevel.MEMORY_AND_DISK
                        )
                        persisted_frame = dataset
                # Truncate accumulated lineage before the fit action to bound plan
                # depth. eager=True materialises now.
                if (
                    checkpoint_enabled
                    and index - last_checkpoint_index >= checkpoint_interval
                ):
                    dataset = dataset.checkpoint(eager=True)
                    last_checkpoint_index = index
                # cacheIntermediateData: persist so the fit action and downstream
                # transforms reuse a materialised frame instead of re-scanning,
                # releasing the previous frame first. One frame held at a time.
                if cache_enabled:
                    if persisted_frame is not None:
                        persisted_frame.unpersist()
                    dataset = dataset.persist(StorageLevel.MEMORY_AND_DISK)
                    persisted_frame = dataset
                model = stage.fit(dataset)
                transformers.append(model)
                if stage in estimator_parent_stages:
                    dataset = model.transform(dataset)
        if persisted_frame is not None:
            persisted_frame.unpersist()
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
