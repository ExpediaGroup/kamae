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

import os
import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple, Type

import networkx as nx
from pyspark import keyword_only
from pyspark.ml import Pipeline
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.pipeline import PipelineReader, PipelineSharedReadWrite, PipelineWriter
from pyspark.ml.util import DefaultParamsReader, DefaultParamsWriter, MLWriter
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
    draws a single persisted sample of the input up-front and fits the estimators
    that opt in (those with `useFitSample=True`) from that shared sample, while
    estimators without it keep fitting on the full input.
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
        "persist (MEMORY_AND_DISK) and materialise it once, and fit the estimators "
        "that opt in from that shared sample. An estimator opts in by setting its "
        "boolean useFitSample param to True. Estimators without useFitSample fit on "
        "the full input, so leave it False for estimators needing exact/global "
        "statistics (vocabulary builders like StringIndexer/OneHot, min/max scalers, "
        "distinct counts) and set it True on sample-robust estimators (mean/std/"
        "quantiles, e.g. ConditionalStandardScale). Avoids re-scanning or persisting "
        "the wide source once per opted-in estimator. Disables cacheIntermediateData "
        "and cacheEstimatorInput. None (default) disables it.",
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
        :param fitSampleFraction: If set to a float in (0, 1], fit the estimators
        that opt in (those with useFitSample=True) from a single up-front persisted
        sample of the input; estimators without useFitSample fit on the full input.
        None (default) disables it.
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
        materialised, and the estimators that opt in (those with `useFitSample=True`)
        are fit from that shared sample with any `sampleFraction` they also set
        temporarily disabled; every other estimator still fits on the full input.
        `cacheIntermediateData` and `cacheEstimatorInput` are disabled (with a
        warning) as they persist frames this option avoids; a warning also names the
        opted-in estimators. See the `fitSampleFraction` param docstring.

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

        # Opt-in: draw one shared sample up-front and fit only the estimators that
        # opt in against it, instead of each re-scanning or persisting the wide
        # source. An estimator opts in via its boolean `useFitSample` param.
        # Estimators without it set still fit on the full input, so exact/global-
        # statistic estimators (vocabulary builders, min/max) stay correct.
        fit_sample_fraction = self.getFitSampleFraction()
        sampled_dataset: Optional[DataFrame] = None
        # (stage, param, original_value) for restoration of overridden sampleFractions.
        overridden_sample_fractions: List[Tuple[Any, Any, Any]] = []
        # Identities of the estimators that fit on the shared sample.
        use_sample_stage_ids: Set[int] = set()
        if fit_sample_fraction is not None:
            sampling_estimators = [
                stage
                for stage in expanded_pipeline_stages
                if isinstance(stage, BaseEstimator)
                and stage.hasParam("useFitSample")
                and stage.getUseFitSample()
            ]
            if not sampling_estimators:
                warnings.warn(
                    "fitSampleFraction is set but no estimator has useFitSample=True, "
                    "so nothing opts in to the shared sample and fitSampleFraction "
                    "has no effect. Set useFitSample=True on the estimators that "
                    "should fit on the sample.",
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "fitSampleFraction fits "
                    f"{sorted({type(s).__name__ for s in sampling_estimators})} on "
                    "one shared pipeline sample (useFitSample=True). Estimators "
                    "without useFitSample set fit on the full input.",
                    stacklevel=2,
                )
                # fitSampleFraction persists a tiny sample instead of the wide frame,
                # so the caching strategies it supersedes are turned off.
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
                # every opted-in estimator reads the cached rows instead of rescanning.
                sampled_dataset.count()
                # The shared sample is already drawn, so opted-in estimators must not
                # sample again (fraction-of-a-fraction would leave far too few rows).
                # Disable any sampleFraction they also set, restoring it in finally.
                for stage in sampling_estimators:
                    use_sample_stage_ids.add(id(stage))
                    param = stage.getParam("sampleFraction")
                    if stage.isSet(param):
                        overridden_sample_fractions.append(
                            (stage, param, stage.getOrDefault(param))
                        )
                        stage.set(param, None)
                overridden_names = sorted(
                    {type(s).__name__ for s, _, _ in overridden_sample_fractions}
                )
                if overridden_names:
                    warnings.warn(
                        f"{overridden_names} set both useFitSample=True and their own "
                        "sampleFraction. The shared pipeline sample wins, so their "
                        "sampleFraction is ignored for this fit. Unset one to silence "
                        "this warning.",
                        stacklevel=2,
                    )
        else:
            opted_in_without_sample = sorted(
                {
                    type(stage).__name__
                    for stage in expanded_pipeline_stages
                    if isinstance(stage, BaseEstimator)
                    and stage.hasParam("useFitSample")
                    and stage.getUseFitSample()
                }
            )
            if opted_in_without_sample:
                warnings.warn(
                    f"{opted_in_without_sample} set useFitSample=True but the pipeline "
                    "has no fitSampleFraction, so there is no shared sample to fit on "
                    "and useFitSample has no effect. Set fitSampleFraction on the "
                    "pipeline to enable sampled fitting.",
                    stacklevel=2,
                )

        # Fit each stage, appending the transformer to the list of transformers.
        # If the stage is a parent of an estimator, transform the dataset.
        transformers: List[BaseTransformer] = []
        try:
            fitted_pipeline_model = self._run_fit_loop(
                expanded_pipeline_stages=expanded_pipeline_stages,
                dataset=dataset,
                sampled_dataset=sampled_dataset,
                use_sample_stage_ids=use_sample_stage_ids,
                estimator_parent_stages=estimator_parent_stages,
                transformers=transformers,
                checkpoint_enabled=checkpoint_enabled,
                checkpoint_interval=checkpoint_interval,
                cache_enabled=cache_enabled,
                cache_estimator_input=cache_estimator_input,
            )
        finally:
            # Restore each opted-in estimator's original sampleFraction and release
            # the shared sample, whether or not the fit succeeded.
            for stage, param, original in overridden_sample_fractions:
                stage.set(param, original)
            if sampled_dataset is not None:
                sampled_dataset.unpersist()
        return fitted_pipeline_model

    def _run_fit_loop(
        self,
        *,
        expanded_pipeline_stages: List["KamaePipelineStage"],
        dataset: DataFrame,
        sampled_dataset: Optional[DataFrame],
        use_sample_stage_ids: Set[int],
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
        used. With fitSampleFraction off (`sampled_dataset is None`) behaviour is
        identical to the previous inline loop.

        When fitSampleFraction is active, two lineages are carried in parallel: the
        full-data lineage and a shared-sample lineage. Every transformer that feeds an
        estimator is applied to both, so an estimator can fit on whichever it opted
        in to - estimators in `use_sample_stage_ids` fit on the sample, all others on
        the full input. The caching options are mutually exclusive with
        fitSampleFraction (disabled in `_fit`), so they only ever act on the full
        lineage when no sample is present.

        `cacheIntermediateData` and `cacheEstimatorInput` are mutually exclusive (see
        `_fit`), so a single `persisted_frame` handle tracks whichever frame is
        persisted. It is kept separate from the lineage variable because that variable
        is reassigned by `model.transform(...)` between boundaries; the handle is what
        lets us unpersist the actual persisted frame at the end.

        :param expanded_pipeline_stages: Flattened pipeline stages to fit.
        :param dataset: Full-data DataFrame to fit non-sampled stages against.
        :param sampled_dataset: Shared sample (or None when fitSampleFraction is off).
        :param use_sample_stage_ids: `id()`s of estimators that fit on the sample.
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
        # Full-data lineage and, when fitSampleFraction is active, a parallel
        # shared-sample lineage kept in lock-step through the estimator-feeding
        # transforms.
        full_dataset = dataset
        sample_dataset = sampled_dataset
        for index, stage in enumerate(expanded_pipeline_stages):
            if isinstance(stage, BaseTransformer):
                transformers.append(stage)
                if stage in estimator_parent_stages:
                    full_dataset = stage.transform(full_dataset)
                    if sample_dataset is not None:
                        sample_dataset = stage.transform(sample_dataset)
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
                    keep_columns = [
                        c for c in full_dataset.columns if c in live_columns
                    ]
                    if keep_columns and len(keep_columns) < len(full_dataset.columns):
                        full_dataset = full_dataset.select(*keep_columns).persist(
                            StorageLevel.MEMORY_AND_DISK
                        )
                        persisted_frame = full_dataset
                # Truncate accumulated lineage before the fit action to bound plan
                # depth. eager=True materialises now.
                if (
                    checkpoint_enabled
                    and index - last_checkpoint_index >= checkpoint_interval
                ):
                    full_dataset = full_dataset.checkpoint(eager=True)
                    if sample_dataset is not None:
                        sample_dataset = sample_dataset.checkpoint(eager=True)
                    last_checkpoint_index = index
                # cacheIntermediateData: persist so the fit action and downstream
                # transforms reuse a materialised frame instead of re-scanning,
                # releasing the previous frame first. One frame held at a time.
                if cache_enabled:
                    if persisted_frame is not None:
                        persisted_frame.unpersist()
                    full_dataset = full_dataset.persist(StorageLevel.MEMORY_AND_DISK)
                    persisted_frame = full_dataset
                fit_dataset = (
                    sample_dataset
                    if sample_dataset is not None and id(stage) in use_sample_stage_ids
                    else full_dataset
                )
                model = stage.fit(fit_dataset)
                transformers.append(model)
                if stage in estimator_parent_stages:
                    full_dataset = model.transform(full_dataset)
                    if sample_dataset is not None:
                        sample_dataset = model.transform(sample_dataset)
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
        pipeline = KamaeSparkPipeline(stages=stages)._resetUid(uid)
        # The base pipeline writer only persists stage uids, so the pipeline-level
        # fit params (checkpointInterval, cache/prune flags, fitSampleFraction, ...)
        # would reset to defaults. Restore any that were saved by the writer.
        saved_params = metadata.get("kamaePipelineParams", {})
        for name, value in saved_params.items():
            if pipeline.hasParam(name):
                pipeline.set(pipeline.getParam(name), value)
        return pipeline


class KamaeSparkPipelineWriter(PipelineWriter):
    """
    Util class for writing a pipeline to a persistent storage path.
    """

    def __init__(self, instance: KamaeSparkPipeline) -> None:
        super().__init__(instance=instance)

    def saveImpl(self, path: str) -> None:
        """
        Saves the pipeline to the given path.

        Mirrors PipelineSharedReadWrite.saveImpl (metadata + stage uids + stages) but
        additionally persists the pipeline-level fit params (which the base writer
        drops) so they survive a save/load round-trip.

        :param path: Path to store the pipeline at.
        :returns: None.
        """
        stages = self.instance.getStages()
        PipelineSharedReadWrite.validateStages(stages)

        json_params = {
            "stageUids": [stage.uid for stage in stages],
            "language": "Python",
        }
        # Only the explicitly-set, non-stages params; defaults stay implicit so
        # older saves (without this metadata) still load with correct defaults.
        pipeline_params = {
            p.name: self.instance.getOrDefault(p)
            for p in self.instance.params
            if p.name != "stages" and self.instance.isSet(p)
        }
        DefaultParamsWriter.saveMetadata(
            self.instance,
            path,
            self.sc,
            extraMetadata={"kamaePipelineParams": pipeline_params},
            paramMap=json_params,
        )
        stages_dir = os.path.join(path, "stages")
        for index, stage in enumerate(stages):
            stage.write().save(
                PipelineSharedReadWrite.getStagePath(
                    stage.uid, index, len(stages), stages_dir
                )
            )
