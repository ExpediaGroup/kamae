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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from pyspark.ml import Estimator
from pyspark.sql import Column, DataFrame
from pyspark.storagelevel import StorageLevel

from kamae.spark.common import SparkOperation
from kamae.spark.transformers import BaseTransformer

if TYPE_CHECKING:
    from pyspark.ml._typing import ParamMap


class BaseEstimator(Estimator, SparkOperation):
    def __init__(self) -> None:
        """
        Initializes the estimator.
        """
        super().__init__()

    def fit(
        self,
        dataset: DataFrame,
        params: Optional[Union["ParamMap", List["ParamMap"], Tuple["ParamMap"]]] = None,
    ) -> BaseTransformer:
        """
        Overrides the fit method of the parent class to add casting of input columns
        to the preferred data type.

        :param dataset: Input dataset.
        :param params: Optional additional parameters.
        :returns: Fitted transformer companion object.
        """
        try:
            dataset = self.prepare_fit_dataset(dataset)

            if self.hasParam("sampleFraction"):
                frac = self.getSampleFraction()
                if frac is not None:
                    dataset = dataset.sample(fraction=frac)

            # Replicate the logic from the existing abstract estimator fit method
            transformer = super().fit(dataset, params)

            self.finish_fit(dataset, transformer)
            return transformer

        except Exception as e:
            raise self.with_error_context(e)

    def with_error_context(self, e: Exception) -> Exception:
        """
        Returns the exception re-created with this estimator's uid and params in its
        message, keeping the original traceback.

        :param e: The exception raised while fitting this estimator.
        :returns: The exception to raise.
        """
        param_dict = {
            param[0].name: param[1] for param in self.extractParamMap().items()
        }
        return e.__class__(
            f"Error in estimator: {self.uid} with params: {param_dict}"
        ).with_traceback(e.__traceback__)

    def prepare_fit_dataset(self, dataset: DataFrame) -> DataFrame:
        """
        Adds the casted input columns to the dataset and points the estimator's
        input columns at them. First step of `fit`.

        :param dataset: Input dataset.
        :returns: The dataset with the casted input columns added.
        """
        dataset = self._create_casted_input_output_columns(
            dataset=dataset, ingress=True
        )
        self._check_input_dtypes_compatible(
            dataset, self._get_single_or_multi_col(ingress=True)
        )

        # Set estimator input columns to casted columns
        self.set_input_columns_to_from_casted(
            dataset=dataset,
            suffix=self.tmp_column_suffix,
        )
        return dataset

    def reset_fit_input_columns(self, dataset: DataFrame) -> None:
        """
        Points the estimator's input columns back at the original (uncasted) columns
        after `prepare_fit_dataset`.

        :param dataset: The dataset returned by `prepare_fit_dataset`.
        :returns: None
        """
        # Reset input columns from casted columns
        self.set_input_columns_to_from_casted(
            dataset=dataset,
            suffix=self.tmp_column_suffix,
            reverse=True,
        )

    def finish_fit(self, dataset: DataFrame, transformer: BaseTransformer) -> None:
        """
        Points the input columns of the estimator and of its fitted transformer back
        at the original (uncasted) columns. Last step of `fit`.

        :param dataset: The dataset returned by `prepare_fit_dataset`.
        :param transformer: The fitted transformer.
        :returns: None
        """
        self.reset_fit_input_columns(dataset)

        # Reset input columns for transformer
        transformer.set_input_columns_to_from_casted(
            dataset=dataset,
            suffix=self.tmp_column_suffix,
            reverse=True,
        )

    def can_fit_by_aggregation(self, dataset: DataFrame) -> bool:
        """
        Whether this estimator fits as one aggregation over the dataset's rows: part
        one (`fit_aggregations`) builds the aggregation, part two (`fit_from_row`)
        builds the fitted transformer from its result row. Such estimators can be
        fitted together in one aggregation (see `KamaeSparkPipeline`
        `fuseEstimatorFits`).

        :param dataset: The input dataset.
        :returns: False by default. Estimators implementing both parts override it.
        """
        return False

    def fit_aggregations(
        self, dataset: DataFrame, prefix: str
    ) -> Tuple[Dict[str, Column], List[Column]]:
        """
        Part one of fitting by aggregation: everything before the aggregation.

        It may run small actions of its own (e.g. reading the first row to get an
        array size), but the aggregation must be over the dataset's rows as they
        are: helper columns may be added, rows may not be filtered, exploded or
        sampled (use conditions inside the aggregations instead).

        :param dataset: The dataset returned by `prepare_fit_dataset`.
        :param prefix: Prefix every helper column name and aggregation alias starts
        with, so several estimators can share one aggregation.
        :returns: Tuple of the helper columns to add (name -> column) and the
        aliased aggregations.
        """
        raise NotImplementedError(f"{type(self).__name__} does not fit by aggregation.")

    def fit_from_row(self, row: Dict[str, Any], prefix: str) -> BaseTransformer:
        """
        Part two of fitting by aggregation: builds the fitted transformer from the
        aggregation result row. Runs on the driver only.

        :param row: The aggregation result row, as a dict of alias -> value.
        :param prefix: The prefix given to `fit_aggregations`.
        :returns: The fitted transformer.
        """
        raise NotImplementedError(f"{type(self).__name__} does not fit by aggregation.")

    def _fit_by_aggregation(
        self, dataset: DataFrame, persist: bool = True
    ) -> BaseTransformer:
        """
        The normal fit of an estimator that fits by aggregation: part one, the
        aggregation, part two.

        :param dataset: The dataset returned by `prepare_fit_dataset`.
        :param persist: Whether to persist the dataset (MEMORY_AND_DISK) for the
        fit, so actions of part one (e.g. reading the first row) and the aggregation
        reuse a materialised result. Skipped if the dataset is already cached.
        :returns: The fitted transformer.
        """
        # Guarded so we do not double-persist data the caller already cached.
        already_cached = dataset.storageLevel.useMemory or dataset.storageLevel.useDisk
        persisted = persist and not already_cached
        if persisted:
            dataset = dataset.persist(StorageLevel.MEMORY_AND_DISK)
        try:
            columns, aggregations = self.fit_aggregations(dataset, prefix="")
            with_columns = dataset.withColumns(columns) if columns else dataset
            return self.fit_from_row(
                with_columns.agg(*aggregations).first().asDict(), prefix=""
            )
        finally:
            if persisted:
                dataset.unpersist()

    def construct_layer_info(self) -> Dict[str, Any]:
        """
        Constructs the layer info dictionary.
        Contains the layer name, the tensorflow layer, and the inputs and outputs.
        This is used when constructing the pipeline graph.

        layer is set to None because estimators do not have a defined tf computation,
        we will not use this information in the pipeline graph (for estimators).

        :returns: Dictionary containing layer information such as
        name, tensorflow layer, inputs, and outputs.
        """
        inputs, outputs = self.get_layer_inputs_outputs()
        return {
            "name": self.getOrDefault("layerName"),
            # Estimators do not have a defined tf computation, only transformers do.
            "layer": None,
            "inputs": inputs,
            "outputs": outputs,
        }
