from typing import List, Optional

import pyspark.sql.functions as F
import tensorflow as tf
from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType, DoubleType, FloatType

from kamae.spark.params import SingleInputSingleOutputParams
from kamae.spark.utils import single_input_single_output_array_transform
from kamae.tensorflow.layers import ArrayReduceMaxLayer

from .base import BaseTransformer


class ArrayReduceMaxTransformer(
    BaseTransformer,
    SingleInputSingleOutputParams,
):
    """
    Reduces an array column to its maximum element.

    Input:  Array[Float/Double] of size N.
    Output: Float/Double scalar (the maximum element).

    Returns defaultValue when the array is empty or null.
    """

    defaultValue = Param(
        Params._dummy(),
        "defaultValue",
        "Value to return when the array is empty or null.",
        typeConverter=TypeConverters.toFloat,
    )

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        defaultValue: float = 0.0,
    ) -> None:
        super().__init__()
        self._setDefault(defaultValue=0.0)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    def setDefaultValue(self, value: float) -> "ArrayReduceMaxTransformer":
        return self._set(defaultValue=value)

    def getDefaultValue(self) -> float:
        return self.getOrDefault(self.defaultValue)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        return [FloatType(), DoubleType()]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        input_col = F.col(self.getInputCol())
        default = self.getDefaultValue()

        output_col = single_input_single_output_array_transform(
            input_col=input_col,
            input_col_datatype=self.get_column_datatype(
                dataset=dataset, column_name=self.getInputCol()
            ),
            func=lambda x: F.coalesce(F.array_max(x), F.lit(default)),
        )
        return dataset.withColumn(self.getOutputCol(), output_col)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        return ArrayReduceMaxLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            default_value=self.getDefaultValue(),
        )