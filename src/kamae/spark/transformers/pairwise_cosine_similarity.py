from typing import List, Optional

import pyspark.sql.functions as F
import tensorflow as tf
from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, DataType, DoubleType, FloatType

from kamae.spark.params import MultiInputSingleOutputParams
from kamae.tensorflow.layers import PairwiseCosineSimilarityLayer

from .base import BaseTransformer


class PairwiseCosineSimilarityTransformer(
    BaseTransformer,
    MultiInputSingleOutputParams,
):
    """
    Computes pairwise cosine similarity between a query embedding and each
    candidate embedding packed into a flat array.

    Input 0: query embedding as Array[Float] of size D.
    Input 1: flat candidate embeddings as Array[Float] of size N*D.
    Output:  Array[Float] of size N containing cosine similarities.
    """

    embeddingDim = Param(
        Params._dummy(),
        "embeddingDim",
        "Dimension of each embedding vector.",
        typeConverter=TypeConverters.toInt,
    )

    @keyword_only
    def __init__(
        self,
        inputCols: Optional[List[str]] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        embeddingDim: Optional[int] = None,
    ) -> None:
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    def setEmbeddingDim(self, value: int) -> "PairwiseCosineSimilarityTransformer":
        return self._set(embeddingDim=value)

    def getEmbeddingDim(self) -> int:
        return self.getOrDefault(self.embeddingDim)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        return [FloatType(), DoubleType()]

    def setInputCols(self, value: List[str]) -> "PairwiseCosineSimilarityTransformer":
        if len(value) != 2:
            raise ValueError(
                f"Expected 2 input columns, received {len(value)} instead."
            )
        return self._set(inputCols=value)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        input_col_names = self.getInputCols()
        embedding_dim = self.getEmbeddingDim()

        query_col = F.col(input_col_names[0])
        flat_candidates_col = F.col(input_col_names[1])

        for col_name in input_col_names:
            dtype = self.get_column_datatype(dataset=dataset, column_name=col_name)
            if not isinstance(dtype, ArrayType):
                raise TypeError(
                    f"Expected ArrayType for {col_name}, got {dtype}."
                )

        num_candidates = (
            F.size(flat_candidates_col) / F.lit(embedding_dim)
        ).cast("int")
        indices = F.sequence(F.lit(0), num_candidates - F.lit(1))

        query_norm = F.sqrt(
            F.aggregate(
                query_col,
                F.lit(0.0).cast("double"),
                lambda acc, x: acc + (x * x).cast("double"),
            )
        )

        def cosine_sim_at_index(idx):
            candidate = F.slice(
                flat_candidates_col,
                idx * F.lit(embedding_dim) + F.lit(1),
                embedding_dim,
            )
            zipped = F.arrays_zip(query_col.alias("q"), candidate.alias("c"))
            dot = F.aggregate(
                zipped,
                F.lit(0.0).cast("double"),
                lambda acc, pair: acc + (pair["q"] * pair["c"]).cast("double"),
            )
            cand_norm = F.sqrt(
                F.aggregate(
                    candidate,
                    F.lit(0.0).cast("double"),
                    lambda acc, x: acc + (x * x).cast("double"),
                )
            )
            return F.coalesce(dot / (query_norm * cand_norm), F.lit(0.0))

        similarities = F.transform(indices, cosine_sim_at_index)
        return dataset.withColumn(self.getOutputCol(), similarities)

    def get_tf_layer(self) -> tf.keras.layers.Layer:
        return PairwiseCosineSimilarityLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputTFDtype(),
            output_dtype=self.getOutputTFDtype(),
            embedding_dim=self.getEmbeddingDim(),
        )