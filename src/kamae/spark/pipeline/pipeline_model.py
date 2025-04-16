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

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union, cast

import keras_tuner as kt
import tensorflow as tf
from pyspark.ml import PipelineModel
from pyspark.ml.pipeline import (
    PipelineModelReader,
    PipelineModelWriter,
    PipelineSharedReadWrite,
)
from pyspark.ml.util import DefaultParamsReader, MLWriter

from kamae.graph import PipelineGraph
from kamae.spark.transformers import BaseTransformer

if TYPE_CHECKING:
    from pyspark.ml._typing import ParamMap


class KamaeSparkPipelineModel(PipelineModel):
    """
    KamaeSparkPipelineModel is a subclass of pyspark.ml.PipelineModel that is used to
    chain together LayerTransformers. It maintains the same functionality
    as pyspark.ml.PipelineModel e.g. serialisation.
    """

    def __init__(self, stages: List[BaseTransformer]):
        """
        Initialises the KamaeSparkPipelineModel object.

        :param stages: List of LayerTransformers to chain together.
        """
        super().__init__(stages=stages)
        self.stages = stages

    def copy(self, extra: Optional["ParamMap"] = None) -> "KamaeSparkPipelineModel":
        """
        Creates a copy of the KamaeSparkPipelineModel object.

        :param extra: Additional optional params to copy to new pipeline model.
        :returns: KamaeSparkPipelineModel object.
        """
        if extra is None:
            extra = dict()
        stages = [stage.copy(extra) for stage in self.stages]
        return KamaeSparkPipelineModel(stages)

    def write(self) -> MLWriter:
        """
        Uses the KamaeSparkPipelineModelWriter class to write the pipeline model to a
        persistent storage path.

        :returns: KamaeSparkPipelineModelWriter object.
        """
        return KamaeSparkPipelineModelWriter(self)

    @classmethod
    def read(cls) -> "KamaeSparkPipelineModelReader":
        """
        Uses the KamaeSparkPipelineModelReader class to read a pipeline model from a
        persistent storage path.

        :returns: KamaeSparkPipelineModelReader object.
        """
        return KamaeSparkPipelineModelReader(cls)

    def get_all_tf_layers(self) -> List[tf.keras.layers.Layer]:
        """
        Gets a list of all tensorflow layers in the pipeline model.

        :returns: List of tensorflow layers within the pipeline model.
        """
        return [stage.get_tf_layer() for stage in self.stages]

    def expand_pipeline_stages(self) -> List[BaseTransformer]:
        """
        Expands the pipeline stages to include all nested pipeline stages.
        If the pipeline stage is itself a pipeline model, it will be expanded
        recursively.

        :returns: List of all pipeline stages flattened to transformer level.
        """
        expanded_stages = []
        for stage in self.stages:
            if isinstance(stage, KamaeSparkPipelineModel):
                # Recursively expand the pipeline stages.
                expanded_stages.extend(stage.expand_pipeline_stages())
            else:
                expanded_stages.append(stage)
        return expanded_stages

    def build_keras_model(
        self, tf_input_schema: Union[List[tf.TypeSpec], List[Dict[str, Any]]]
    ) -> tf.keras.Model:
        """
        Builds a keras model from the pipeline model using the PipelineGraph
        helper class.

        :param tf_input_schema: List of dictionaries containing the input schema for
        the model. Specifically the name, shape and dtype of each input.
        These will be passed as is to the Keras Input layer.
        :returns: Keras model.
        """
        stage_dict = {
            stage.getOrDefault("layerName"): stage.construct_layer_info()
            for stage in self.expand_pipeline_stages()
        }
        pipeline_graph = PipelineGraph(stage_dict=stage_dict)
        return pipeline_graph.build_keras_model(tf_input_schema=tf_input_schema)

    def get_keras_tuner_model_builder(
        self,
        tf_input_schema: Union[List[tf.TypeSpec], List[Dict[str, Any]]],
        hp_dict: Dict[str, List[Dict[str, Any]]],
    ) -> Callable[[kt.HyperParameters], tf.keras.Model]:
        """
        Builds a keras tuner model builder (function) from the pipeline model
        using the PipelineGraph helper class.
        """
        stage_dict = {
            stage.getOrDefault("layerName"): stage.construct_layer_info()
            for stage in self.stages
        }
        pipeline_graph = PipelineGraph(stage_dict=stage_dict)
        return pipeline_graph.get_keras_tuner_model_builder(
            tf_input_schema=tf_input_schema, hp_dict=hp_dict
        )


class KamaeSparkPipelineModelReader(PipelineModelReader):
    """
    Util class for reading a pipeline model from a persistent storage path.
    """

    def __init__(self, cls: Type["KamaeSparkPipelineModel"]):
        super().__init__(cls=cls)

    def load(self, path: str) -> "KamaeSparkPipelineModel":
        """
        Loads a pipeline model from a given path.

        :param path: Path to stored pipeline model.
        :returns: KamaeSparkPipelineModel object.
        """
        metadata = DefaultParamsReader.loadMetadata(path, self.sc)
        uid, stages = PipelineSharedReadWrite.load(metadata, self.sc, path)
        return KamaeSparkPipelineModel(
            stages=cast(List[BaseTransformer], stages)
        )._resetUid(uid)


class KamaeSparkPipelineModelWriter(PipelineModelWriter):
    """
    Util class for writing a pipeline model to a persistent storage path.
    """

    def __init__(self, instance: "KamaeSparkPipelineModel"):
        super().__init__(instance=instance)
