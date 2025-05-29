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

"""Literally a copy of pyspark.ml.util DefaultParamsWritable/Readable/Writer/Reader.
This is needed because Databricks runtimes > 13.3LTS have slow metadata writes because
they have introduced this v4.0.0 change: https://github.com/apache/spark/pull/47453
They have moved to using atomic writes for metadata, however this is intolerably slow
for large pipelines. We temporarily fix this back to the open source code.
This may need revisiting in the future."""
import os
from typing import Any, Dict, Optional

from pyspark import SparkContext
from pyspark.ml.param import Params
from pyspark.ml.util import (
    DefaultParamsReadable,
    DefaultParamsReader,
    DefaultParamsWritable,
    DefaultParamsWriter,
    MLWriter,
)


class KamaeDefaultParamsWriter(DefaultParamsWriter):
    """
    DefaultParamsWriter with a workaround for slow metadata writes in Databricks.
    Replicates the functionality of DefaultParamsWriter in PySpark 3.5.0 since
    Databricks uses different functionality
    """

    @staticmethod
    def saveMetadata(
        instance: "Params",
        path: str,
        sc: SparkContext,
        extraMetadata: Optional[Dict[str, Any]] = None,
        paramMap: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Saves metadata + Params to: path + "/metadata"
        - class
        - timestamp
        - sparkVersion
        - uid
        - paramMap
        - defaultParamMap (since 2.4.0)
        - (optionally, extra metadata)
        Parameters
        ----------
        extraMetadata : dict, optional
            Extra metadata to be saved at same level as uid, paramMap, etc.
        paramMap : dict, optional
            If given, this is saved in the "paramMap" field.
        """
        metadataPath = os.path.join(path, "metadata")
        metadataJson = DefaultParamsWriter._get_metadata_to_save(
            instance, sc, extraMetadata, paramMap
        )
        # This is the line we need to maintain.
        sc.parallelize([metadataJson], 1).saveAsTextFile(metadataPath)


class KamaeDefaultParamsWritable(DefaultParamsWritable):
    """
    DefaultParamsWritable with a workaround for slow metadata writes in Databricks.
    Replicates the functionality of DefaultParamsWritable in PySpark 3.5.0 since
    Databricks uses different functionality
    """

    def write(self) -> MLWriter:
        """Returns a DefaultParamsWriter instance for this class."""
        from pyspark.ml.param import Params

        if isinstance(self, Params):
            return KamaeDefaultParamsWriter(self)
        else:
            raise TypeError(
                """Cannot use KamaeDefaultParamsWritable with type %s because it does
                not extend Params.""",
                type(self),
            )


class KamaeDefaultParamsReader(DefaultParamsReader):
    """
    DefaultParamsReadable with a workaround for slow metadata writes in Databricks.
    Replicates the functionality of DefaultParamsReadable in PySpark 3.5.0 since
    Databricks uses different functionality
    """

    @staticmethod
    def loadMetadata(
        path: str, sc: SparkContext, expectedClassName: str = ""
    ) -> Dict[str, Any]:
        """
        Load metadata saved using :py:meth:`DefaultParamsWriter.saveMetadata`
        Parameters
        ----------
        path : str
        sc : :py:class:`pyspark.SparkContext`
        expectedClassName : str, optional
            If non empty, this is checked against the loaded metadata.
        """
        metadataPath = os.path.join(path, "metadata")
        # This is the line we need to maintain.
        metadataStr = sc.textFile(metadataPath, 1).first()
        loadedVals = DefaultParamsReader._parseMetaData(metadataStr, expectedClassName)
        return loadedVals


class KamaeDefaultParamsReadable(DefaultParamsReadable):
    """
    DefaultParamsReadable with a workaround for slow metadata writes in Databricks.
    Replicates the functionality of DefaultParamsReadable in PySpark 3.5.0 since
    Databricks uses different functionality
    """

    @classmethod
    def read(cls) -> "KamaeDefaultParamsReader":
        """Returns a KamaeDefaultParamsReader instance for this class."""
        return KamaeDefaultParamsReader(cls)
