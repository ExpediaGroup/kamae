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

import numpy as np
import pytest
import tensorflow as tf

from kamae.spark.transformers import StringSequenceToEmbeddingTransformer


class TestStringSequenceToEmbedding:
    @pytest.fixture(scope="class")
    def example_dataframe(self, spark_session):
        return spark_session.createDataFrame(
            [
                ("1|2|3,4|5|6,0|0|0,0|0|0",),
                ("7|8|9,1|1|1,0|0|0,0|0|0",),
                ("1|2|3",),  # short input, requires padding
                ("1|2|3,4|5|6,7|8|9,1|1|1,9|9|9",),  # long input, requires truncation
            ],
            ["embedding_str"],
        )

    def test_string_sequence_to_embedding_transform_defaults(self):
        transformer = StringSequenceToEmbeddingTransformer()
        assert transformer.getSeqLen() == 10
        assert transformer.getEmbeddingDim() == 32
        assert transformer.getSeparator() == "|"
        assert transformer.getSequenceSeparator() == ","
        assert transformer.getPadValue() == "0"
        assert transformer.getReverse() is False
        assert transformer.getLayerName() == transformer.uid
        assert transformer.getOutputCol() == f"{transformer.uid}__output"

    def test_spark_transform_basic(self, example_dataframe):
        transformer = StringSequenceToEmbeddingTransformer(
            inputCol="embedding_str",
            outputCol="embedding",
            seqLen=4,
            embeddingDim=3,
        )
        actual = transformer.transform(example_dataframe)
        rows = actual.select("embedding").collect()
        expected = [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [7.0, 8.0, 9.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [1.0, 2.0, 3.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [1.0, 1.0, 1.0],
            ],
        ]
        np.testing.assert_allclose(
            np.array([r["embedding"] for r in rows]),
            np.array(expected),
            atol=1e-6,
        )

    def test_spark_transform_reverse(self, example_dataframe):
        transformer = StringSequenceToEmbeddingTransformer(
            inputCol="embedding_str",
            outputCol="embedding",
            seqLen=4,
            embeddingDim=3,
            reverse=True,
        )
        actual = transformer.transform(example_dataframe)
        rows = actual.select("embedding").collect()
        expected = [
            # Reverse only the non-pad prefix (first two vectors).
            [
                [4.0, 5.0, 6.0],
                [1.0, 2.0, 3.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [1.0, 1.0, 1.0],
                [7.0, 8.0, 9.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            # Single non-pad vector remains unchanged when reversed.
            [
                [1.0, 2.0, 3.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            # All four slots filled: full reverse.
            [
                [1.0, 1.0, 1.0],
                [7.0, 8.0, 9.0],
                [4.0, 5.0, 6.0],
                [1.0, 2.0, 3.0],
            ],
        ]
        np.testing.assert_allclose(
            np.array([r["embedding"] for r in rows]),
            np.array(expected),
            atol=1e-6,
        )

    @pytest.mark.parametrize(
        "input_strings, seq_len, embedding_dim, separator, sequence_separator, pad_value, reverse",
        [
            (
                [
                    "1|2|3,4|5|6,0|0|0,0|0|0",
                    "7|8|9,1|1|1,0|0|0,0|0|0",
                    "1|2|3",
                    "1|2|3,4|5|6,7|8|9,1|1|1,9|9|9",
                ],
                4,
                3,
                "|",
                ",",
                "0",
                False,
            ),
            (
                [
                    "1|2|3,4|5|6,0|0|0,0|0|0",
                    "7|8|9,1|1|1,2|2|2,0|0|0",
                ],
                4,
                3,
                "|",
                ",",
                "0",
                True,
            ),
            (
                [
                    "1:2:3;4:5:6",
                    "9:9:9;0:0:0",
                ],
                2,
                3,
                ":",
                ";",
                "0",
                False,
            ),
        ],
    )
    def test_spark_tf_parity(
        self,
        spark_session,
        input_strings,
        seq_len,
        embedding_dim,
        separator,
        sequence_separator,
        pad_value,
        reverse,
    ):
        transformer = StringSequenceToEmbeddingTransformer(
            inputCol="embedding_str",
            outputCol="embedding",
            seqLen=seq_len,
            embeddingDim=embedding_dim,
            separator=separator,
            sequenceSeparator=sequence_separator,
            padValue=pad_value,
            reverse=reverse,
        )

        spark_df = spark_session.createDataFrame(
            [(s,) for s in input_strings], ["embedding_str"]
        )
        spark_values = np.array(
            [
                row["embedding"]
                for row in transformer.transform(spark_df).select("embedding").collect()
            ]
        )

        tf_input = tf.constant([[s] for s in input_strings])
        tf_values = transformer.get_tf_layer()(tf_input).numpy()
        # Drop the artificial list axis (size 1) inserted for the TF input.
        tf_values = tf_values[:, 0, :, :]

        np.testing.assert_allclose(
            spark_values,
            tf_values,
            atol=1e-6,
            err_msg="Spark and TF outputs differ",
        )
