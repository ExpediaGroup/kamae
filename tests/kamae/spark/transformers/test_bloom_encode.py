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

from kamae.spark.transformers import BloomEncodeTransformer


class TestBloomEncode:
    @pytest.fixture(scope="class")
    def bloom_encoder_example_input_with_string_arrays(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, [["a", "c", "c"], ["a", "c", "c"], ["a", "a", "a"]]),
                (4, 2, 6, [["a", "d", "c"], ["a", "t", "s"], ["x", "o", "p"]]),
                (7, 8, 3, [["l", "c", "c"], ["a", "h", "c"], ["a", "w", "a"]]),
            ],
            ["col1", "col2", "col3", "col4"],
        )

    @pytest.fixture(scope="class")
    def bloom_encoder_col4_array_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    [["a", "c", "c"], ["a", "c", "c"], ["a", "a", "a"]],
                    [
                        [[34, 95, 8], [34, 16, 64], [34, 16, 64]],
                        [[34, 95, 8], [34, 16, 64], [34, 16, 64]],
                        [[34, 95, 8], [34, 95, 8], [34, 95, 8]],
                    ],
                ),
                (
                    4,
                    2,
                    6,
                    [["a", "d", "c"], ["a", "t", "s"], ["x", "o", "p"]],
                    [
                        [[34, 95, 8], [28, 54, 80], [34, 16, 64]],
                        [[34, 95, 8], [85, 67, 27], [61, 22, 41]],
                        [[0, 59, 16], [92, 86, 90], [94, 92, 70]],
                    ],
                ),
                (
                    7,
                    8,
                    3,
                    [["l", "c", "c"], ["a", "h", "c"], ["a", "w", "a"]],
                    [
                        [[54, 5, 34], [34, 16, 64], [34, 16, 64]],
                        [[34, 95, 8], [31, 53, 85], [34, 16, 64]],
                        [[34, 95, 8], [58, 67, 64], [34, 95, 8]],
                    ],
                ),
            ],
            ["col1", "col2", "col3", "col4", "bloom_encode_col4"],
        )

    @pytest.fixture(scope="class")
    def bloom_encoder_col4_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], [34, 95, 8]),
                (4, 2, 6, "b", "c", [4, 2, 6], [92, 62, 96]),
                (7, 8, 3, "a", "a", [7, 8, 3], [34, 95, 8]),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "bloom_encode_col4",
            ],
        )

    @pytest.fixture(scope="class")
    def bloom_encoder_col5_expected(self, spark_session):
        return spark_session.createDataFrame(
            [
                (1, 2, 3, "a", "c", [1, 2, 3], [173, 185, 180, 152]),
                (4, 2, 6, "b", "c", [4, 2, 6], [173, 185, 180, 152]),
                (7, 8, 3, "a", "a", [7, 8, 3], [10, 16, 26, 75]),
            ],
            [
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col1_col2_col3",
                "bloom_encode_col5",
            ],
        )

    @pytest.mark.parametrize(
        "input_dataframe, input_col, output_col, num_bins, mask_value, num_hash_fns, feature_cardinality, use_heuristic_num_bins, expected_dataframe",
        [
            (
                "bloom_encoder_example_input_with_string_arrays",
                "col4",
                "bloom_encode_col4",
                100,
                None,
                3,
                None,
                False,
                "bloom_encoder_col4_array_expected",
            ),
            (
                "example_dataframe",
                "col4",
                "bloom_encode_col4",
                100,
                None,
                3,
                None,
                False,
                "bloom_encoder_col4_expected",
            ),
            (
                "example_dataframe",
                "col5",
                "bloom_encode_col5",
                None,
                "d",
                4,
                1000,
                True,
                "bloom_encoder_col5_expected",
            ),
        ],
    )
    def test_spark_bloom_encoder(
        self,
        input_dataframe,
        input_col,
        output_col,
        num_bins,
        mask_value,
        num_hash_fns,
        feature_cardinality,
        use_heuristic_num_bins,
        expected_dataframe,
        request,
    ):
        # given
        example_dataframe = request.getfixturevalue(input_dataframe)
        expected = request.getfixturevalue(expected_dataframe)
        # when
        transformer = BloomEncodeTransformer(
            inputCol=input_col,
            outputCol=output_col,
            numBins=num_bins,
            numHashFns=num_hash_fns,
            featureCardinality=feature_cardinality,
            maskValue=mask_value,
            useHeuristicNumBins=use_heuristic_num_bins,
        )
        actual = transformer.transform(example_dataframe)
        # then
        actual.select(output_col).show(20, False)
        expected.select(output_col).show(20, False)
        diff = actual.exceptAll(expected)
        assert diff.isEmpty(), "Expected and actual dataframes are not equal"

    def test_bloom_encoder_defaults(self):
        # when
        bloom_encoder = BloomEncodeTransformer()
        # then
        assert bloom_encoder.getLayerName() == bloom_encoder.uid
        assert bloom_encoder.getOutputCol() == f"{bloom_encoder.uid}__output"
        assert bloom_encoder.getMaskValue() is None
        assert bloom_encoder.getNumBins() is None
        assert bloom_encoder.getFeatureCardinality() is None
        assert bloom_encoder.getNumHashFns() == 3
        assert not bloom_encoder.getUseHeuristicNumBins()

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, num_bins, mask_value, num_hash_fns, feature_cardinality, use_heuristic_num_bins",
        [
            (
                tf.constant(["a", "b", "c", "d", "e", "f"]),
                None,
                "double",
                100,
                "f",
                3,
                None,
                False,
            ),
            (tf.constant([1, 2, 3]), "string", None, 200, "c", 4, 1000, True),
            (tf.constant(["e", "f", "g"]), None, "int", 3000, None, 5, 10000, False),
            (tf.constant(["h", "i", "j"]), None, None, 43, "h", 2, 50, True),
            (
                tf.constant([True, False, False]),
                "string",
                "smallint",
                99,
                None,
                5,
                None,
                False,
            ),
        ],
    )
    def test_bloom_encoder_spark_tf_parity(
        self,
        spark_session,
        input_tensor,
        input_dtype,
        output_dtype,
        num_bins,
        mask_value,
        num_hash_fns,
        feature_cardinality,
        use_heuristic_num_bins,
    ):
        # given
        transformer = BloomEncodeTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            numBins=num_bins,
            numHashFns=num_hash_fns,
            featureCardinality=feature_cardinality,
            maskValue=mask_value,
            useHeuristicNumBins=use_heuristic_num_bins,
        )

        if input_tensor.dtype.name == "string":
            spark_df = spark_session.createDataFrame(
                [(v.decode("utf-8"),) for v in input_tensor.numpy().tolist()], ["input"]
            )
        else:
            spark_df = spark_session.createDataFrame(
                [(v,) for v in input_tensor.numpy().tolist()], ["input"]
            )
        spark_values = (
            transformer.transform(spark_df)
            .select("output")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        tensorflow_values = [
            v.decode("utf-8") if isinstance(v, bytes) else v
            for v in transformer.get_tf_layer()(input_tensor).numpy().tolist()
        ]

        # then
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
