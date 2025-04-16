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

from kamae.spark.transformers import UnixTimestampToDateTimeTransformer


class TestUnixTimestampToDateTime:
    @pytest.fixture(scope="class")
    def unix_timestamp_to_date_time_transform_base(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1724410227,
                    [[3986011876, 3986011876], [3986011876, 3986011876]],
                    1724410227000,
                    [[3986011876000, 3986011876000], [3986011876000, 3986011876000]],
                ),
                (
                    1587639027,
                    [[1587639027, 4107667827], [1587639027, 4107667827]],
                    1587639027000,
                    [[1587639027000, 4107667827000], [1587639027000, 4107667827000]],
                ),
                (
                    3986011876,
                    [[3986011876, 1587639027], [3986011876, 1587639027]],
                    3986011876000,
                    [[3986011876000, 1587639027000], [3986011876000, 1587639027000]],
                ),
            ],
            [
                "timestamp_seconds",
                "timestamp_array_seconds",
                "timestamp_milliseconds",
                "timestamp_array_milliseconds",
            ],
        )

    @pytest.fixture(scope="class")
    def unix_timestamp_to_date_time_transform_timestamp_seconds(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1724410227,
                    [[3986011876, 3986011876], [3986011876, 3986011876]],
                    1724410227000,
                    [[3986011876000, 3986011876000], [3986011876000, 3986011876000]],
                    "2024-08-23",
                ),
                (
                    1587639027,
                    [[1587639027, 4107667827], [1587639027, 4107667827]],
                    1587639027000,
                    [[1587639027000, 4107667827000], [1587639027000, 4107667827000]],
                    "2020-04-23",
                ),
                (
                    3986011876,
                    [[3986011876, 1587639027], [3986011876, 1587639027]],
                    3986011876000,
                    [[3986011876000, 1587639027000], [3986011876000, 1587639027000]],
                    "2096-04-23",
                ),
            ],
            [
                "timestamp_seconds",
                "timestamp_array_seconds",
                "timestamp_milliseconds",
                "timestamp_array_milliseconds",
                "date_timestamp_seconds",
            ],
        )

    @pytest.fixture(scope="class")
    def unix_timestamp_to_date_time_transform_timestamp_milliseconds(
        self, spark_session
    ):
        return spark_session.createDataFrame(
            [
                (
                    1724410227,
                    [[3986011876, 3986011876], [3986011876, 3986011876]],
                    1724410227000,
                    [[3986011876000, 3986011876000], [3986011876000, 3986011876000]],
                    "2024-08-23 10:50:27.000",
                ),
                (
                    1587639027,
                    [[1587639027, 4107667827], [1587639027, 4107667827]],
                    1587639027000,
                    [[1587639027000, 4107667827000], [1587639027000, 4107667827000]],
                    "2020-04-23 10:50:27.000",
                ),
                (
                    3986011876,
                    [[3986011876, 1587639027], [3986011876, 1587639027]],
                    3986011876000,
                    [[3986011876000, 1587639027000], [3986011876000, 1587639027000]],
                    "2096-04-23 09:31:16.000",
                ),
            ],
            [
                "timestamp_seconds",
                "timestamp_array_seconds",
                "timestamp_milliseconds",
                "timestamp_array_milliseconds",
                "date_time_timestamp_milliseconds",
            ],
        )

    @pytest.fixture(scope="class")
    def unix_timestamp_to_date_time_transform_timestamp_seconds_array(
        self, spark_session
    ):
        return spark_session.createDataFrame(
            [
                (
                    1724410227,
                    [[3986011876, 3986011876], [3986011876, 3986011876]],
                    1724410227000,
                    [[3986011876000, 3986011876000], [3986011876000, 3986011876000]],
                    [
                        ["2096-04-23 09:31:16.000", "2096-04-23 09:31:16.000"],
                        ["2096-04-23 09:31:16.000", "2096-04-23 09:31:16.000"],
                    ],
                ),
                (
                    1587639027,
                    [[1587639027, 4107667827], [1587639027, 4107667827]],
                    1587639027000,
                    [[1587639027000, 4107667827000], [1587639027000, 4107667827000]],
                    [
                        ["2020-04-23 10:50:27.000", "2100-03-02 10:50:27.000"],
                        ["2020-04-23 10:50:27.000", "2100-03-02 10:50:27.000"],
                    ],
                ),
                (
                    3986011876,
                    [[3986011876, 1587639027], [3986011876, 1587639027]],
                    3986011876000,
                    [[3986011876000, 1587639027000], [3986011876000, 1587639027000]],
                    [
                        ["2096-04-23 09:31:16.000", "2020-04-23 10:50:27.000"],
                        ["2096-04-23 09:31:16.000", "2020-04-23 10:50:27.000"],
                    ],
                ),
            ],
            [
                "timestamp_seconds",
                "timestamp_array_seconds",
                "timestamp_milliseconds",
                "timestamp_array_milliseconds",
                "date_time_timestamp_seconds_array",
            ],
        )

    @pytest.fixture(scope="class")
    def unix_timestamp_to_date_time_transform_timestamp_milliseconds_array(
        self, spark_session
    ):
        return spark_session.createDataFrame(
            [
                (
                    1724410227,
                    [[3986011876, 3986011876], [3986011876, 3986011876]],
                    1724410227000,
                    [[3986011876000, 3986011876000], [3986011876000, 3986011876000]],
                    [["2096-04-23", "2096-04-23"], ["2096-04-23", "2096-04-23"]],
                ),
                (
                    1587639027,
                    [[1587639027, 4107667827], [1587639027, 4107667827]],
                    1587639027000,
                    [[1587639027000, 4107667827000], [1587639027000, 4107667827000]],
                    [["2020-04-23", "2100-03-02"], ["2020-04-23", "2100-03-02"]],
                ),
                (
                    3986011876,
                    [[3986011876, 1587639027], [3986011876, 1587639027]],
                    3986011876000,
                    [[3986011876000, 1587639027000], [3986011876000, 1587639027000]],
                    [["2096-04-23", "2020-04-23"], ["2096-04-23", "2020-04-23"]],
                ),
            ],
            [
                "timestamp_seconds",
                "timestamp_array_seconds",
                "timestamp_milliseconds",
                "timestamp_array_milliseconds",
                "date_timestamp_milliseconds_array",
            ],
        )

    @pytest.mark.parametrize(
        "input_df, input_col, output_col, unit, include_time, expected_df",
        [
            (
                "unix_timestamp_to_date_time_transform_base",
                "timestamp_seconds",
                "date_timestamp_seconds",
                "seconds",
                False,
                "unix_timestamp_to_date_time_transform_timestamp_seconds",
            ),
            (
                "unix_timestamp_to_date_time_transform_base",
                "timestamp_milliseconds",
                "date_time_timestamp_milliseconds",
                "milliseconds",
                True,
                "unix_timestamp_to_date_time_transform_timestamp_milliseconds",
            ),
            (
                "unix_timestamp_to_date_time_transform_base",
                "timestamp_array_seconds",
                "date_time_timestamp_seconds_array",
                "s",
                True,
                "unix_timestamp_to_date_time_transform_timestamp_seconds_array",
            ),
            (
                "unix_timestamp_to_date_time_transform_base",
                "timestamp_array_milliseconds",
                "date_timestamp_milliseconds_array",
                "ms",
                False,
                "unix_timestamp_to_date_time_transform_timestamp_milliseconds_array",
            ),
        ],
    )
    def test_unix_timestamp_to_date_time_transform(
        self,
        input_df,
        input_col,
        output_col,
        unit,
        include_time,
        expected_df,
        request,
    ):
        expected = request.getfixturevalue(expected_df)
        input_df = request.getfixturevalue(input_df)

        unix_timestamp_to_date_time_transform = UnixTimestampToDateTimeTransformer(
            inputCol=input_col,
            outputCol=output_col,
            unit=unit,
            includeTime=include_time,
        )
        actual = unix_timestamp_to_date_time_transform.transform(input_df)
        diff = expected.exceptAll(actual)
        assert diff.isEmpty()

    def test_unix_timestamp_to_date_time_defaults(self):
        # when
        unix_timestamp_to_date_time_transform = UnixTimestampToDateTimeTransformer()
        # then
        assert (
            unix_timestamp_to_date_time_transform.getLayerName()
            == unix_timestamp_to_date_time_transform.uid
        )
        assert unix_timestamp_to_date_time_transform.getUnit() == "s"
        assert unix_timestamp_to_date_time_transform.getIncludeTime()
        assert (
            unix_timestamp_to_date_time_transform.getOutputCol()
            == f"{unix_timestamp_to_date_time_transform.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensor, input_dtype, output_dtype, unit, include_time",
        [
            (
                tf.constant([1587639027.987, 4107667827.675]),
                "double",
                None,
                "s",
                True,
            ),
            (
                tf.constant([1587639027653, 4107667827789]),
                "bigint",
                None,
                "ms",
                False,
            ),
        ],
    )
    def test_unix_timestamp_to_date_time_transform_spark_tf_parity(
        self,
        spark_session,
        input_tensor,
        input_dtype,
        output_dtype,
        unit,
        include_time,
    ):
        # given
        transformer = UnixTimestampToDateTimeTransformer(
            inputCol="input",
            outputCol="output",
            inputDtype=input_dtype,
            outputDtype=output_dtype,
            unit=unit,
            includeTime=include_time,
        )

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
            v.decode("utf-8")
            for v in transformer.get_tf_layer()(input_tensor).numpy().tolist()
        ]
        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
