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

from kamae.spark.transformers import DateParseTransformer


class TestDateParse:
    @pytest.fixture(scope="class")
    def date_parse_transform_base(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "2022-01-02",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                    ],
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                    ],
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                    ],
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_base_w_missing(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "",
                    [
                        ["2022-01-02", "", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", ""],
                    ],
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["", "2026-01-31", "2024-04-11"],
                    ],
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29",
                    [
                        ["2022-01-02", "", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", ""],
                    ],
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_base_timestamp(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "2022-01-02 17:28:32.321",
                    [
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                    ],
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12 18:19:20.444",
                    [
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                    ],
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29 11:23:20.212",
                    [
                        [["2035-03-16 13:01:45.345"], ["2023-11-02 00:05:00.00"]],
                        [["2025-03-06 23:01:45.345"], ["2090-01-02 00:05:00.00"]],
                    ],
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_base_timestamp_w_missing(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "",
                    [
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                    ],
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12 18:19:20.444",
                    [
                        [[""], ["2023-01-02 00:05:00.00"]],
                        [["2045-03-06 23:01:45.345"], [""]],
                    ],
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29 11:23:20.212",
                    [
                        [[""], ["2023-11-02 00:05:00.00"]],
                        [["2025-03-06 23:01:45.345"], ["2090-01-02 00:05:00.00"]],
                    ],
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_expected_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "2022-01-02",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                    ],
                    1,
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                    ],
                    8,
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                    ],
                    2,
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6", "col5_month"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_expected_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "2022-01-02",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                    ],
                    7,
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                    ],
                    6,
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                    ],
                    6,
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6", "col5_day_of_week"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_expected_3(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "2022-01-02",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                    ],
                    2022,
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                    ],
                    2023,
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                    ],
                    2020,
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6", "col5_year"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_expected_4(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "2022-01-02",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                    ],
                    2,
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                    ],
                    224,
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                    ],
                    60,
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6", "col5_day_of_year"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_expected_5(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "2022-01-02",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                    ],
                    2,
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                    ],
                    12,
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                    ],
                    29,
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6", "col5_day_of_month"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_expected_6(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "2022-01-02",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                    ],
                    0,
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                    ],
                    0,
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                    ],
                    0,
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6", "col5_minute"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_expected_7(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "2022-01-02 17:28:32.321",
                    [
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                    ],
                    321,
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12 18:19:20.444",
                    [
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                    ],
                    444,
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29 11:23:20.212",
                    [
                        [["2035-03-16 13:01:45.345"], ["2023-11-02 00:05:00.00"]],
                        [["2025-03-06 23:01:45.345"], ["2090-01-02 00:05:00.00"]],
                    ],
                    212,
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6", "col5_millisecond"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_expected_8(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "2022-01-02 17:28:32.321",
                    [
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                    ],
                    [[[345], [0]], [[345], [0]]],
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12 18:19:20.444",
                    [
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                    ],
                    [[[345], [0]], [[345], [0]]],
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29 11:23:20.212",
                    [
                        [["2035-03-16 13:01:45.345"], ["2023-11-02 00:05:00.00"]],
                        [["2025-03-06 23:01:45.345"], ["2090-01-02 00:05:00.00"]],
                    ],
                    [[[345], [0]], [[345], [0]]],
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6", "col6_millisecond"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_expected_9(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "2022-01-02 17:28:32.321",
                    [
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                    ],
                    [[[6], [2]], [[6], [2]]],
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12 18:19:20.444",
                    [
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                    ],
                    [[[6], [2]], [[6], [2]]],
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29 11:23:20.212",
                    [
                        [["2035-03-16 13:01:45.345"], ["2023-11-02 00:05:00.00"]],
                        [["2025-03-06 23:01:45.345"], ["2090-01-02 00:05:00.00"]],
                    ],
                    [[[16], [2]], [[6], [2]]],
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6", "col6_day_of_month"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_expected_10(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "2022-01-02 17:28:32.321",
                    [
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                    ],
                    [[[2045], [2023]], [[2045], [2023]]],
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12 18:19:20.444",
                    [
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                    ],
                    [[[2045], [2023]], [[2045], [2023]]],
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29 11:23:20.212",
                    [
                        [["2035-03-16 13:01:45.345"], ["2023-11-02 00:05:00.00"]],
                        [["2025-03-06 23:01:45.345"], ["2090-01-02 00:05:00.00"]],
                    ],
                    [[[2035], [2023]], [[2025], [2090]]],
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6", "col6_year"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_expected_w_missing_1(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "",
                    [
                        ["2022-01-02", "", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", ""],
                    ],
                    -1,
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["", "2026-01-31", "2024-04-11"],
                    ],
                    8,
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29",
                    [
                        ["2022-01-02", "", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", ""],
                    ],
                    2,
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6", "col5_month"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_expected_w_missing_2(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "",
                    [
                        ["2022-01-02", "", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", ""],
                    ],
                    -1,
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["", "2026-01-31", "2024-04-11"],
                    ],
                    6,
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29",
                    [
                        ["2022-01-02", "", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", ""],
                    ],
                    6,
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6", "col5_day_of_week"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_expected_w_missing_3(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "",
                    [
                        ["2022-01-02", "", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", ""],
                    ],
                    -1,
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["", "2026-01-31", "2024-04-11"],
                    ],
                    2023,
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29",
                    [
                        ["2022-01-02", "", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", ""],
                    ],
                    2020,
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6", "col5_year"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_expected_w_missing_4(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "",
                    [
                        ["2022-01-02", "", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", ""],
                    ],
                    -1,
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["", "2026-01-31", "2024-04-11"],
                    ],
                    224,
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29",
                    [
                        ["2022-01-02", "", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", ""],
                    ],
                    60,
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6", "col5_day_of_year"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_expected_w_missing_5(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "",
                    [
                        ["2022-01-02", "", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", ""],
                    ],
                    -1,
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["", "2026-01-31", "2024-04-11"],
                    ],
                    12,
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29",
                    [
                        ["2022-01-02", "", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", ""],
                    ],
                    29,
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6", "col5_day_of_month"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_expected_w_missing_6(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "",
                    [
                        ["2022-01-02", "", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", ""],
                    ],
                    -1,
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12",
                    [
                        ["2022-01-02", "2026-01-31", "2024-04-11"],
                        ["", "2026-01-31", "2024-04-11"],
                    ],
                    0,
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29",
                    [
                        ["2022-01-02", "", "2024-04-11"],
                        ["2022-01-02", "2026-01-31", ""],
                    ],
                    0,
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6", "col5_minute"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_expected_w_missing_7(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "",
                    [
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                    ],
                    -1,
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12 18:19:20.444",
                    [
                        [[""], ["2023-01-02 00:05:00.00"]],
                        [["2045-03-06 23:01:45.345"], [""]],
                    ],
                    444,
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29 11:23:20.212",
                    [
                        [[""], ["2023-11-02 00:05:00.00"]],
                        [["2025-03-06 23:01:45.345"], ["2090-01-02 00:05:00.00"]],
                    ],
                    212,
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6", "col5_millisecond"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_expected_w_missing_8(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "",
                    [
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                    ],
                    [
                        [[345], [0]],
                        [[345], [0]],
                    ],
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12 18:19:20.444",
                    [
                        [[""], ["2023-01-02 00:05:00.00"]],
                        [["2045-03-06 23:01:45.345"], [""]],
                    ],
                    [
                        [[-1], [0]],
                        [[345], [-1]],
                    ],
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29 11:23:20.212",
                    [
                        [[""], ["2023-11-02 00:05:00.00"]],
                        [["2025-03-06 23:01:45.345"], ["2090-01-02 00:05:00.00"]],
                    ],
                    [
                        [[-1], [0]],
                        [[345], [0]],
                    ],
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6", "col6_millisecond"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_expected_w_missing_9(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "",
                    [
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                    ],
                    [
                        [[6], [2]],
                        [[6], [2]],
                    ],
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12 18:19:20.444",
                    [
                        [[""], ["2023-01-02 00:05:00.00"]],
                        [["2045-03-06 23:01:45.345"], [""]],
                    ],
                    [
                        [[-1], [2]],
                        [[6], [-1]],
                    ],
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29 11:23:20.212",
                    [
                        [[""], ["2023-11-02 00:05:00.00"]],
                        [["2025-03-06 23:01:45.345"], ["2090-01-02 00:05:00.00"]],
                    ],
                    [
                        [[-1], [2]],
                        [[6], [2]],
                    ],
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6", "col6_day_of_month"],
        )

    @pytest.fixture(scope="class")
    def date_parse_transform_expected_w_missing_10(self, spark_session):
        return spark_session.createDataFrame(
            [
                (
                    1,
                    2,
                    3,
                    "a",
                    "",
                    [
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                        [["2045-03-06 23:01:45.345"], ["2023-01-02 00:05:00.00"]],
                    ],
                    [
                        [[2045], [2023]],
                        [[2045], [2023]],
                    ],
                ),
                (
                    4,
                    2,
                    6,
                    "b",
                    "2023-08-12 18:19:20.444",
                    [
                        [[""], ["2023-01-02 00:05:00.00"]],
                        [["2045-03-06 23:01:45.345"], [""]],
                    ],
                    [
                        [[-1], [2023]],
                        [[2045], [-1]],
                    ],
                ),
                (
                    7,
                    8,
                    3,
                    "a",
                    "2020-02-29 11:23:20.212",
                    [
                        [[""], ["2023-11-02 00:05:00.00"]],
                        [["2025-03-06 23:01:45.345"], ["2090-01-02 00:05:00.00"]],
                    ],
                    [
                        [[-1], [2023]],
                        [[2025], [2090]],
                    ],
                ),
            ],
            ["col1", "col2", "col3", "col4", "col5", "col6", "col6_year"],
        )

    @pytest.mark.parametrize(
        "input_df, input_col, output_col, date_part, default_value, expected_df",
        [
            (
                "date_parse_transform_base",
                "col5",
                "col5_month",
                "MonthOfYear",
                None,
                "date_parse_transform_expected_1",
            ),
            (
                "date_parse_transform_base",
                "col5",
                "col5_day_of_week",
                "DayOfWeek",
                None,
                "date_parse_transform_expected_2",
            ),
            (
                "date_parse_transform_base",
                "col5",
                "col5_year",
                "Year",
                None,
                "date_parse_transform_expected_3",
            ),
            (
                "date_parse_transform_base",
                "col5",
                "col5_day_of_year",
                "DayOfYear",
                None,
                "date_parse_transform_expected_4",
            ),
            (
                "date_parse_transform_base",
                "col5",
                "col5_day_of_month",
                "DayOfMonth",
                None,
                "date_parse_transform_expected_5",
            ),
            (
                "date_parse_transform_base",
                "col5",
                "col5_minute",
                "Minute",
                None,
                "date_parse_transform_expected_6",
            ),
            (
                "date_parse_transform_base_timestamp",
                "col5",
                "col5_millisecond",
                "Millisecond",
                None,
                "date_parse_transform_expected_7",
            ),
            (
                "date_parse_transform_base_timestamp",
                "col6",
                "col6_millisecond",
                "Millisecond",
                None,
                "date_parse_transform_expected_8",
            ),
            (
                "date_parse_transform_base_timestamp",
                "col6",
                "col6_day_of_month",
                "DayOfMonth",
                None,
                "date_parse_transform_expected_9",
            ),
            (
                "date_parse_transform_base_timestamp",
                "col6",
                "col6_year",
                "Year",
                None,
                "date_parse_transform_expected_10",
            ),
            (
                "date_parse_transform_base_w_missing",
                "col5",
                "col5_month",
                "MonthOfYear",
                -1,
                "date_parse_transform_expected_w_missing_1",
            ),
            (
                "date_parse_transform_base_w_missing",
                "col5",
                "col5_day_of_week",
                "DayOfWeek",
                -1,
                "date_parse_transform_expected_w_missing_2",
            ),
            (
                "date_parse_transform_base_w_missing",
                "col5",
                "col5_year",
                "Year",
                -1,
                "date_parse_transform_expected_w_missing_3",
            ),
            (
                "date_parse_transform_base_w_missing",
                "col5",
                "col5_day_of_year",
                "DayOfYear",
                -1,
                "date_parse_transform_expected_w_missing_4",
            ),
            (
                "date_parse_transform_base_w_missing",
                "col5",
                "col5_day_of_month",
                "DayOfMonth",
                -1,
                "date_parse_transform_expected_w_missing_5",
            ),
            (
                "date_parse_transform_base_w_missing",
                "col5",
                "col5_minute",
                "Minute",
                -1,
                "date_parse_transform_expected_w_missing_6",
            ),
            (
                "date_parse_transform_base_timestamp_w_missing",
                "col5",
                "col5_millisecond",
                "Millisecond",
                -1,
                "date_parse_transform_expected_w_missing_7",
            ),
            (
                "date_parse_transform_base_timestamp_w_missing",
                "col6",
                "col6_millisecond",
                "Millisecond",
                -1,
                "date_parse_transform_expected_w_missing_8",
            ),
            (
                "date_parse_transform_base_timestamp_w_missing",
                "col6",
                "col6_day_of_month",
                "DayOfMonth",
                -1,
                "date_parse_transform_expected_w_missing_9",
            ),
            (
                "date_parse_transform_base_timestamp_w_missing",
                "col6",
                "col6_year",
                "Year",
                -1,
                "date_parse_transform_expected_w_missing_10",
            ),
        ],
    )
    def test_date_parse_transform(
        self,
        input_df,
        input_col,
        output_col,
        date_part,
        default_value,
        expected_df,
        request,
    ):
        expected = request.getfixturevalue(expected_df)
        input_df = request.getfixturevalue(input_df)
        date_parse_transform = DateParseTransformer(
            inputCol=input_col,
            outputCol=output_col,
            datePart=date_part,
            defaultValue=default_value,
        )
        actual = date_parse_transform.transform(input_df)
        diff = expected.exceptAll(actual)
        assert diff.isEmpty()

    def test_date_parse_defaults(self):
        # when
        date_parse_transform = DateParseTransformer(datePart="MonthOfYear")
        # then
        assert date_parse_transform.getLayerName() == date_parse_transform.uid
        assert (
            date_parse_transform.getOutputCol() == f"{date_parse_transform.uid}__output"
        )

    @pytest.mark.parametrize(
        "input_tensor, date_part, default_value, input_dtype, output_dtype",
        [
            (
                tf.constant(["2022-01-02", "2023-08-12", "2020-02-29"]),
                "MonthOfYear",
                None,
                "string",
                "double",
            ),
            (
                tf.constant(["2022-01-02", "2023-08-12", "2020-02-29"]),
                "DayOfMonth",
                None,
                None,
                "bigint",
            ),
            (
                tf.constant(["2022-01-02", "2023-08-12", "2020-02-29"]),
                "DayOfYear",
                None,
                None,
                "smallint",
            ),
            (
                tf.constant(["2022-01-02", "2023-08-12", "2020-02-29"]),
                "Year",
                None,
                "string",
                "string",
            ),
            (
                tf.constant(["2022-01-02", "2023-08-12", "2020-02-29"]),
                "DayOfWeek",
                None,
                "string",
                "string",
            ),
            (
                tf.constant(["2022-01-02", "2023-08-12", "2020-02-29"]),
                "Hour",
                None,
                "string",
                "tinyint",
            ),
            (
                tf.constant(["2022-01-02", "2023-08-12", "2020-02-29"]),
                "Minute",
                None,
                None,
                "int",
            ),
            (
                tf.constant(["2022-01-02", "2023-08-12", "2020-02-29"]),
                "Second",
                None,
                None,
                None,
            ),
            (
                tf.constant(["2022-01-02", "2023-08-12", "2020-02-29"]),
                "Millisecond",
                None,
                None,
                "string",
            ),
            (
                tf.constant(["2022-01-02", "2023-08-12", "2020-02-29"]),
                "MonthOfYear",
                None,
                "string",
                "double",
            ),
            (
                tf.constant(
                    [
                        "2022-01-02 23:48:42.321",
                        "2023-08-12 18:19:20.444",
                        "2020-02-29 11:23:20.212",
                    ]
                ),
                "DayOfMonth",
                None,
                None,
                "bigint",
            ),
            (
                tf.constant(
                    [
                        "2022-01-02 23:48:42.321",
                        "2023-08-12 18:19:20.444",
                        "2020-02-29 11:23:20.212",
                    ]
                ),
                "DayOfYear",
                None,
                None,
                "smallint",
            ),
            (
                tf.constant(
                    [
                        "2022-01-02 23:48:42.321",
                        "2023-08-12 18:19:20.444",
                        "2020-02-29 11:23:20.212",
                    ]
                ),
                "Year",
                None,
                "string",
                "string",
            ),
            (
                tf.constant(
                    [
                        "2022-01-02 23:48:42.321",
                        "2023-08-12 18:19:20.444",
                        "2020-02-29 11:23:20.212",
                    ]
                ),
                "DayOfWeek",
                None,
                "string",
                "string",
            ),
            (
                tf.constant(
                    [
                        "2022-01-02 23:48:42.321",
                        "2023-08-12 18:19:20.444",
                        "2020-02-29 11:23:20.212",
                    ]
                ),
                "Hour",
                None,
                "string",
                "tinyint",
            ),
            (
                tf.constant(
                    [
                        "2022-01-02 23:48:42.321",
                        "2023-08-12 18:19:20.444",
                        "2020-02-29 11:23:20.212",
                    ]
                ),
                "Minute",
                None,
                None,
                "int",
            ),
            (
                tf.constant(
                    [
                        "2022-01-02 23:48:42.321",
                        "2023-08-12 18:19:20.444",
                        "2020-02-29 11:23:20.212",
                    ]
                ),
                "Second",
                None,
                None,
                None,
            ),
            (
                tf.constant(
                    [
                        "2022-01-02 23:48:42.321",
                        "2023-08-12 18:19:20.444",
                        "2020-02-29 11:23:20.212",
                    ]
                ),
                "Millisecond",
                None,
                None,
                "string",
            ),
            (
                tf.constant(["2022-01-02", "", "2020-02-29"]),
                "MonthOfYear",
                -1,
                "string",
                "double",
            ),
            (
                tf.constant(["2022-01-02", "2023-08-12", "2020-02-29"]),
                "DayOfMonth",
                -1,
                None,
                "bigint",
            ),
            (
                tf.constant(["2022-01-02", "2023-08-12", "2020-02-29"]),
                "DayOfYear",
                -1,
                None,
                "smallint",
            ),
            (
                tf.constant(["2022-01-02", "2023-08-12", "2020-02-29"]),
                "Year",
                -1,
                "string",
                "string",
            ),
            (
                tf.constant(["", "2023-08-12", "2020-02-29"]),
                "DayOfWeek",
                -1,
                "string",
                "string",
            ),
            (
                tf.constant(["2022-01-02", "2023-08-12", "2020-02-29"]),
                "Hour",
                -1,
                "string",
                "tinyint",
            ),
            (tf.constant(["2022-01-02", "", "2020-02-29"]), "Minute", -1, None, "int"),
            (tf.constant(["2022-01-02", "2023-08-12", ""]), "Second", -1, None, None),
            (
                tf.constant(["2022-01-02", "2023-08-12", "2020-02-29"]),
                "Millisecond",
                -1,
                None,
                "string",
            ),
            (
                tf.constant(["", "2023-08-12", "2020-02-29"]),
                "MonthOfYear",
                -1,
                "string",
                "double",
            ),
            (
                tf.constant(
                    [
                        "2022-01-02 23:48:42.321",
                        "",
                        "2020-02-29 11:23:20.212",
                    ]
                ),
                "DayOfMonth",
                -1,
                None,
                "bigint",
            ),
            (
                tf.constant(
                    [
                        "2022-01-02 23:48:42.321",
                        "",
                        "2020-02-29 11:23:20.212",
                    ]
                ),
                "DayOfYear",
                -1,
                None,
                "smallint",
            ),
            (
                tf.constant(
                    [
                        "",
                        "2023-08-12 18:19:20.444",
                        "2020-02-29 11:23:20.212",
                    ]
                ),
                "Year",
                -1,
                "string",
                "string",
            ),
            (
                tf.constant(
                    [
                        "2022-01-02 23:48:42.321",
                        "2023-08-12 18:19:20.444",
                        "2020-02-29 11:23:20.212",
                    ]
                ),
                "DayOfWeek",
                -1,
                "string",
                "string",
            ),
            (
                tf.constant(
                    [
                        "2022-01-02 23:48:42.321",
                        "2023-08-12 18:19:20.444",
                        "",
                    ]
                ),
                "Hour",
                -1,
                "string",
                "tinyint",
            ),
            (
                tf.constant(
                    [
                        "2022-01-02 23:48:42.321",
                        "",
                        "2020-02-29 11:23:20.212",
                    ]
                ),
                "Minute",
                -1,
                None,
                "int",
            ),
            (
                tf.constant(
                    [
                        "",
                        "2023-08-12 18:19:20.444",
                        "2020-02-29 11:23:20.212",
                    ]
                ),
                "Second",
                -1,
                None,
                None,
            ),
            (
                tf.constant(
                    [
                        "2022-01-02 23:48:42.321",
                        "2023-08-12 18:19:20.444",
                        "",
                    ]
                ),
                "Millisecond",
                -1,
                None,
                "string",
            ),
        ],
    )
    def test_date_parse_transform_spark_tf_parity(
        self,
        spark_session,
        input_tensor,
        date_part,
        default_value,
        input_dtype,
        output_dtype,
    ):
        # given
        transformer = DateParseTransformer(
            inputCol="input",
            outputCol="output",
            datePart=date_part,
            defaultValue=default_value,
            inputDtype=input_dtype,
            outputDtype=output_dtype,
        )

        spark_df = spark_session.createDataFrame(
            [(v.decode("utf-8"),) for v in input_tensor.numpy().tolist()], ["input"]
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

        np.testing.assert_equal(
            spark_values,
            tensorflow_values,
            err_msg="Spark and Tensorflow transform outputs are not equal",
        )
