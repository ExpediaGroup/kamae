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

from pyspark.ml.param import TypeConverters

from .param_spec import _UNSET, ParamSpec

# ---------------------------------------------------------------------------
# Validators (value-only — no Spark self dependency)
# ---------------------------------------------------------------------------


def _validate_unit(value):
    allowed_units = ["milliseconds", "seconds", "ms", "s"]
    if value not in allowed_units:
        raise ValueError(f"Unit must be one of {allowed_units}")
    if value == "milliseconds":
        value = "ms"
    if value == "seconds":
        value = "s"
    return value


def _validate_num_oov_indices(value):
    if value is not None and value <= 0:
        raise ValueError("numOOVIndices must be a positive integer")
    return value


def _validate_max_num_labels(value):
    if value is not None and value <= 0:
        raise ValueError("maxNumLabels must be a positive integer")
    return value


def _validate_string_order_type(value):
    possible = ["frequencyAsc", "frequencyDesc", "alphabeticalAsc", "alphabeticalDesc"]
    if value is not None and value not in possible:
        raise ValueError(f"stringOrderType must be one of {', '.join(possible)}")
    return value


def _validate_nan_fill_value(value):
    if value is None:
        raise ValueError("nanFillValue cannot be None")
    return value


def _validate_sample_fraction(value):
    if value is not None:
        val = float(value)
        if not (0.0 < val < 1.0):
            raise ValueError(
                f"sampleFraction must be in the range (0.0, 1.0). Got {val}"
            )
    return value


def _validate_mean_stddev(value):
    if None in set(value):
        ids = [i for i, x in enumerate(value) if x is None]
        raise ValueError("Got null values at positions: ", ids)
    return value


def _validate_sort_order(value):
    if value not in ["asc", "desc"]:
        raise ValueError("sortOrder must be either 'asc' or 'desc'")
    return value


def _validate_lat_lon_constant(value):
    if len(value) != 2:
        raise ValueError("latLonConstant must be a list of two floats: [lat, lon]")
    if value[0] < -90.0 or value[0] > 90.0:
        raise ValueError("Latitude must be between -90 and 90")
    if value[1] < -180.0 or value[1] > 180.0:
        raise ValueError("Longitude must be between -180 and 180")
    return value


# ---------------------------------------------------------------------------
# Shared parameter groups (canonical keys are camelCase)
# ---------------------------------------------------------------------------

UNIX_TIMESTAMP_PARAMS = {
    "unit": ParamSpec(
        spark_typeconverter=TypeConverters.toString,
        default="s",
        doc="Unit of the timestamp. Can be 'milliseconds'/'ms' or 'seconds'/'s'. Default is 's'.",
        validator=_validate_unit,
    ),
}

DEFAULT_INT_VALUE_PARAMS = {
    "defaultValue": ParamSpec(
        spark_typeconverter=TypeConverters.toInt,
        default=_UNSET,
        doc="Default int value to use in the transformer.",
    ),
}

MASK_VALUE_PARAMS = {
    "maskValue": ParamSpec(
        spark_typeconverter=TypeConverters.toFloat,
        default=None,
        doc="Value to be used as a mask.",
    ),
}

DROP_UNSEEN_PARAMS = {
    "dropUnseen": ParamSpec(
        spark_typeconverter=TypeConverters.toBoolean,
        default=False,
        doc="Whether to drop unseen label index in the one hot encoder layer.",
    ),
}

STANDARD_SCALE_PARAMS = {
    "mean": ParamSpec(
        spark_typeconverter=TypeConverters.toListFloat,
        default=None,
        doc="Mean of the feature values.",
        validator=_validate_mean_stddev,
    ),
    "stddev": ParamSpec(
        spark_typeconverter=TypeConverters.toListFloat,
        default=None,
        doc="Standard deviation of the feature values.",
        validator=_validate_mean_stddev,
    ),
}


LAT_LON_CONSTANT_PARAMS = {
    "latLonConstant": ParamSpec(
        spark_typeconverter=TypeConverters.toListFloat,
        default=None,
        doc="Constant lat & lon as [lat, lon]. When set, only two input columns are needed.",
        validator=_validate_lat_lon_constant,
    ),
}

SAMPLE_FRACTION_PARAMS = {
    "sampleFraction": ParamSpec(
        spark_typeconverter=TypeConverters.toFloat,
        default=None,
        doc="Fraction of data to sample for statistics estimation (exclusive 0.0-1.0). Default None (no sampling).",
        validator=_validate_sample_fraction,
    ),
}

STRING_INDEX_PARAMS = {
    "labelsArray": ParamSpec(
        spark_typeconverter=TypeConverters.toListString,
        default=_UNSET,
        doc="Ordered list of labels to use for the indexer.",
    ),
    "stringOrderType": ParamSpec(
        spark_typeconverter=TypeConverters.toString,
        default="frequencyDesc",
        doc="How to order the strings. Options: 'frequencyAsc', 'frequencyDesc', 'alphabeticalAsc', 'alphabeticalDesc'.",
        validator=_validate_string_order_type,
    ),
    "maskToken": ParamSpec(
        spark_typeconverter=TypeConverters.toString,
        default=None,
        doc="Mask token to use for string indexing.",
    ),
    "numOOVIndices": ParamSpec(
        spark_typeconverter=TypeConverters.toInt,
        default=1,
        doc="Number of out of vocabulary indices to use.",
        validator=_validate_num_oov_indices,
    ),
    "maxNumLabels": ParamSpec(
        spark_typeconverter=TypeConverters.toInt,
        default=_UNSET,
        doc="Max number of labels to use.",
        validator=_validate_max_num_labels,
    ),
}

MASK_STRING_VALUE_PARAMS = {
    "maskValue": ParamSpec(
        spark_typeconverter=TypeConverters.toString,
        default=None,
        doc="String value to be used as a mask by the transformer.",
    ),
}

LISTWISE_PARAMS = {
    "queryIdCol": ParamSpec(
        spark_typeconverter=TypeConverters.toString,
        default=None,
        doc="Column name to aggregate summary statistics upon, such as 'search_id'.",
    ),
    "topN": ParamSpec(
        spark_typeconverter=TypeConverters.toInt,
        default=None,
        doc="Limit to how far into the list to aggregate.",
    ),
    "sortOrder": ParamSpec(
        spark_typeconverter=TypeConverters.toString,
        default="asc",
        doc="Option of 'asc' or 'desc' which defines order for listwise operation.",
        validator=_validate_sort_order,
    ),
}

LISTWISE_FILTER_PARAMS = {
    "minFilterValue": ParamSpec(
        spark_typeconverter=TypeConverters.toInt,
        default=None,
        doc="Value which equal to or greater than will be aggregated upon, anything less will be removed - this is primarily to deal with padded features.",
    ),
    "nanFillValue": ParamSpec(
        spark_typeconverter=TypeConverters.toFloat,
        default=0.0,
        doc="The value to fill Nan with.",
        validator=_validate_nan_fill_value,
    ),
    "axis": ParamSpec(
        spark_typeconverter=TypeConverters.toInt,
        default=1,
        doc="The axis to calculate the statistics across. Defaults to 1.",
    ),
}

LISTWISE_SEGMENT_PARAMS = {
    "withSegment": ParamSpec(
        spark_typeconverter=TypeConverters.toBoolean,
        default=False,
        doc="Whether to use the second input column to partition the statistic calculation. Defaults to False.",
    ),
}
