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

"""Tests for JIT compatibility of Keras layers."""

import inspect

import keras
import pytest
import tensorflow as tf

import kamae.keras.core.layers as core_layers_mod
import kamae.keras.tensorflow.layers as tf_layers_mod

# Multi-backend layers
from kamae.keras.core.layers import (
    AbsoluteValueLayer,
    ArrayConcatenateLayer,
    ArrayCropLayer,
    ArrayReduceMaxLayer,
    ArraySplitLayer,
    ArraySubtractMinimumLayer,
    BearingAngleLayer,
    BinLayer,
    ConditionalStandardScaleLayer,
    CosineSimilarityLayer,
    DivideLayer,
    ExpLayer,
    ExponentLayer,
    HaversineDistanceLayer,
    IdentityLayer,
    ImputeLayer,
    LogicalAndLayer,
    LogicalNotLayer,
    LogicalOrLayer,
    LogLayer,
    MaxLayer,
    MeanLayer,
    MinLayer,
    MinMaxScaleLayer,
    ModuloLayer,
    MultiplyLayer,
    NumericalIfStatementLayer,
    PairwiseCosineSimilarityLayer,
    RoundLayer,
    RoundToDecimalLayer,
    StandardScaleLayer,
    SubtractLayer,
    SumLayer,
)

# TF-only layers
from kamae.keras.tensorflow.layers import (
    BloomEncodeLayer,
    BucketizeLayer,
    CurrentDateLayer,
    CurrentDateTimeLayer,
    CurrentUnixTimestampLayer,
    DateAddLayer,
    DateDiffLayer,
    DateParseLayer,
    DateTimeToUnixTimestampLayer,
    HashIndexLayer,
    IfStatementLayer,
    LambdaFunctionLayer,
    ListMaxLayer,
    ListMeanLayer,
    ListMedianLayer,
    ListMinLayer,
    ListRankLayer,
    ListStdDevLayer,
    MinHashIndexLayer,
    OneHotEncodeLayer,
    OneHotLayer,
    OrdinalArrayEncodeLayer,
    StringAffixLayer,
    StringArrayConstantLayer,
    StringCaseLayer,
    StringConcatenateLayer,
    StringContainsLayer,
    StringContainsListLayer,
    StringEqualsIfStatementLayer,
    StringIndexLayer,
    StringIsInListLayer,
    StringListToStringLayer,
    StringMapLayer,
    StringReplaceLayer,
    StringToStringListLayer,
    SubStringDelimAtIndexLayer,
    UnixTimestampToDateTimeLayer,
)

# JIT-compatible layers (jit_compatible = True)
JIT_COMPATIBLE_LAYERS = [
    # All 31 core layers
    (AbsoluteValueLayer, [tf.random.normal((32, 10))], None),
    (
        ArrayConcatenateLayer,
        [tf.random.normal((32, 10, 100, 3)), tf.random.normal((32, 10, 100, 3))],
        {"axis": -2},
    ),
    (ArrayReduceMaxLayer, [tf.random.normal((32, 10))], {"default_value": 0.0}),
    (ArraySplitLayer, [tf.random.normal((32, 10, 100, 3))], {"axis": -2}),
    (
        ArraySubtractMinimumLayer,
        [tf.random.normal((32, 10, 10, 3))],
        {"axis": 1, "pad_value": 0},
    ),
    (
        ArrayCropLayer,
        [tf.constant(1.0, shape=(1, 4))],
        {"array_length": 3, "pad_value": -1.0},
    ),
    (
        BearingAngleLayer,
        [
            tf.constant(0.0, shape=(100, 10, 1)),
            tf.constant(90.0, shape=(100, 10, 1)),
        ],
        {"lat_lon_constant": [-45.9, 180.67]},
    ),
    (
        BinLayer,
        [tf.random.normal((100, 56, 3))],
        {
            "condition_operators": ["eq", "neq", "lt", "leq", "gt", "geq"],
            "bin_values": [0, 1, 2, 3, 4, 5],
            "bin_labels": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "default_label": 6.0,
        },
    ),
    (
        ConditionalStandardScaleLayer,
        [tf.random.normal((100, 10, 5))],
        {
            "mean": [0.0, 1.0, 5.6, 7.8, 9.0],
            "variance": [1.0, 1.0, 1.0, 1.0, 1.0],
            "axis": -1,
            "skip_zeros": True,
        },
    ),
    (
        CosineSimilarityLayer,
        [tf.random.normal((100, 10, 10, 5)), tf.random.normal((100, 10, 10, 5))],
        None,
    ),
    (
        PairwiseCosineSimilarityLayer,
        [tf.random.normal((32, 4)), tf.random.normal((32, 12))],
        {"embedding_dim": 4},
    ),
    (DivideLayer, [tf.random.normal((100, 10, 5))], {"divisor": 2}),
    (ExpLayer, [tf.random.normal((100, 10, 5))], None),
    (ExponentLayer, [tf.random.normal((100, 10, 5))], {"exponent": 2}),
    (
        HaversineDistanceLayer,
        [
            tf.constant(-90.0, shape=(100, 10, 1)),
            tf.constant(178.9, shape=(100, 10, 1)),
        ],
        {"lat_lon_constant": [-45.9, 180.67], "unit": "miles"},
    ),
    (IdentityLayer, [tf.random.normal((100, 10, 5))], None),
    (
        ImputeLayer,
        [tf.constant([[[-999.0], [6.0], [9.0], [100.0]]])],
        {
            "impute_value": 2.0,
            "mask_value": -999.0,
        },
    ),
    (LogLayer, [tf.random.normal((100, 10, 5))], None),
    (
        LogicalAndLayer,
        [tf.constant(True, shape=(10, 1, 5)), tf.constant(False, shape=(10, 1, 5))],
        None,
    ),
    (LogicalNotLayer, [tf.constant(True, shape=(10, 1, 5))], None),
    (
        LogicalOrLayer,
        [tf.constant(True, shape=(10, 1, 5)), tf.constant(False, shape=(10, 1, 5))],
        None,
    ),
    (MaxLayer, [tf.random.normal((100, 10, 5))], {"max_constant": 10}),
    (MeanLayer, [tf.random.normal((100, 10, 5))], {"mean_constant": 10}),
    (MinLayer, [tf.random.normal((100, 10, 5))], {"min_constant": 10}),
    (
        MinMaxScaleLayer,
        [
            tf.concat(
                [
                    tf.random.uniform((100, 10, 1), minval=-i, maxval=i)
                    for i in range(1, 6)
                ],
                axis=-1,
            )
        ],
        {
            "min": [-i for i in range(1, 6)],
            "max": [i for i in range(1, 6)],
            "axis": -1,
        },
    ),
    (ModuloLayer, [tf.random.normal((1000, 32, 1))], {"divisor": 10}),
    (MultiplyLayer, [tf.random.normal((1, 5))], {"multiplier": 50}),
    (
        NumericalIfStatementLayer,
        [tf.random.normal((100, 10, 5)), tf.random.normal((100, 10, 5))],
        {"condition_operator": "gt", "value_to_compare": 5, "result_if_true": 1},
    ),
    (
        RoundLayer,
        [tf.random.normal((10, 10, 10, 1))],
        {"round_type": "ceil"},
    ),
    (RoundToDecimalLayer, [tf.random.normal((100, 5))], {"decimals": 2}),
    (
        StandardScaleLayer,
        [tf.random.normal((100, 10, 5))],
        {
            "mean": [0.0, 1.0, 5.6, 7.8, 9.0],
            "variance": [1.0, 1.0, 1.0, 1.0, 1.0],
            "axis": -1,
        },
    ),
    (SubtractLayer, [tf.random.normal((100, 10, 5))], {"subtrahend": 10}),
    (SumLayer, [tf.random.normal((100, 10, 5))], {"addend": -1}),
    # TF-only JIT-compatible layers
    (
        ListRankLayer,
        [tf.random.normal((1, 2, 3))],
        {"axis": 1, "sort_order": "desc"},
    ),
    (
        BucketizeLayer,
        [tf.random.normal((100, 1))],
        {"splits": [-0.5, 0, 0.1, 0.2, 3]},
    ),
    (ListMaxLayer, [tf.random.normal((100, 10, 5))], None),
    (ListMeanLayer, [tf.random.normal((100, 10, 5))], None),
    (
        ListMedianLayer,
        [tf.constant([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])],
        {
            "axis": 1,
            "top_n": 5,
            "sort_order": "desc",
            "nan_fill_value": 0,
            "min_filter_value": 0,
        },
    ),
    (ListMinLayer, [tf.random.normal((100, 10, 5))], None),
    (
        ListStdDevLayer,
        [tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])],
        {
            "axis": -1,
            "top_n": 5,
            "sort_order": "desc",
            "nan_fill_value": 0,
            "min_filter_value": 0,
        },
    ),
]


# JIT-incompatible layers (jit_compatible = False)
JIT_INCOMPATIBLE_LAYERS = [
    (
        BloomEncodeLayer,
        [tf.strings.as_string(tf.random.normal((100, 23, 32, 1)))],
        {"num_hash_fns": 3, "num_bins": 100},
    ),
    (CurrentDateLayer, [tf.constant(100, shape=(100, 10, 1))], None),
    (CurrentDateTimeLayer, [tf.constant(100, shape=(100, 10, 1))], None),
    (
        CurrentUnixTimestampLayer,
        [tf.constant(100, shape=(100, 10, 1))],
        {"unit": "ms"},
    ),
    (
        DateAddLayer,
        [
            tf.constant("2023-03-02", shape=(100, 10, 1)),
        ],
        {"num_days": 10},
    ),
    (
        DateDiffLayer,
        [
            tf.constant("2023-03-02", shape=(100, 10, 1)),
            tf.constant("2023-02-02", shape=(100, 10, 1)),
        ],
        {"default_value": 1},
    ),
    (
        DateParseLayer,
        [tf.constant("2023-02-02", shape=(100, 10, 1))],
        {"date_part": "DayOfWeek", "default_value": 1},
    ),
    (
        DateTimeToUnixTimestampLayer,
        [tf.constant("2021-07-14", shape=(100, 10, 1))],
        {"unit": "s"},
    ),
    (
        HashIndexLayer,
        [tf.strings.as_string(tf.random.normal((100, 10, 5)))],
        {"num_bins": 100},
    ),
    (
        IfStatementLayer,
        [tf.constant("hello", shape=(100, 10, 5))],
        {
            "condition_operator": "eq",
            "value_to_compare": "world",
            "result_if_true": "yes",
            "result_if_false": "no",
        },
    ),
    (
        LambdaFunctionLayer,
        [tf.constant([[1, 2, 3], [4, 5, 6]])],
        {
            "function": lambda x: tf.square(x),
            "input_dtype": "float",
            "output_dtype": "float",
            "output_shape": (3,),
        },
    ),
    (
        MinHashIndexLayer,
        [tf.strings.as_string(tf.random.normal((100, 10, 5)))],
        {"num_permutations": 10, "mask_value": None, "axis": -1},
    ),
    (
        OneHotEncodeLayer,
        [tf.constant("a", shape=(100, 10, 1))],
        {"num_oov_indices": 1, "vocabulary": ["a", "b"], "drop_unseen": True},
    ),
    (
        OneHotLayer,
        [tf.constant("a", shape=(100, 10, 1))],
        {"num_oov_indices": 1, "vocabulary": ["a", "b"], "drop_unseen": True},
    ),
    (
        OrdinalArrayEncodeLayer,
        [tf.constant([["a", "a", "b", "-1"]])],
        {"pad_value": "-1"},
    ),
    (
        StringAffixLayer,
        [tf.constant("a", shape=(100, 10, 1))],
        {"prefix": "b", "suffix": "c"},
    ),
    (
        StringArrayConstantLayer,
        [tf.constant("a", shape=(100, 10, 1))],
        {"constant_string_array": "b"},
    ),
    (
        StringCaseLayer,
        [tf.constant("hEllO wOrLd", shape=(100, 10, 1))],
        {"string_case_type": "lower"},
    ),
    (
        StringConcatenateLayer,
        [
            tf.constant("a", shape=(10, 1, 1, 5, 2)),
            tf.constant("b", shape=(10, 1, 1, 5, 2)),
        ],
        {"separator": "y"},
    ),
    (
        StringContainsLayer,
        [
            tf.constant("a", shape=(100, 10, 1)),
            tf.constant("b", shape=(100, 10, 1)),
        ],
        {"negation": True},
    ),
    (
        StringContainsListLayer,
        [tf.constant("a", shape=(230, 67, 1))],
        {"negation": True, "string_constant_list": ["a", "b", "c"]},
    ),
    (
        StringEqualsIfStatementLayer,
        [
            tf.constant("a", shape=(23, 1, 1, 67)),
            tf.constant("b", shape=(23, 1, 1, 67)),
        ],
        {"result_if_true": "a", "result_if_false": "b"},
    ),
    (
        StringIndexLayer,
        [tf.constant("a", shape=(23, 5))],
        {
            "num_oov_indices": 2,
            "encoding": "utf-8",
            "vocabulary": ["a", "b"],
            "mask_token": "c",
        },
    ),
    (
        StringIsInListLayer,
        [tf.constant("a", shape=(23, 5))],
        {"string_constant_list": ["a", "b", "c"], "negation": False},
    ),
    (
        StringListToStringLayer,
        [tf.constant("a", shape=(23, 5))],
        {"separator": "b", "axis": -1},
    ),
    (
        StringMapLayer,
        [tf.constant("a", shape=(100, 5))],
        {
            "string_match_values": ["a", "c"],
            "string_replace_values": ["b", "c"],
            "default_replace_value": "z",
        },
    ),
    (
        StringReplaceLayer,
        [tf.constant("a_b_c_d_e", shape=(1, 5, 45))],
        {
            "string_match_constant": "_",
            "string_replace_constant": "-",
            "regex": False,
        },
    ),
    (
        StringToStringListLayer,
        [tf.constant("a", shape=(100, 5))],
        {"separator": "b", "default_value": "hello", "list_length": 5},
    ),
    (
        SubStringDelimAtIndexLayer,
        [tf.constant("a_b_c_d_e", shape=(1, 5, 45))],
        {"delimiter": "_", "index": 3, "default_value": "hello"},
    ),
    (
        UnixTimestampToDateTimeLayer,
        [tf.constant(100000, shape=(100, 10, 1), dtype=tf.int64)],
        {"include_time": True, "unit": "s"},
    ),
]


@pytest.mark.parametrize("layer_cls, input_tensors, kwargs", JIT_COMPATIBLE_LAYERS)
def test_jit_compatible_layers_pass(layer_cls, input_tensors, kwargs):
    """Test that layers marked jit_compatible=True can be JIT-compiled."""
    if kwargs is None:
        kwargs = {}

    layer = layer_cls(**kwargs)
    assert (
        layer.jit_compatible is True
    ), f"{layer_cls.__name__} should have jit_compatible=True"

    @tf.function(jit_compile=True)
    def jit_call(*inputs):
        if len(inputs) == 1:
            return layer(inputs[0])
        return layer(list(inputs))

    # Must not raise
    result = jit_call(*input_tensors)
    assert result is not None


@pytest.mark.parametrize("layer_cls, input_tensors, kwargs", JIT_INCOMPATIBLE_LAYERS)
def test_jit_incompatible_layers_fail(layer_cls, input_tensors, kwargs):
    """Test that layers marked jit_compatible=False fail JIT compilation.

    This ensures that if a layer becomes JIT-safe (e.g., TF upgrade), the test
    breaks and prompts the developer to update the jit_compatible flag.
    """
    if kwargs is None:
        kwargs = {}

    layer = layer_cls(**kwargs)
    assert (
        layer.jit_compatible is False
    ), f"{layer_cls.__name__} should have jit_compatible=False"

    @tf.function(jit_compile=True)
    def jit_call(*inputs):
        if len(inputs) == 1:
            return layer(inputs[0])
        return layer(list(inputs))

    # Must raise Exception when trying to JIT compile
    with pytest.raises(Exception):
        result = jit_call(*input_tensors)
        # Force evaluation if result is symbolic
        if hasattr(result, "numpy"):
            result.numpy()


def test_all_layers_have_jit_compatible_attribute():
    """Test that all layers have jit_compatible attribute defined."""
    # Get all classes from kamae.keras.core.layers (multi-backend)
    multi_backend_layers = [
        obj
        for name, obj in vars(core_layers_mod).items()
        if isinstance(obj, type)
        and issubclass(obj, keras.Layer)
        and obj is not keras.Layer
        and name != "BaseLayer"  # Exclude base class
    ]

    # Get all classes from kamae.keras.tensorflow.layers (TF-only)
    tf_only_layers = [
        obj
        for name, obj in vars(tf_layers_mod).items()
        if isinstance(obj, type)
        and issubclass(obj, tf.keras.layers.Layer)
        and obj is not tf.keras.layers.Layer
    ]

    all_layers = multi_backend_layers + tf_only_layers

    for layer_cls in all_layers:
        assert hasattr(
            layer_cls, "jit_compatible"
        ), f"{layer_cls.__name__} missing jit_compatible attribute"
        assert isinstance(
            layer_cls.jit_compatible, bool
        ), f"{layer_cls.__name__}.jit_compatible must be bool, got {type(layer_cls.jit_compatible)}"


def test_all_layers_in_jit_tests():
    """Test that all layers appear in exactly one of the JIT test lists."""
    # Get all layer classes
    multi_backend_layers = [
        obj
        for name, obj in vars(core_layers_mod).items()
        if isinstance(obj, type)
        and issubclass(obj, keras.Layer)
        and obj is not keras.Layer
        and name != "BaseLayer"
    ]

    tf_only_layers = [
        obj
        for name, obj in vars(tf_layers_mod).items()
        if isinstance(obj, type)
        and issubclass(obj, tf.keras.layers.Layer)
        and obj is not tf.keras.layers.Layer
    ]

    all_layers = set(multi_backend_layers + tf_only_layers)

    # Get tested layers
    jit_compatible_tested = {param[0] for param in JIT_COMPATIBLE_LAYERS}
    jit_incompatible_tested = {param[0] for param in JIT_INCOMPATIBLE_LAYERS}

    # Check coverage
    tested_layers = jit_compatible_tested | jit_incompatible_tested
    missing = all_layers - tested_layers
    assert (
        not missing
    ), f"Layers missing from JIT tests: {[l.__name__ for l in missing]}"

    # Check no duplicates
    duplicates = jit_compatible_tested & jit_incompatible_tested
    assert (
        not duplicates
    ), f"Layers in both JIT test lists: {[l.__name__ for l in duplicates]}"
