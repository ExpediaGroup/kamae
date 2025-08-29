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
import inspect
import os

import keras
import numpy as np
import pytest
import tensorflow as tf
from packaging.version import Version

import kamae.tensorflow.layers as layers_mod

keras_version = Version(keras.__version__)
# If keras >= 2.13.0, we need to enable unsafe deserialization in order to load the
# LambdaFunctionLayer.
# Before 2.13.0, keras the default behavior is to allow unsafe deserialization.
if keras_version >= Version("2.13.0"):
    from keras.src.saving import serialization_lib

    serialization_lib.enable_unsafe_deserialization()

is_keras_3 = keras_version >= Version("3.0.0")

from kamae.tensorflow.layers import (
    AbsoluteValueLayer,
    ArrayConcatenateLayer,
    ArrayCropLayer,
    ArraySplitLayer,
    ArraySubtractMinimumLayer,
    BearingAngleLayer,
    BinLayer,
    BloomEncodeLayer,
    BucketizeLayer,
    ConditionalStandardScaleLayer,
    CosineSimilarityLayer,
    CurrentDateLayer,
    CurrentDateTimeLayer,
    CurrentUnixTimestampLayer,
    DateAddLayer,
    DateDiffLayer,
    DateParseLayer,
    DateTimeToUnixTimestampLayer,
    DivideLayer,
    ExpLayer,
    ExponentLayer,
    HashIndexLayer,
    HaversineDistanceLayer,
    IdentityLayer,
    IfStatementLayer,
    ImputeLayer,
    LambdaFunctionLayer,
    ListMaxLayer,
    ListMeanLayer,
    ListMedianLayer,
    ListMinLayer,
    ListRankLayer,
    ListStdDevLayer,
    LogicalAndLayer,
    LogicalNotLayer,
    LogicalOrLayer,
    LogLayer,
    MaxLayer,
    MeanLayer,
    MinHashIndexLayer,
    MinLayer,
    MinMaxScaleLayer,
    ModuloLayer,
    MultiplyLayer,
    NumericalIfStatementLayer,
    OneHotEncodeLayer,
    OneHotLayer,
    OrdinalArrayEncodeLayer,
    RoundLayer,
    RoundToDecimalLayer,
    StandardScaleLayer,
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
    SubtractLayer,
    SumLayer,
    UnixTimestampToDateTimeLayer,
)


@pytest.mark.parametrize(
    # Skip predict should only be used for layers that are not deterministic
    # e.g. datetime/timestamp layers
    "layer_cls, input_tensors, kwargs, skip_predict",
    [
        (AbsoluteValueLayer, [tf.random.normal((32, 10))], None, False),
        (
            ArrayConcatenateLayer,
            [tf.random.normal((32, 10, 100, 3)), tf.random.normal((32, 10, 100, 3))],
            {"axis": -2},
            False,
        ),
        (ArraySplitLayer, [tf.random.normal((32, 10, 100, 3))], {"axis": -2}, False),
        (
            ArraySubtractMinimumLayer,
            [tf.random.normal((32, 10, 10, 3))],
            {"axis": 1, "pad_value": 0},
            False,
        ),
        (
            ArrayCropLayer,
            [tf.constant("a", shape=(1, 4))],
            {"array_length": 3, "pad_value": "-1"},
            False,
        ),
        (
            BearingAngleLayer,
            [
                tf.constant(0.0, shape=(100, 10, 1)),
                tf.constant(90.0, shape=(100, 10, 1)),
            ],
            {"lat_lon_constant": [-45.9, 180.67]},
            False,
        ),
        (
            BinLayer,
            [tf.random.normal((100, 56, 3))],
            {
                "condition_operators": ["eq", "neq", "lt", "leq", "gt", "geq"],
                "bin_values": [0, 1, 2, 3, 4, 5],
                "bin_labels": ["a", "b", "c", "d", "e", "f"],
                "default_label": "g",
            },
            False,
        ),
        (
            BloomEncodeLayer,
            [tf.strings.as_string(tf.random.normal((100, 23, 32, 1)))],
            {"num_hash_fns": 3, "num_bins": 100},
            False,
        ),
        (
            BucketizeLayer,
            [tf.random.normal((100, 1))],
            {"splits": [-0.5, 0, 0.1, 0.2, 3]},
            False,
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
            False,
        ),
        (
            CosineSimilarityLayer,
            [tf.random.normal((100, 10, 10, 5)), tf.random.normal((100, 10, 10, 5))],
            None,
            False,
        ),
        (CurrentDateLayer, [tf.constant(100, shape=(100, 10, 1))], None, False),
        (CurrentDateTimeLayer, [tf.constant(100, shape=(100, 10, 1))], None, True),
        (
            CurrentUnixTimestampLayer,
            [tf.constant(100, shape=(100, 10, 1))],
            {"unit": "ms"},
            True,
        ),
        (
            DateAddLayer,
            [
                tf.constant("2023-03-02", shape=(100, 10, 1)),
            ],
            {"num_days": 10},
            False,
        ),
        (
            DateDiffLayer,
            [
                tf.constant("2023-03-02", shape=(100, 10, 1)),
                tf.constant("2023-02-02", shape=(100, 10, 1)),
            ],
            {"default_value": 1},
            False,
        ),
        (
            DateParseLayer,
            [tf.constant("2023-02-02", shape=(100, 10, 1))],
            {"date_part": "DayOfWeek", "default_value": 1},
            False,
        ),
        (
            DateTimeToUnixTimestampLayer,
            [tf.constant("2021-07-14", shape=(100, 10, 1))],
            {"unit": "s"},
            False,
        ),
        (DivideLayer, [tf.random.normal((100, 10, 5))], {"divisor": 2}, False),
        (ExpLayer, [tf.random.normal((100, 10, 5))], None, False),
        (ExponentLayer, [tf.random.normal((100, 10, 5))], {"exponent": 2}, False),
        (
            HashIndexLayer,
            [tf.strings.as_string(tf.random.normal((100, 10, 5)))],
            {"num_bins": 100},
            False,
        ),
        (
            HaversineDistanceLayer,
            [
                tf.constant(-90.0, shape=(100, 10, 1)),
                tf.constant(178.9, shape=(100, 10, 1)),
            ],
            {"lat_lon_constant": [-45.9, 180.67], "unit": "miles"},
            False,
        ),
        (IdentityLayer, [tf.random.normal((100, 10, 5))], None, False),
        (
            IfStatementLayer,
            [tf.random.normal((100, 10, 5)), tf.random.normal((100, 10, 5))],
            {
                "condition_operator": "gt",
                "value_to_compare": 5.0,
                "result_if_true": 1.0,
            },
            False,
        ),
        (
            ListRankLayer,
            [tf.random.normal((1, 2, 3))],
            {"axis": 1, "sort_order": "desc"},
            False,
        ),
        (LogLayer, [tf.random.normal((100, 10, 5))], None, False),
        (
            LogicalAndLayer,
            [tf.constant(True, shape=(10, 1, 5)), tf.constant(False, shape=(10, 1, 5))],
            None,
            False,
        ),
        (LogicalNotLayer, [tf.constant(True, shape=(10, 1, 5))], None, False),
        (
            LogicalOrLayer,
            [tf.constant(True, shape=(10, 1, 5)), tf.constant(False, shape=(10, 1, 5))],
            None,
            False,
        ),
        (MaxLayer, [tf.random.normal((100, 10, 5))], {"max_constant": 10}, False),
        (MeanLayer, [tf.random.normal((100, 10, 5))], {"mean_constant": 10}, False),
        (MinLayer, [tf.random.normal((100, 10, 5))], {"min_constant": 10}, False),
        (
            MinHashIndexLayer,
            [tf.strings.as_string(tf.random.normal((100, 10, 5)))],
            {"num_permutations": 10, "mask_value": None, "axis": -1},
            False,
        ),
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
            False,
        ),
        (ModuloLayer, [tf.random.normal((1000, 32, 1))], {"divisor": 10}, False),
        (MultiplyLayer, [tf.random.normal((1, 5))], {"multiplier": 50}, False),
        (
            NumericalIfStatementLayer,
            [tf.random.normal((100, 10, 5)), tf.random.normal((100, 10, 5))],
            {"condition_operator": "gt", "value_to_compare": 5, "result_if_true": 1},
            False,
        ),
        (
            OneHotEncodeLayer,
            [tf.constant("a", shape=(100, 10, 1))],
            {"num_oov_indices": 1, "vocabulary": ["a", "b"], "drop_unseen": True},
            False,
        ),
        (
            OneHotLayer,
            [tf.constant("a", shape=(100, 10, 1))],
            {"num_oov_indices": 1, "vocabulary": ["a", "b"], "drop_unseen": True},
            False,
        ),
        (
            OrdinalArrayEncodeLayer,
            [tf.constant([["a", "a", "b", "-1"]])],
            {"pad_value": "-1"},
            False,
        ),
        (
            RoundLayer,
            [tf.random.normal((10, 10, 10, 1))],
            {"round_type": "ceil"},
            False,
        ),
        (RoundToDecimalLayer, [tf.random.normal((100, 5))], {"decimals": 2}, False),
        (
            StandardScaleLayer,
            [tf.random.normal((100, 10, 5))],
            {
                "mean": [0.0, 1.0, 5.6, 7.8, 9.0],
                "variance": [1.0, 1.0, 1.0, 1.0, 1.0],
                "axis": -1,
            },
            False,
        ),
        (
            ImputeLayer,
            [tf.constant([[[-999.0], [6.0], [9.0], [100.0]]])],
            {
                "impute_value": 2.0,
                "mask_value": -999.0,
            },
            False,
        ),
        (
            LambdaFunctionLayer,
            [tf.constant([[1, 2, 3], [4, 5, 6]])],
            {
                "function": lambda x: tf.square(x) - tf.math.log(x),
                "input_dtype": "float",
                "output_dtype": "float",
            },
            False,
        ),
        (
            ListMaxLayer,
            [tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])],
            {
                "axis": 1,
                "top_n": 5,
                "sort_order": "descending",
                "nan_fill_value": 0,
                "min_filter_value": 0,
            },
            False,
        ),
        (
            ListMeanLayer,
            [tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])],
            {
                "axis": 1,
                "top_n": 5,
                "sort_order": "descending",
                "nan_fill_value": 0,
                "min_filter_value": 0,
            },
            False,
        ),
        (
            ListMedianLayer,
            [tf.constant([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])],
            {
                "axis": 1,
                "top_n": 5,
                "sort_order": "descending",
                "nan_fill_value": 0,
                "min_filter_value": 0,
            },
            False,
        ),
        (
            ListMinLayer,
            [tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])],
            {
                "axis": -1,
                "top_n": 5,
                "sort_order": "descending",
                "nan_fill_value": 0,
                "min_filter_value": 0,
            },
            False,
        ),
        (
            ListStdDevLayer,
            [tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])],
            {
                "axis": -1,
                "top_n": 5,
                "sort_order": "descending",
                "nan_fill_value": 0,
                "min_filter_value": 0,
            },
            False,
        ),
        (
            StringAffixLayer,
            [tf.constant("a", shape=(100, 10, 1))],
            {"prefix": "b", "suffix": "c"},
            False,
        ),
        (
            StringArrayConstantLayer,
            [tf.constant("a", shape=(100, 10, 1))],
            {"constant_string_array": "b"},
            False,
        ),
        (
            StringCaseLayer,
            [tf.constant("hEllO wOrLd", shape=(100, 10, 1))],
            {"string_case_type": "lower"},
            False,
        ),
        (
            StringConcatenateLayer,
            [
                tf.constant("a", shape=(10, 1, 1, 5, 2)),
                tf.constant("b", shape=(10, 1, 1, 5, 2)),
            ],
            {"separator": "y"},
            False,
        ),
        (
            StringContainsLayer,
            [
                tf.constant("a", shape=(100, 10, 1)),
                tf.constant("b", shape=(100, 10, 1)),
            ],
            {"negation": True},
            False,
        ),
        (
            StringContainsListLayer,
            [tf.constant("a", shape=(230, 67, 1))],
            {"negation": True, "string_constant_list": ["a", "b", "c"]},
            False,
        ),
        (
            StringEqualsIfStatementLayer,
            [
                tf.constant("a", shape=(23, 1, 1, 67)),
                tf.constant("b", shape=(23, 1, 1, 67)),
            ],
            {"result_if_true": "a", "result_if_false": "b"},
            False,
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
            False,
        ),
        (
            StringIsInListLayer,
            [tf.constant("a", shape=(23, 5))],
            {"string_constant_list": ["a", "b", "c"], "negation": False},
            False,
        ),
        (
            StringListToStringLayer,
            [tf.constant("a", shape=(23, 5))],
            {"separator": "b", "axis": -1},
            False,
        ),
        (
            StringMapLayer,
            [tf.constant("a", shape=(100, 5))],
            {
                "string_match_values": ["a", "c"],
                "string_replace_values": ["b", "c"],
                "default_replace_value": "z",
            },
            False,
        ),
        (
            StringReplaceLayer,
            [tf.constant("a_b_c_d_e", shape=(1, 5, 45))],
            {
                "string_match_constant": "_",
                "string_replace_constant": "-",
                "regex": False,
            },
            False,
        ),
        (
            StringToStringListLayer,
            [tf.constant("a", shape=(100, 5))],
            {"separator": "b", "default_value": "hello", "list_length": 5},
            False,
        ),
        (
            SubStringDelimAtIndexLayer,
            [tf.constant("a_b_c_d_e", shape=(1, 5, 45))],
            {"delimiter": "_", "index": 3, "default_value": "hello"},
            False,
        ),
        (SubtractLayer, [tf.random.normal((100, 10, 5))], {"subtrahend": 10}, False),
        (SumLayer, [tf.random.normal((100, 10, 5))], {"addend": -1}, False),
        (
            UnixTimestampToDateTimeLayer,
            [tf.constant(100000, shape=(100, 10, 1), dtype=tf.int64)],
            {"include_time": True, "unit": "s"},
            False,
        ),
    ],
)
def test_layer_serialisation(
    tmp_path,
    layer_cls,
    input_tensors,
    skip_predict,
    kwargs,
):
    """
    Tests whether a layer is serialisable in a Model and that the output from the model
    matches calling the layer directly.
    """
    if is_keras_3 and layer_cls == LambdaFunctionLayer:
        # TODO: Understand why
        pytest.skip(reason="LambdaFunctionLayer does not serialise properly in keras 3")
    if kwargs is None:
        kwargs = {}

    # instantiation
    layer = layer_cls(**kwargs)

    # test get_weights , set_weights at layer level
    weights = layer.get_weights()
    layer.set_weights(weights)

    layer_config = layer.get_config()
    recovered_layer = layer.__class__.from_config(layer_config)
    recovered_layer.set_weights(weights)

    # test in functional API
    model_inputs = [
        tf.keras.layers.Input(shape=inp.shape[1:], dtype=inp.dtype)
        for inp in input_tensors
    ]
    model_outputs = layer(model_inputs if len(model_inputs) > 1 else model_inputs[0])

    # check with the functional API
    model = tf.keras.Model(inputs=model_inputs, outputs=model_outputs)

    # Test saving and reloading
    model_path = os.path.join(tmp_path, layer.name)
    if is_keras_3:
        model_path += ".keras"
    model.save(model_path)
    reloaded_model = tf.keras.models.load_model(model_path)

    # test serialization, weight setting at model level
    model_config = model.get_config()
    recovered_model = model.__class__.from_config(model_config)
    weights = model.get_weights()
    recovered_model.set_weights(weights)

    # Test model, recovered_model, layer & recovered_layer match
    model_output = model.predict(input_tensors)
    recovered_model_output = recovered_model.predict(input_tensors)
    reloaded_model_output = reloaded_model.predict(input_tensors)
    layer_output = layer(input_tensors)
    recovered_layer_output = recovered_layer(input_tensors)

    # Layers output (potentially lists of) tensors. We make these numpy arrays
    layer_output = (
        layer_output.numpy()
        if isinstance(layer_output, tf.Tensor)
        else np.array([out.numpy() for out in layer_output])
    )
    recovered_layer_output = (
        recovered_layer_output.numpy()
        if isinstance(recovered_layer_output, tf.Tensor)
        else np.array([out.numpy() for out in recovered_layer_output])
    )

    if not skip_predict:
        # Check that the model, recovered model, layer and recovered layer outputs match
        if np.issubdtype(layer_output.dtype, np.floating):
            np.testing.assert_allclose(model_output, recovered_model_output, atol=1e-6)
            np.testing.assert_allclose(model_output, reloaded_model_output, atol=1e-6)
            np.testing.assert_allclose(layer_output, recovered_layer_output, atol=1e-6)
            np.testing.assert_allclose(model_output, layer_output, atol=1e-6)
        else:
            np.testing.assert_equal(model_output, recovered_model_output)
            np.testing.assert_equal(model_output, reloaded_model_output)
            np.testing.assert_equal(layer_output, recovered_layer_output)
            np.testing.assert_equal(model_output, layer_output)


def test_all_layers_tested_for_serialisation():
    """
    Checks that all layers in kamae.tensorflow.layers have a serialisation test.
    """
    # Get all classes defined in kamae.tensorflow.layers
    all_layers = [
        obj
        for name, obj in vars(layers_mod).items()
        if isinstance(obj, type)
        and issubclass(obj, tf.keras.layers.Layer)
        and obj is not tf.keras.layers.Layer
    ]

    # Extract all layer_cls from the test parameterization
    parametrize_mark = next(
        mark
        for mark in getattr(test_layer_serialisation, "pytestmark", [])
        if getattr(mark, "name", None) == "parametrize"
    )
    tested_layers = {param[0] for param in parametrize_mark.args[1]}

    missing = [layer for layer in all_layers if layer not in tested_layers]
    assert (
        not missing
    ), f"Missing serialisation tests for: {[l.__name__ for l in missing]}"
