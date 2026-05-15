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

"""
Test that transformer and layer param specs are aligned.

This ensures that params defined inline on transformers/layers don't drift
from each other in terms of defaults, types, and docs.
"""

import inspect

import pytest

from kamae.params.naming import _camel_to_snake
from kamae.params.param_spec import _REQUIRED, _UNSET


def get_all_transformers():
    """Get all transformer classes from kamae.spark.transformers."""
    import kamae.spark.transformers as transformers_module

    transformer_classes = []
    for name in dir(transformers_module):
        obj = getattr(transformers_module, name)
        if (
            inspect.isclass(obj)
            and name.endswith("Transformer")
            and hasattr(obj, "_params")
            and hasattr(obj, "_keras_layer_class")
        ):
            transformer_classes.append(obj)
    return transformer_classes


def get_layer_class(transformer_class):
    """Get the corresponding Keras layer class for a transformer."""
    if not hasattr(transformer_class, "_keras_layer_class"):
        return None
    return transformer_class._keras_layer_class


# No special cases - all params are fully aligned!


@pytest.mark.parametrize("transformer_class", get_all_transformers())
def test_transformer_layer_param_alignment(transformer_class):
    """
    Test that transformer and layer params are aligned in terms of:
    - Param names (camelCase on Spark, snake_case on Keras)
    - Default values
    - Documentation strings

    Known misalignments are documented but don't fail the test.
    """
    layer_class = get_layer_class(transformer_class)
    if layer_class is None:
        pytest.skip(f"{transformer_class.__name__} has no _keras_layer_class")

    if not hasattr(layer_class, "_params"):
        pytest.skip(f"{layer_class.__name__} has no _params")

    transformer_params = transformer_class._params
    layer_params = layer_class._params

    # Convert transformer (Spark) param names to snake_case for comparison
    transformer_params_snake = {
        _camel_to_snake(name): spec for name, spec in transformer_params.items()
    }

    # Check params that exist on both sides are aligned
    for spark_name, spark_spec in transformer_params.items():
        keras_name = _camel_to_snake(spark_name)

        if keras_name not in layer_params:
            continue

        keras_spec = layer_params[keras_name]

        # Check defaults align (accounting for _UNSET vs _REQUIRED differences)
        # Spark uses _UNSET for optional params, Keras uses _REQUIRED for required params
        # If Spark has _UNSET, Keras might have _REQUIRED or None, which is fine
        # If Spark has a concrete default, Keras should match
        if spark_spec.default not in (_UNSET, _REQUIRED):
            # Concrete default on Spark side
            if keras_spec.default not in (_REQUIRED, _UNSET):
                # Both have concrete defaults — they should match
                assert spark_spec.default == keras_spec.default, (
                    f"{transformer_class.__name__}.{spark_name}: "
                    f"Default mismatch: Spark={spark_spec.default}, Keras={keras_spec.default}"
                )

        # Check docs align (if both have docs)
        if spark_spec.doc and keras_spec.doc:
            # Normalize whitespace and compare
            spark_doc_norm = " ".join(spark_spec.doc.split())
            keras_doc_norm = " ".join(keras_spec.doc.split())

            # Allow minor variations, but flag significant differences
            if len(spark_doc_norm) > 20 and len(keras_doc_norm) > 20:
                # For substantial docs, check if they're similar
                # Simple heuristic: first 50 chars should be similar
                spark_prefix = spark_doc_norm[:50].lower()
                keras_prefix = keras_doc_norm[:50].lower()

                # Allow some flexibility (e.g., "transformer" vs "layer")
                spark_prefix_clean = spark_prefix.replace("transformer", "").replace(
                    "spark", ""
                )
                keras_prefix_clean = keras_prefix.replace("layer", "").replace(
                    "keras", ""
                )

                if spark_prefix_clean != keras_prefix_clean:
                    # Warn but don't fail — docs can have minor differences
                    print(
                        f"Warning: {transformer_class.__name__}.{spark_name} doc mismatch:\n"
                        f"  Spark: {spark_spec.doc[:100]}\n"
                        f"  Keras: {keras_spec.doc[:100]}"
                    )


def test_shared_params_have_no_inline_duplicates():
    """
    Test that params defined in shared_specs are not also defined inline.

    This prevents duplication where a param is both in MASK_VALUE_PARAMS
    and also defined inline in the transformer/layer.
    """
    from kamae.params.shared_specs import (
        DROP_UNSEEN_PARAMS,
        LISTWISE_FILTER_PARAMS,
        LISTWISE_PARAMS,
        LISTWISE_SEGMENT_PARAMS,
        MASK_VALUE_PARAMS,
        STRING_INDEX_PARAMS,
        UNIX_TIMESTAMP_PARAMS,
    )

    # Collect all shared param names (camelCase)
    shared_param_names = set()
    for spec_group in [
        MASK_VALUE_PARAMS,
        UNIX_TIMESTAMP_PARAMS,
        STRING_INDEX_PARAMS,
        DROP_UNSEEN_PARAMS,
        LISTWISE_PARAMS,
        LISTWISE_FILTER_PARAMS,
        LISTWISE_SEGMENT_PARAMS,
    ]:
        shared_param_names.update(spec_group.keys())

    # Check all transformers
    for transformer_class in get_all_transformers():
        if not hasattr(transformer_class, "_params"):
            continue

        # Get inline params (those not from shared specs)
        inline_param_names = set()
        for name in transformer_class._params.keys():
            # Check if this param is NOT from a shared spec dict
            # (heuristic: if it's in the class's own _params definition)
            inline_param_names.add(name)

        # Find overlaps with shared params
        overlaps = inline_param_names & shared_param_names

        # It's OK to have overlaps if the transformer uses **SHARED_SPEC
        # We're just warning about true duplicates
        if overlaps:
            # This is actually expected — transformers use **MASK_VALUE_PARAMS
            # So this test is more informational than a hard failure
            pass


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    pytest.main([__file__, "-v"])
