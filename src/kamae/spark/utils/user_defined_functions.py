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

from typing import List, Optional, Union

import tensorflow as tf

from kamae.spark.utils.indexer_utils import safe_hash64


def hash_udf(
    label: str, num_bins: int, mask_value: Optional[str] = None
) -> Union[int, None]:
    """
    User defined Spark function (UDF) to hash a string to an integer value.

    This uses FarmHash64. We have a python binding of the Google library
    https://github.com/google/farmhash within the pyfarmhash package.

    Therefore, this matches the hash function used in the tensorflow layer.

    :param label: String to hash.
    :param num_bins: Number of bins to hash to.
    :param mask_value: Mask value to use for hash indexing.
    :returns: Hashed integer value.
    """
    if label is None:
        return None
    if label == mask_value:
        return 0

    hash_val = safe_hash64(label)
    if mask_value is not None:
        # If masking value is set, then the zero index is reserved for it.
        # Therefore, we reduce the num_bins by 1 and add 1 to the binned value.
        return (hash_val % (num_bins - 1)) + 1
    else:
        return hash_val % num_bins


def indexer_udf(
    label: str,
    labels: List[str],
    mask_token: Optional[str] = None,
    num_oov_indices: int = 1,
) -> int:
    """
    User defined Spark function (UDF) to index a label based on the labels array
    mask token, and number of out of vocabulary indices.

    If the label is the mask token, we return 0. If the label is not in the labels
    array and num_oov_indices is 0, we raise an error. Otherwise, we calculate the
    offset for the index based on the mask token and num_oov_indices. Lastly, if
    the number of out of vocabulary indices is more than 1 we hash the label to an
    out of vocabulary index.

    Raises an error if the provided label is hashed and contains a null character.

    :param label: Current label to index.
    :param labels: Array of labels to index.
    :param mask_token: Mask token to use for the 0 index.
    :param num_oov_indices: Number of out of vocabulary indices.
    :returns: Indexed integer value.
    """
    if label is None:
        if num_oov_indices > 0:
            return 0
        else:
            raise ValueError(
                """Found null label but numOOVIndices is 0.
                Consider setting numOOVIndices to 1 or more to cater for null labels."""
            )

    if mask_token is not None and label == mask_token:
        # If the label is the mask token,
        # we want to return 0, since 0 is reserved for masked tokens
        return 0

    # Calculate the offset for the index
    start_index = 1 if mask_token is not None else 0
    offset = num_oov_indices + start_index

    # Attempt to get the index.
    try:
        return labels.index(label) + offset
    except ValueError:
        if num_oov_indices == 0:
            # If we have no out of vocabulary indices and the label is not in the
            # labels array, raise an error.
            raise ValueError(
                f"""Label {label} not found in labels array and
                numOOVIndices is 0. Consider setting numOOVIndices to 1 or more."""
            )
        # If the label is not in the labels array and not equal to the mask token,
        # then we need to hash it to an out of vocabulary index.
        hashed_int = safe_hash64(label)
        return (hashed_int % num_oov_indices) + start_index


def one_hot_encoding_udf(
    label: str,
    labels: List[str],
    mask_token: Optional[str] = None,
    num_oov_indices: int = 1,
    drop_unseen: bool = False,
) -> List[float]:
    """
    User defined Spark function (UDF) to create a one-hot encoding of a label based
    on the labels array, mask token, number of out of vocabulary indices, and whether
    to drop unseen labels.

    :param label: Current label to index.
    :param labels: Array of labels to index.
    :param mask_token: Mask token to use for the 0 index.
    :param num_oov_indices: Number of out of vocabulary indices.
    :param drop_unseen: Whether to drop unseen label indices.
    :returns: List of floats representing the one-hot encoding.
    """
    index = indexer_udf(
        label=label,
        labels=labels,
        mask_token=mask_token,
        num_oov_indices=num_oov_indices,
    )
    mask_offset = 1 if mask_token is not None else 0
    if drop_unseen:
        encoding = [0.0] * len(labels)
        shifted_index = index - (num_oov_indices + mask_offset)
        if shifted_index >= 0:
            encoding[shifted_index] = 1.0
    else:
        encoding = [0.0] * (len(labels) + num_oov_indices + mask_offset)
        encoding[index] = 1.0
    return encoding


def ordinal_array_encode_udf(
    value: List[str], pad_value: Optional[str] = None
) -> List[int]:
    """
    User defined Spark function (UDF) to encode a list of strings as an ordinal array.
    Example:
    value = ['a', 'b', 'c']
    ordinal_array = [0, 1, 2]

    :param value: List of strings to encode.
    :param pad_value: Value to use for padding. Padded values get an index of -1.
    :returns: List of integers representing the ordinal array.
    """
    string_index_mapping = {pad_value: -1}
    ordinal_array = []
    for string in value:
        if string not in string_index_mapping:
            string_index_mapping[string] = len(string_index_mapping) - 1
        ordinal_array.append(string_index_mapping[string])
    return ordinal_array


def min_hash_udf(
    labels: List[str], num_permutations: int, mask_value: Optional[str] = None
) -> List[int]:
    """
    User defined Spark function (UDF) to encode a list of strings as a min hash array.

    :param labels: List of strings to encode.
    :param num_permutations: Number of permutations to use. Output will be a list of
    size num_permutations.
    :param mask_value: Mask value to use for hash indexing. If set, the mask value
    will be ignored when computing the min hash.
    :returns: List of integers representing the min hash array.
    """
    min_hash_array = []
    if not labels:
        # Ensure at least one label
        labels.append("")
    for i in range(num_permutations):
        # Set the number of bins to the maximum integer value. We just want to hash
        # the input without binning it, so we use the maximum integer value.
        # This matches the behavior of the TensorFlow layer.
        if mask_value is not None:
            hashed_vals = [
                tf.int32.max
                if label == mask_value
                else hash_udf(label=f"{label}{i}", num_bins=tf.int32.max)
                for label in labels
            ]
        else:
            hashed_vals = [
                hash_udf(label=f"{label}{i}", num_bins=tf.int32.max) for label in labels
            ]
        min_hash_val = min(hashed_vals)
        min_hash_bit = min_hash_val & 1
        min_hash_array.append(min_hash_bit)

    return min_hash_array
