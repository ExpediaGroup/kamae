# Kamae
[![CI](https://github.com/ExpediaGroup/kamae/actions/workflows/ci.yaml/badge.svg)](https://github.com/ExpediaGroup/kamae/actions/workflows/ci.yaml)
![PyPI - Version](https://img.shields.io/pypi/v/kamae)

Kamae bridges the gap between offline data processing and online model serving. Build preprocessing pipelines in [Spark](https://spark.apache.org/) for big data workloads, then export them as [Keras](https://keras.io/) models for low-latency inference.

## Why Kamae?

Training and serving often happen on different platforms. Spark for batch processing at scale, TensorFlow for low-latency inference. Manually reimplementing preprocessing logic in both places creates:
- **Training/serving skew**: Subtle bugs from inconsistent implementations
- **Development overhead**: Writing and maintaining duplicate code
- **Deployment friction**: Changes require updates in multiple systems

Kamae solves this by generating the inference model directly from your Spark pipeline, guaranteeing consistency between training and serving.

## Installation

```bash
pip install kamae
```

**Platform notes**: Kamae supports `tensorflow>=2.9.1,<2.19.0`. For Mac ARM with `tensorflow<2.13.0`, install `tensorflow-macos` manually. TensorFlow no longer supports Mac x86_64 from version 2.18.0 onwards.

## Quick Start

```python
from pyspark.sql import SparkSession
from kamae.spark.estimators import StandardScaleEstimator, StringIndexEstimator
from kamae.spark.pipeline import KamaeSparkPipeline
from kamae.spark.transformers import LogTransformer, ArrayConcatenateTransformer

# Define preprocessing in Spark
spark = SparkSession.builder.getOrCreate()
data = spark.createDataFrame(
    [(1, 2, "a"), (4, 5, "b"), (7, 8, "c")],
    ["col1", "col2", "category"]
)

pipeline = KamaeSparkPipeline(stages=[
    LogTransformer(inputCol="col1", outputCol="log_col1", alpha=1, inputDtype="float"),
    ArrayConcatenateTransformer(inputCols=["log_col1", "col2"], outputCol="features", inputDtype="float"),
    StandardScaleEstimator(inputCol="features", outputCol="scaled_features"),
    StringIndexEstimator(inputCol="category", outputCol="category_indexed"),
])

fitted_pipeline = pipeline.fit(data)
fitted_pipeline.transform(data).show()  # Use in Spark

# Export for TensorFlow Serving
tf_input_schema = [
    {"name": "col1", "dtype": "int32", "shape": (None, 1)},
    {"name": "col2", "dtype": "int32", "shape": (None, 1)},
    {"name": "category", "dtype": "string", "shape": (None, 1)},
]
keras_model = fitted_pipeline.build_keras_model(tf_input_schema=tf_input_schema)
keras_model.save("./preprocessing_model.keras")
```

## Usage

**Spark Pipeline (Recommended)**: Build preprocessing pipelines using Spark transformers and estimators, fit on DataFrames, then export as Keras models. See [examples](examples/spark) for common patterns.

**Direct Keras Layers**: Import and compose Keras layers directly for non-tabular data or custom workflows. Browse available layers in the [transformation table](#supported-preprocessing-layers) below.

For Scikit-learn support (experimental, unmaintained), see [sklearn examples](examples/sklearn).

## Documentation

- **[Examples](examples/spark)**: Full working examples for common use cases
- **[Chaining models](docs/chaining_models.md)**: Use Kamae preprocessing models as inputs to trainable models
- **[Type parity](docs/achieving_type_parity.md)**: Ensuring consistent dtypes between Spark and Keras
- **[Shape parity](docs/achieving_shape_parity.md)**: Ensuring consistent shapes between Spark and Keras
- **[Testing inference](docs/testing_inference.md)**: Validate model outputs with TensorFlow Serving
- **[Adding transformers](docs/adding_transformer.md)**: Contributing new transformations

## Supported Preprocessing Layers

|         Transformation          |                                                                                   Description                                                                                   |                            Keras Layer                             |                             Spark Transformer                             |                  Scikit-learn Transformer                   |
|:-------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------:|:-------------------------------------------------------------------------:|:-----------------------------------------------------------:|
|          AbsoluteValue          |                                                                         Applies the `abs(x)` transform.                                                                         |       [Link](src/kamae/tensorflow/layers/absolute_value.py)        |          [Link](src/kamae/spark/transformers/absolute_value.py)           |                     Not yet implemented                     |
|        ArrayConcatenate         |                                                                Assembles multiple features into a single array.                                                                 |      [Link](src/kamae/tensorflow/layers/array_concatenate.py)      |         [Link](src/kamae/spark/transformers/array_concatenate.py)         | [Link](src/kamae/sklearn/transformers/array_concatenate.py) |
|            ArrayCrop            |                                                               Crops or pads a feature array to a consistent size.                                                               |         [Link](src/kamae/tensorflow/layers/array_crop.py)          |            [Link](src/kamae/spark/transformers/array_crop.py)             |                     Not yet implemented                     |
|           ArraySplit            |                                                                 Splits a feature array into multiple features.                                                                  |         [Link](src/kamae/tensorflow/layers/array_split.py)         |            [Link](src/kamae/spark/transformers/array_split.py)            |    [Link](src/kamae/sklearn/transformers/array_split.py)    |
|      ArraySubtractMinimum       |                                Subtracts the minimum element in an array from therest to compute a timestamp difference. Ignores padded values.                                 |   [Link](src/kamae/tensorflow/layers/array_subtract_minimum.py)    |      [Link](src/kamae/spark/transformers/array_subtract_minimum.py)       |                     Not yet implemented                     |
|          BearingAngle           |                                  Compute the bearing angle (https://en.wikipedia.org/wiki/Bearing_(navigation)) between two pairs of lat/long.                                  |        [Link](src/kamae/tensorflow/layers/bearing_angle.py)        |           [Link](src/kamae/spark/transformers/bearing_angle.py)           |                     Not yet implemented                     |
|               Bin               |                               Bins a numerical column into string categorical bins. Users can specify the bin values, labels and a default label.                               |             [Link](src/kamae/tensorflow/layers/bin.py)             |                [Link](src/kamae/spark/transformers/bin.py)                |                     Not yet implemented                     |
|           BloomEncode           | Hash encodes a string feature multiple times to create an array of indices. Useful for compressing input dimensions for embeddings. Paper: https://arxiv.org/pdf/1706.03993.pdf |        [Link](src/kamae/tensorflow/layers/bloom_encode.py)         |           [Link](src/kamae/spark/transformers/bloom_encode.py)            |                     Not yet implemented                     |
|            Bucketize            |                                                                  Buckets a numerical column into integer bins.                                                                  |          [Link](src/kamae/tensorflow/layers/bucketize.py)          |             [Link](src/kamae/spark/transformers/bucketize.py)             |                     Not yet implemented                     |
|    ConditionalStandardScale     |         Normalises by the mean and standard deviation, with ability to: apply a mask on another column, not scale the zeros, and apply a non standard scaling function.         | [Link](src/kamae/tensorflow/layers/conditional_standard_scale.py)  |     [Link](src/kamae/spark/estimators/conditional_standard_scale.py)      |                     Not yet implemented                     |
|        CosineSimilarity         |                                                           Computes the cosine similarity between two array features.                                                            |      [Link](src/kamae/tensorflow/layers/cosine_similarity.py)      |         [Link](src/kamae/spark/transformers/cosine_similarity.py)         |                     Not yet implemented                     |
|           CurrentDate           |                                                             Returns the current date for use in other transformers.                                                             |        [Link](src/kamae/tensorflow/layers/current_date.py)         |           [Link](src/kamae/spark/transformers/current_date.py)            |                     Not yet implemented                     |
|         CurrentDateTime         |                                       Returns the current date time in the format yyyy-MM-dd HH:mm:ss.SSS for use in other transformers.                                        |      [Link](src/kamae/tensorflow/layers/current_date_time.py)      |         [Link](src/kamae/spark/transformers/current_date_time.py)         |                     Not yet implemented                     |
|      CurrentUnixTimestamp       |                                       Returns the current unix timestamp in either seconds or milliseconds for use in other transformers.                                       |   [Link](src/kamae/tensorflow/layers/current_unix_timestamp.py)    |      [Link](src/kamae/spark/transformers/current_unix_timestamp.py)       |                     Not yet implemented                     |
|             DateAdd             |                            Adds a static or dynamic number of days to a date feature. NOTE: Destroys any time component of the datetime if present.                             |          [Link](src/kamae/tensorflow/layers/date_add.py)           |             [Link](src/kamae/spark/transformers/date_add.py)              |                     Not yet implemented                     |
|            DateDiff             |                                                             Computes the number of days between two date features.                                                              |          [Link](src/kamae/tensorflow/layers/date_diff.py)          |             [Link](src/kamae/spark/transformers/date_diff.py)             |                     Not yet implemented                     |
|            DateParse            |                                            Parses a string date of format YYYY-MM-DD to extract a given date part. E.g. day of year.                                            |         [Link](src/kamae/tensorflow/layers/date_parse.py)          |            [Link](src/kamae/spark/transformers/date_parse.py)             |                     Not yet implemented                     |
|     DateTimeToUnixTimestamp     |                                                                Converts a UTC datetime string to unix timestamp.                                                                | [Link](src/kamae/tensorflow/layers/date_time_to_unix_timestamp.py) |    [Link](src/kamae/spark/transformers/date_time_to_unix_timestamp.py)    |                     Not yet implemented                     |
|             Divide              |                                             Divides a single feature by a constant or divides multiple features against each other.                                             |           [Link](src/kamae/tensorflow/layers/divide.py)            |              [Link](src/kamae/spark/transformers/divide.py)               |                     Not yet implemented                     |
|               Exp               |                                                                  Applies the exp(x) operation to the feature.                                                                   |             [Link](src/kamae/tensorflow/layers/exp.py)             |                [Link](src/kamae/spark/transformers/exp.py)                |                     Not yet implemented                     |
|            Exponent             |                                                    Applies the x^exponent to a single feature or x^y for multiple features.                                                     |          [Link](src/kamae/tensorflow/layers/exponent.py)           |             [Link](src/kamae/spark/transformers/exponent.py)              |                     Not yet implemented                     |
|            HashIndex            |                                                     Transforms strings to indices via a hash table of predeterminded size.                                                      |         [Link](src/kamae/tensorflow/layers/hash_index.py)          |            [Link](src/kamae/spark/transformers/hash_index.py)             |                     Not yet implemented                     |
|        HaversineDistance        |                            Computes the [haversine distance](https://en.wikipedia.org/wiki/Haversine_formula) between latitude and longitude pairs.                             |     [Link](src/kamae/tensorflow/layers/haversine_distance.py)      |        [Link](src/kamae/spark/transformers/haversine_distance.py)         |                     Not yet implemented                     |
|            Identity             |                                                           Applies the identity operation, leaving the input the same.                                                           |          [Link](src/kamae/tensorflow/layers/identity.py)           |             [Link](src/kamae/spark/transformers/identity.py)              |     [Link](src/kamae/sklearn/transformers/identity.py)      |
|           IfStatement           |                                                  Computes a simple if statement on a set of columns/tensors and/or constants.                                                   |        [Link](src/kamae/tensorflow/layers/if_statement.py)         |           [Link](src/kamae/spark/transformers/if_statement.py)            |                     Not yet implemented                     |
|             Impute              |                                              Performs imputation of either mean or median value of the data over a specified mask.                                              |           [Link](src/kamae/tensorflow/layers/impute.py)            |              [Link](src/kamae/spark/transformers/impute.py)               |                     Not yet implemented                     |
|         LambdaFunction          |                              Transforms an input (or multiple inputs) to an output (or multiple outputs) with a user provided tensorflow function.                              |       [Link](src/kamae/tensorflow/layers/lambda_function.py)       |          [Link](src/kamae/spark/transformers/lambda_function.py)          |                     Not yet implemented                     |
|             ListMax             |                               Computes the listwise max of a feature, optionally calculated only on the top items based on another given feature.                               |          [Link](src/kamae/tensorflow/layers/list_max.py)           |             [Link](src/kamae/spark/transformers/list_max.py)              |                     Not yet implemented                     |
|            ListMean             |                              Computes the listwise mean of a feature, optionally calculated only on the top items based on another given feature.                               |          [Link](src/kamae/tensorflow/layers/list_mean.py)          |             [Link](src/kamae/spark/transformers/list_mean.py)             |                     Not yet implemented                     |
|           ListMedian            |                             Computes the listwise median of a feature, optionally calculated only on the top items based on another given feature.                              |         [Link](src/kamae/tensorflow/layers/list_median.py)         |            [Link](src/kamae/spark/transformers/list_median.py)            |                     Not yet implemented                     |
|             ListMin             |                               Computes the listwise min of a feature, optionally calculated only on the top items based on another given feature.                               |          [Link](src/kamae/tensorflow/layers/list_min.py)           |             [Link](src/kamae/spark/transformers/list_min.py)              |                     Not yet implemented                     |
|            ListRank             |                                                               Computes the listwise rank (ordering) of a feature.                                                               |          [Link](src/kamae/tensorflow/layers/list_rank.py)          |             [Link](src/kamae/spark/transformers/list_rank.py)             |                     Not yet implemented                     |
|           ListStdDev            |                       Computes the listwise standard deviation of a feature, optionally calculated only on the top items based on another given feature.                        |        [Link](src/kamae/tensorflow/layers/list_std_dev.py)         |           [Link](src/kamae/spark/transformers/list_std_dev.py)            |                     Not yet implemented                     |
|               Log               |                                                           Applies the natural logarithm `log(alpha + x)` transform  .                                                           |             [Link](src/kamae/tensorflow/layers/log.py)             |                [Link](src/kamae/spark/transformers/log.py)                |        [Link](src/kamae/sklearn/transformers/log.py)        |
|           LogicalAnd            |                                                          Performs an and(x, y) operation on multiple boolean features.                                                          |         [Link](src/kamae/tensorflow/layers/logical_and.py)         |            [Link](src/kamae/spark/transformers/logical_and.py)            |                     Not yet implemented                     |
|           LogicalNot            |                                                            Performs a not(x) operation on a single boolean feature.                                                             |         [Link](src/kamae/tensorflow/layers/logical_not.py)         |            [Link](src/kamae/spark/transformers/logical_not.py)            |                     Not yet implemented                     |
|            LogicalOr            |                                                          Performs an or(x, y) operation on multiple boolean features.                                                           |         [Link](src/kamae/tensorflow/layers/logical_or.py)          |            [Link](src/kamae/spark/transformers/logical_or.py)             |                     Not yet implemented                     |
|               Max               |                                                  Computes the maximum of a feature with a constant or multiple other features.                                                  |             [Link](src/kamae/tensorflow/layers/max.py)             |                [Link](src/kamae/spark/transformers/max.py)                |                     Not yet implemented                     |
|              Mean               |                                                   Computes the mean of a feature with a constant or multiple other features.                                                    |            [Link](src/kamae/tensorflow/layers/mean.py)             |               [Link](src/kamae/spark/transformers/mean.py)                |                     Not yet implemented                     |
|               Min               |                                                  Computes the minimum of a feature with a constant or multiple other features.                                                  |             [Link](src/kamae/tensorflow/layers/min.py)             |                [Link](src/kamae/spark/transformers/min.py)                |                     Not yet implemented                     |
|          MinHashIndex           |                            Creates an integer bit array from a set of strings using the [MinHash algorithm](https://en.wikipedia.org/wiki/MinHash).                             |       [Link](src/kamae/tensorflow/layers/min_hash_index.py)        |          [Link](src/kamae/spark/transformers/min_hash_index.py)           |                     Not yet implemented                     |
|           MinMaxScale           |                                                    Scales the input feature by the min/max resulting in a feature in [0, 1].                                                    |        [Link](src/kamae/tensorflow/layers/min_max_scale.py)        |           [Link](src/kamae/spark/transformers/min_max_scale.py)           |                     Not yet implemented                     |
|             Modulo              |                                           Computes the modulo of a feature with the mod divisor being a constant or another feature.                                            |           [Link](src/kamae/tensorflow/layers/modulo.py)            |              [Link](src/kamae/spark/transformers/modulo.py)               |                     Not yet implemented                     |
|            Multiply             |                                               Multiplies a single feature by a constant or multiples multiple features together.                                                |          [Link](src/kamae/tensorflow/layers/multiply.py)           |             [Link](src/kamae/spark/transformers/multiply.py)              |                     Not yet implemented                     |
|      NumericalIfStatement       |                         Performs a simple if else statement witha given operator. Value to check, result if true or false can be constants or features.                         |   [Link](src/kamae/tensorflow/layers/numerical_if_statement.py)    |      [Link](src/kamae/spark/transformers/numerical_if_statement.py)       |                     Not yet implemented                     |
|          OneHotEncode           |                                                                     Transforms a string to a one-hot array.                                                                     |       [Link](src/kamae/tensorflow/layers/one_hot_encode.py)        |           [Link](src/kamae/spark/estimators/one_hot_encode.py)            |                     Not yet implemented                     |
|       OrdinalArrayEncode        |                                          Encodes strings in an array according to the order in which they appear. Only for 2D tensors.                                          |    [Link](src/kamae/tensorflow/layers/ordinal_array_encoder.py)    |        [Link](src/kamae/spark/estimators/ordinal_array_encoder.py)        |                     Not yet implemented                     |
|              Round              |                                        Rounds a floating feature to the nearest integer using `ceil`, `floor` or a standard `round` op.                                         |            [Link](src/kamae/tensorflow/layers/round.py)            |               [Link](src/kamae/spark/transformers/round.py)               |                     Not yet implemented                     |
|         RoundToDecimal          |                                                           Rounds a floating feature to the nearest decimal precision.                                                           |      [Link](src/kamae/tensorflow/layers/round_to_decimal.py)       |         [Link](src/kamae/spark/transformers/round_to_decimal.py)          |                     Not yet implemented                     |
|       SharedOneHotEncode        |                                   Transforms a string to a one-hot array, using labels across multiple inputs to determine the one-hot size.                                    |       [Link](src/kamae/tensorflow/layers/one_hot_encode.py)        |        [Link](src/kamae/spark/estimators/shared_one_hot_encode.py)        |                     Not yet implemented                     |
|        SharedStringIndex        |                                      Transforms strings to indices via a vocabulary lookup, sharing the vocabulary across multiple inputs.                                      |        [Link](src/kamae/tensorflow/layers/string_index.py)         |         [Link](src/kamae/spark/estimators/shared_string_index.py)         |                     Not yet implemented                     |
| SingleFeatureArrayStandardScale |                        Normalises by the mean and standard deviation calculated over all elements of all inputs, with ability to mask a specified value.                        |       [Link](src/kamae/tensorflow/layers/standard_scale.py)        | [Link](src/kamae/spark/estimators/single_feature_array_standard_scale.py) |                     Not yet implemented                     |
|          StandardScale          |                                             Normalises by the mean and standard deviation, with ability to mask a specified value.                                              |       [Link](src/kamae/tensorflow/layers/standard_scale.py)        |           [Link](src/kamae/spark/estimators/standard_scale.py)            |   [Link](src/kamae/sklearn/estimators/standard_scale.py)    |
|           StringAffix           |                                                             Prefixes and suffixes a string with provided constants.                                                             |        [Link](src/kamae/tensorflow/layers/string_affix.py)         |           [Link](src/kamae/spark/transformers/string_affix.py)            |                     Not yet implemented                     |
|       StringArrayConstant       |                                                              Inserts provided string array constant into a column.                                                              |    [Link](src/kamae/tensorflow/layers/string_array_constant.py)    |       [Link](src/kamae/spark/transformers/string_array_constant.py)       |                     Not yet implemented                     |
|           StringCase            |                                                           Applies an upper or lower casing operation to the feature.                                                            |         [Link](src/kamae/tensorflow/layers/string_case.py)         |            [Link](src/kamae/spark/transformers/string_case.py)            |                     Not yet implemented                     |
|        StringConcatenate        |                                                               Joins string columns using the provided separator.                                                                |     [Link](src/kamae/tensorflow/layers/string_concatenate.py)      |        [Link](src/kamae/spark/transformers/string_concatenate.py)         |                     Not yet implemented                     |
|         StringContains          |                                              Checks for the existence of a constant or tensor-element substring within a feature.                                               |       [Link](src/kamae/tensorflow/layers/string_contains.py)       |          [Link](src/kamae/spark/transformers/string_contains.py)          |                     Not yet implemented                     |
|       StringContainsList        |                                            Checks for the existence of any string from a list of string constants within a feature.                                             |    [Link](src/kamae/tensorflow/layers/string_contains_list.py)     |       [Link](src/kamae/spark/transformers/string_contains_list.py)        |                     Not yet implemented                     |
|     StringEqualsIfStatement     |                          Performs a simple if else statement on string equality. Value to check, result if true or false can be constants or features.                          | [Link](src/kamae/tensorflow/layers/string_equals_if_statement.py)  |    [Link](src/kamae/spark/transformers/string_equals_if_statement.py)     |                     Not yet implemented                     |
|           StringIndex           |                                                              Transforms strings to indices via a vocabulary lookup                                                              |        [Link](src/kamae/tensorflow/layers/string_index.py)         |            [Link](src/kamae/spark/estimators/string_index.py)             |                     Not yet implemented                     |
|       StringListToString        |                                                    Concatenates a list of strings to a single string with a given delimiter.                                                    |    [Link](src/kamae/tensorflow/layers/string_list_to_string.py)    |       [Link](src/kamae/spark/transformers/string_list_to_string.py)       |                     Not yet implemented                     |
|            StringMap            |                    Maps a list of string values to a list of other string values with a standard CASE WHEN statement. Can provide a default value for ELSE.                     |         [Link](src/kamae/tensorflow/layers/string_map.py)          |            [Link](src/kamae/spark/transformers/string_map.py)             |                     Not yet implemented                     |
|         StringIsInList          |                                                     Checks if the feature is equal to at least one of the strings provided.                                                     |      [Link](src/kamae/tensorflow/layers/string_isin_list.py)       |         [Link](src/kamae/spark/transformers/string_isin_list.py)          |                     Not yet implemented                     |
|          StringReplace          |                                        Performs a regex replace operation on a feature with constant params or between multiple features                                        |       [Link](src/kamae/tensorflow/layers/string_replace.py)        |          [Link](src/kamae/spark/transformers/string_replace.py)           |                     Not yet implemented                     |
|       StringToStringList        |                               Splits a string by a separator, returning a list of parametrised length (with a default value for missing inputs).                                |    [Link](src/kamae/tensorflow/layers/string_to_string_list.py)    |       [Link](src/kamae/spark/transformers/string_to_string_list.py)       |                     Not yet implemented                     |
|      SubStringDelimAtIndex      |           Splits a string column using the provided delimiter, and returns the value at the index given. If the index is out of bounds, returns a given default value           |  [Link](src/kamae/tensorflow/layers/sub_string_delim_at_index.py)  |     [Link](src/kamae/spark/transformers/sub_string_delim_at_index.py)     |                     Not yet implemented                     |
|            Subtract             |                                           Subtracts a constant from a single feature or subtracts multiple features from each other.                                            |          [Link](src/kamae/tensorflow/layers/subtract.py)           |             [Link](src/kamae/spark/transformers/subtract.py)              |                     Not yet implemented                     |
|               Sum               |                                                     Adds a constant to a single feature or sums multiple features together.                                                     |             [Link](src/kamae/tensorflow/layers/sum.py)             |                [Link](src/kamae/spark/transformers/sum.py)                |                     Not yet implemented                     |
|     UnixTimestampToDateTime     |                                                               Converts a unix timestamp to a UTC datetime string.                                                               | [Link](src/kamae/tensorflow/layers/unix_timestamp_to_date_time.py) |    [Link](src/kamae/spark/transformers/unix_timestamp_to_date_time.py)    |                     Not yet implemented                     |

## Development

### Setup

Requirements: Python 3.10 (for development), `pipx` ([installation instructions](https://github.com/pypa/pipx?tab=readme-ov-file#install-pipx))

```bash
make setup      # Install dependencies and pre-commit hooks
make all        # Run tests, formatting, and linting
make help       # See all available commands
```

The package supports Python 3.8-3.12 in production.

### Common Commands

```bash
make run-example        # Run example pipeline
make test-tf-serving    # Test TensorFlow Serving inference
make test-end-to-end    # Run example + test serving
```

### Contributing

Create a branch from `main` and open a pull request. Follow the [adding transformers guide](docs/adding_transformer.md) for new transformers.

**Code quality**: Pre-commit hooks enforce formatting and linting. Install with `uv run pre-commit install`. PRs must pass all tests in `tests/`.

**Versioning**: Automated via [semantic-release](https://semantic-release.gitbook.io/semantic-release/). Use conventional commit prefixes in PR titles: `fix:` (patch), `feat:` (minor), `BREAKING CHANGE:` (major).

**Contact**: Questions? Reach out to the [Kamae team](https://github.com/orgs/ExpediaGroup/teams/kamae-admins).
