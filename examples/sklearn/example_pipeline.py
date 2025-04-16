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

import joblib
import pandas as pd

from kamae.sklearn.estimators import StandardScaleEstimator
from kamae.sklearn.pipeline import KamaeSklearnPipeline
from kamae.sklearn.transformers import (
    ArrayConcatenateTransformer,
    ArraySplitTransformer,
    IdentityTransformer,
    LogTransformer,
)

if __name__ == "__main__":
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

    # Create some dummy pandas data
    df = pd.DataFrame(
        {
            "col1": [10, 4.8, 7.3],
            "col2": [2.5, 5.3, 8.2],
            "col3": [3.7, 6.4, 9.4],
            "col4": [[1.6, 4.0, 7.0], [2.4, 5.5, 8.1], [3.1, 6.4, 9.1]],
        },
    )
    print("Original dataframe:")
    print(df.head())

    # Create a scikit-learn pipeline
    log_transformer = LogTransformer(
        input_col="col1",
        output_col="log_col1",
        alpha=1,
        layer_name="log_one_plus_x",
    )
    identity_transformer = IdentityTransformer(
        input_col="col3",
        output_col="identity_col3",
        layer_name="identity_col3_output",
    )
    vector_assembler = ArrayConcatenateTransformer(
        input_cols=["log_col1", "col2", "identity_col3", "col4"],
        output_col="vec_assembled",
        layer_name="vector_assembler",
    )
    standard_scaler = StandardScaleEstimator(
        input_col="vec_assembled",
        output_col="scaled_assembled_vec",
        layer_name="standard_scaler",
    )
    vector_slicer = ArraySplitTransformer(
        input_col="scaled_assembled_vec",
        output_cols=[
            "sliced_col1",
            "sliced_col2",
            "sliced_col3",
            "sliced_col4_1",
            "sliced_col4_2",
            "sliced_col4_3",
        ],
        layer_name="vector_slicer",
    )
    test_pipeline = KamaeSklearnPipeline(
        steps=[
            ("identity_transformer", identity_transformer),
            ("log_transformer", log_transformer),
            ("vec_assembler", vector_assembler),
            ("standard_scaler", standard_scaler),
            ("vector_slicer", vector_slicer),
        ]
    )

    # Fit the pipeline
    test_pipeline.fit(df)
    # Transform the pipeline
    transformed_df = test_pipeline.transform(df)

    print("Transformed dataframe:")
    print(transformed_df.head())

    print("Saving pipeline using joblib...")
    joblib.dump(test_pipeline, "./output/test_sklearn_pipeline.joblib")

    print("Loading pipeline using joblib...")
    loaded_pipeline = joblib.load("./output/test_sklearn_pipeline.joblib")

    print("Transforming dataframe using loaded pipeline...")
    loaded_transformed_df = loaded_pipeline.transform(df)
    print(loaded_transformed_df.head())

    # Get keras model
    tf_input_schema = [
        {
            "name": "col1",
            "dtype": "float32",
            "shape": (None, 1),
        },
        {
            "name": "col2",
            "dtype": "float32",
            "shape": (None, 1),
        },
        {
            "name": "col3",
            "dtype": "float32",
            "shape": (None, 1),
        },
        {
            "name": "col4",
            "dtype": "float32",
            "shape": (None, 3),
        },
    ]
    print("Building keras model...")
    keras_model = loaded_pipeline.build_keras_model(tf_input_schema=tf_input_schema)
    print(keras_model.summary())
