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

from abc import ABC, abstractmethod
from random import choice
from string import ascii_uppercase
from typing import List, Optional, Tuple

import pyspark.sql.functions as F
from pyspark import keyword_only
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import DataType, NumericType

from kamae.spark.params import (
    HasInputDtype,
    HasLayerName,
    HasOutputDtype,
    InputOutputExtractor,
)
from kamae.spark.utils import (
    get_element_type,
    single_input_single_output_scalar_transform,
)


class SparkOperation(
    ABC, HasLayerName, HasInputDtype, HasOutputDtype, InputOutputExtractor
):
    """
    Abstract class used in Spark transformers and estimators. Provides common utils for
    param setting, input/output dtype casting, and layer name setting.
    """

    def __init__(self):
        """
        Initializes the spark operation class.
        """
        super().__init__()
        self._setDefault(layerName=self.uid, inputDtype=None, outputDtype=None)
        self.tmp_column_suffix = self.generate_tmp_column_suffix()

    @property
    @abstractmethod
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the spark operation.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the spark operation.
        """
        raise NotImplementedError

    @keyword_only
    def setParams(self, **kwargs) -> "SparkOperation":
        """
        Sets all given keyword parameters.

        :returns: Class instance.
        """
        input_kwargs = self._input_kwargs
        if (
            "inputCol" in input_kwargs
            and "inputCols" in input_kwargs
            and input_kwargs["inputCol"] is not None
            and input_kwargs["inputCols"] is not None
        ):
            raise ValueError("Only one of inputCol or inputCols can be set, not both.")
        for param_name, param_value in input_kwargs.items():
            # Only if the param value is not None, do we set the param
            if param_value is not None:
                # Get the setter method for the parameter
                setter_method_name = f"set{param_name[0].upper()}{param_name[1:]}"
                setter_method = getattr(self, setter_method_name)
                # Set the parameter
                setter_method(param_value)
        return self

    @staticmethod
    def generate_tmp_column_suffix(str_len: int = 25):
        """
        Returns a random string of length `str_len` to append to temporary columns.

        This ensures that there is minimal collision between temporary columns created
        and the original columns.

        :param str_len: Length of the random string.
        :returns: Random string of length `str_len`.
        """
        return "".join(choice(ascii_uppercase) for _ in range(str_len))

    @staticmethod
    def _resolve_tmp_from_true_column_name(
        column_name: str, suffix: str, reverse: bool = False
    ):
        """
        Resolves the temporary column name from the true column name or vice versa.

        Used to rename input columns when casting is necessary to avoid overwriting
        the original columns.

        :param column_name: The current column name.
        :param suffix: The suffix to append to the column name.
        :param reverse: Whether to resolve the temporary column name from the true
        column name or vice versa.
        """
        return (
            f"{column_name}_{suffix}"
            if not reverse
            else column_name.replace(f"_{suffix}", "")
        )

    @staticmethod
    def _cast(column: Column, column_dtype: DataType, cast_dtype: str) -> Column:
        """
        Casts the input column to the specified datatype.

        :param column: Pyspark column to cast.
        :param column_dtype: The datatype class of the column.
        :param cast_dtype: The datatype string to cast the column to.
        :returns: Pyspark column cast to the specified datatype.
        """
        # There is an edge case where we can have a negatively signed 0.
        # This will not match tensorflow, so we will ensure the sign of zeros is
        # always positive.
        if isinstance(column_dtype, NumericType):
            # We need to cast back to the original type after multiplying by the sign
            # since the multiplication by sign will return a double. Therefore, we can
            # have cases where the original type is an int, it is cast to a double, and
            # then we want a string. So we have 1 -> 1.0 -> "1.0" rather than "1".
            # TODO: I really don't like this, there must be a better way to do this.
            func = lambda x: F.when(  # noqa: E731
                x == F.lit(0),
                (x * F.signum(x)).cast(column_dtype.simpleString()).cast(cast_dtype),
            ).otherwise(x.cast(cast_dtype))
        else:
            func = lambda x: x.cast(cast_dtype)  # noqa: E731
        return single_input_single_output_scalar_transform(
            input_col=column, input_col_datatype=column_dtype, func=func
        )

    def _cast_input_output_columns(
        self, columns: List[Column], column_datatypes: List[DataType], ingress: bool
    ) -> List[Tuple[Column, bool]]:
        """
        Casts either the input and output columns to the given inputDtype or
        outputDtype, if specified. Ingress is a boolean that indicates whether we are
        casting the input (True) or output (False) columns.

        :param columns: List of input/output columns to cast.
        :param column_datatypes: List of input/output column datatypes.
        :param ingress: Boolean indicating whether we are casting the input (True) or
        output (False) columns.
        :returns: List of tuple; first element is input/output columns cast to the
        inputDtype/outputDtype, second element is a boolean indicating whether the
        column was cast.
        """
        if ingress:
            cast_dtype = self.getInputDtype()
            if (
                cast_dtype is not None
                and self.compatible_dtypes is not None
                and cast_dtype
                not in [dtype.simpleString() for dtype in self.compatible_dtypes]
            ):
                raise ValueError(
                    f"""inputDtype {cast_dtype} is not a compatible dtype for
                    transformer with uid {self.uid}.
                    Compatible dtypes are{[
                        dtype.simpleString() for dtype in self.compatible_dtypes
                    ]}."""
                )
        else:
            cast_dtype = self.getOutputDtype()

        if cast_dtype is not None:
            return [
                (
                    self._cast(
                        column=column, column_dtype=column_dtype, cast_dtype=cast_dtype
                    ),
                    True,
                )
                if get_element_type(column_dtype).simpleString() != cast_dtype
                else (
                    column,
                    False,
                )
                for column, column_dtype in zip(columns, column_datatypes)
            ]
        return [(column, False) for column in columns]

    def _cast_input_columns(
        self, input_columns: List[Column], input_column_datatypes: List[DataType]
    ) -> List[Tuple[Column, bool]]:
        """
        Casts the input columns to the given inputDtype, if specified. All columns are
        cast to this. This might not be ideal, there may be layers where some inputs are
        expected to be different types. In these cases, the subclass should
        implement the cast_input_columns method.

        :param input_columns: List of input columns to cast.
        :param input_column_datatypes: List of input column datatypes.
        :returns: List of tuple; first element is input columns cast to the inputDtype,
        second element is a boolean indicating whether the column was cast.
        """
        return self._cast_input_output_columns(
            columns=input_columns, column_datatypes=input_column_datatypes, ingress=True
        )

    def _cast_output_columns(
        self, output_columns: List[Column], output_column_datatypes: List[DataType]
    ) -> List[Tuple[Column, bool]]:
        """
        Casts the output columns to the given outputDtype, if specified. All columns are
        cast to this. This might not be ideal, there may be layers where some outputs
        are expected to be different types. In these cases, the subclass should
        implement the cast_output_columns method.

        :param output_columns: List of output columns to cast.
        :param output_column_datatypes: List of output column datatypes.
        :returns: List of tuple; first element is output columns cast to the
        outputDtype, second element is a boolean indicating whether the column was cast.
        """
        return self._cast_input_output_columns(
            columns=output_columns,
            column_datatypes=output_column_datatypes,
            ingress=False,
        )

    def _create_casted_input_output_columns(
        self, dataset: DataFrame, ingress: bool
    ) -> DataFrame:
        """
        Recreates the input or output columns, creating temporary columns with the
        casted input columns if necessary. This is done to avoid overwriting the
        original columns and creating inconsistencies if multiple transforms use the
        same inputs.

        :param dataset: The input dataset.
        :param ingress: Whether the input columns are being cast or the output columns.
        :returns: The dataset with the cast input/output columns.
        """
        col_names = self._get_single_or_multi_col(ingress=ingress)
        columns = [F.col(c) for c in col_names]
        col_datatypes = [dataset.schema[column].dataType for column in col_names]
        casted_cols = (
            self._cast_input_columns(columns, col_datatypes)
            if ingress
            else self._cast_output_columns(columns, col_datatypes)
        )
        for column_name, column_tuple in zip(col_names, casted_cols):
            column = column_tuple[0]
            column_is_cast = column_tuple[1]
            if ingress:
                if column_is_cast:
                    # If we are casting inputs, we create new columns temporarily
                    # to avoid overwriting the original columns and creating
                    # inconsistencies if multiple transforms use the same inputs
                    tmp_input_name = self._resolve_tmp_from_true_column_name(
                        column_name, suffix=self.tmp_column_suffix
                    )
                    dataset = dataset.withColumn(tmp_input_name, column)
            else:
                # Output casting can just be done directly, since this is already a new
                # column created by the transform itself
                if column_is_cast:
                    dataset = dataset.withColumn(column_name, column)
        return dataset

    def drop_tmp_casted_input_columns(self, dataset: DataFrame) -> DataFrame:
        """
        Drops the temporary columns from the dataset that are created when casting the
        input columns.

        :param dataset: The input dataset.
        :returns: The dataset with the temporary columns dropped.
        """
        for column_name in self._get_single_or_multi_col(ingress=True):
            tmp_input_name = self._resolve_tmp_from_true_column_name(
                column_name, suffix=self.tmp_column_suffix
            )
            if tmp_input_name in dataset.columns:
                dataset = dataset.drop(tmp_input_name)
        return dataset

    def set_input_columns_to_from_casted(
        self, dataset: DataFrame, suffix: str, reverse: bool = False
    ) -> None:
        """
        Sets the input columns to the temporary casted columns or back if reverse is
        True.

        :param dataset: The input dataset. Used solely to understand if a tmp column
        has been created. These are only created if the column needed to be cast.
        :param suffix: The suffix to append to the column name.
        :param reverse: Whether to set the input columns back to the original columns.
        :returns: None
        """
        col_names = self._get_single_or_multi_col(ingress=True)
        renamed_columns = [
            self._resolve_tmp_from_true_column_name(column_name, suffix, reverse)
            for column_name in col_names
        ]
        # Check if renaming is necessary. If the renamed column is not present in the
        # dataset, we should not set the input columns to the renamed columns.
        new_col_names = []
        for original_column_name, renamed_column_name in zip(
            col_names, renamed_columns
        ):
            if renamed_column_name in dataset.columns:
                new_col_names.append(renamed_column_name)
            else:
                new_col_names.append(original_column_name)

        if self.hasParam("inputCols") and self.isDefined("inputCols"):
            self._set(inputCols=new_col_names)
        elif self.hasParam("inputCol") and self.isDefined("inputCol"):
            self._set(inputCol=new_col_names[0])
        else:
            raise ValueError("No input columns to set")

    def _check_input_dtypes_compatible(
        self, dataset: DataFrame, column_names: List[str]
    ) -> None:
        """
        Checks if the input tensors are compatible with the compatible_dtypes of the
        layer.

        :param dataset: The input dataset.
        :param column_names: The names of the input columns.
        :raises ValueError: If the input columns are not compatible with the
        compatible_dtypes of the transformer.
        :returns: None
        """
        for c in column_names:
            tmp_column_name = self._resolve_tmp_from_true_column_name(
                c, suffix=self.tmp_column_suffix
            )
            # Either the tmp column name or the original is present.
            check_column_name = (
                tmp_column_name if tmp_column_name in dataset.columns else c
            )
            col_dtype = self.get_column_datatype(dataset, check_column_name)
            underlying_col_dtype = get_element_type(col_dtype)
            if (
                self.compatible_dtypes is not None
                and underlying_col_dtype not in self.compatible_dtypes
            ):
                raise TypeError(
                    f"""Input column with name {c} and dtype
                    {underlying_col_dtype.simpleString()} is not a compatible dtype for
                    transformer with uid {self.uid}.
                    Compatible dtypes are {[
                        dtype.simpleString() for dtype in self.compatible_dtypes
                    ]}."""
                )
