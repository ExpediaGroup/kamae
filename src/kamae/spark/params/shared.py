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

from typing import List, Union

from pyspark.ml.param import Param, Params, TypeConverters


class StringConstantParams(Params):
    """
    Mixin class for a string constant.
    """

    stringConstant = Param(
        Params._dummy(),
        "stringConstant",
        "String constant to use in many string transformers",
        typeConverter=TypeConverters.toString,
    )

    def setStringConstant(self, value: str) -> "StringConstantParams":
        """
        Sets the stringConstant parameter.

        :param value: String constant value to use in different string transformers.
        :returns: Instance of class mixed in.
        """
        return self._set(stringConstant=value)

    def getStringConstant(self) -> str:
        """
        Gets the stringConstant parameter.

        :returns: String constant value to use in different string transformers.
        """
        return self.getOrDefault(self.stringConstant)


class NegationParams(Params):
    """
    Mixin class containing negation parameter needed for transforms that output a
    boolean.
    """

    negation = Param(
        Params._dummy(),
        "negation",
        "Whether to negate the operation.",
        typeConverter=TypeConverters.toBoolean,
    )

    def setNegation(self, value: bool) -> "NegationParams":
        """
        Sets the negation parameter.

        :param value: Bool value of whether to negate the operation.
        :returns: Instance of class mixed in.
        """
        return self._set(negation=value)

    def getNegation(self) -> bool:
        """
        Gets the negation parameter.

        :returns: Bool value of whether to negate the operation.
        """
        return self.getOrDefault(self.negation)


class MathFloatConstantParams(Params):
    """
    Mixin class for a math float constant.
    """

    mathFloatConstant = Param(
        Params._dummy(),
        "mathFloatConstant",
        "Float constant used for math operations",
        typeConverter=TypeConverters.toFloat,
    )

    def __init__(self) -> None:
        super().__init__()
        self._setDefault(layerName=self.uid)

    def getMathFloatConstant(self) -> float:
        """
        Gets the value of the mathFloatConstant parameter.

        :returns: Float constant used for math operations.
        """
        return self.getOrDefault(self.mathFloatConstant)

    def setMathFloatConstant(self, value: float) -> "MathFloatConstantParams":
        """
        Sets the value of the mathFloatConstant parameter.

        :param value: Float constant used for math operations.
        :returns: Class instance.
        """
        return self._set(mathFloatConstant=value)


class StringRegexParams(Params):
    """
    Mixin class for string transformers that use regex.
    """

    regex = Param(
        Params._dummy(),
        "regex",
        "Whether to use regex in the string contains operation.",
        typeConverter=TypeConverters.toBoolean,
    )

    def setRegex(self, value: bool) -> "StringRegexParams":
        """
        Sets the regex parameter.

        :param value: Bool value of whether to use regex in the string contains
        operation.
        :returns: Instance of class mixed in.
        """
        return self._set(regex=value)

    def getRegex(self) -> bool:
        """
        Gets the regex parameter.

        :returns: Bool value of whether to use regex in the string contains
         operation.
        """
        return self.getOrDefault(self.regex)


class ConstantStringArrayParams(Params):
    """
    Mixin class containing separator parameter needed for constant string array
    transforms.
    """

    constantStringArray = Param(
        Params._dummy(),
        "constantStringArray",
        "Value to fill the column with.",
        typeConverter=TypeConverters.toListString,
    )

    def setConstantStringArray(self, value: List[str]) -> "ConstantStringArrayParams":
        """
        Sets the constantStringArray parameter.

        :param value: List of strings to use as a constant string array.
        :returns: Instance of class mixed in.
        """
        return self._set(constantStringArray=value)

    def getConstantStringArray(self) -> List[str]:
        """
        Gets the constantStringArray parameter.

        :returns: List of strings to use as a constant string array.
        """
        return self.getOrDefault(self.constantStringArray)


class LabelsArrayParams(Params):
    labelsArray = Param(
        Params._dummy(),
        "labelsArray",
        "Ordered list of labels to use for the indexer",
        typeConverter=TypeConverters.toListString,
    )

    def setLabelsArray(self, value: List[str]) -> "LabelsArrayParams":
        """
        Sets the labelArray parameter.

        :param value: List of strings to use in indexing transformers.
        :returns: Instance of class mixed in.
        """
        return self._set(labelsArray=value)

    def getLabelsArray(self) -> List[str]:
        """
        Gets the labelArray parameter.

        :returns: List of strings to use in indexing transformers.
        """
        return self.getOrDefault(self.labelsArray)


class DropUnseenParams(Params):
    """
    Mixin class containing parameters needed to drop unseen index in the
    one hot encoder layer.
    """

    dropUnseen = Param(
        Params._dummy(),
        "dropUnseen",
        "Whether to the drop unseen label index in the one hot encoder layer",
        typeConverter=TypeConverters.toBoolean,
    )

    def setDropUnseen(self, value: bool) -> "DropUnseenParams":
        """
        Sets the dropUnseen parameter.

        :param value: Bool value of whether to drop unseen label index
        in the one hot encoder layer.
        :returns: Instance of class mixed in.
        """
        return self._set(dropUnseen=value)

    def getDropUnseen(self) -> bool:
        """
        Gets the dropUnseen parameter.

        :returns: Bool value of whether to drop unseen label index
        in the one hot encoder layer.
        """
        return self.getOrDefault(self.dropUnseen)


class StringIndexParams(LabelsArrayParams):
    """
    Mixin class containing parameters needed for string indexer and one hot encoder
    layers.
    """

    stringOrderType = Param(
        Params._dummy(),
        "stringOrderType",
        """How to order the strings. Options are
        'frequencyAsc', 'frequencyDesc', 'alphabeticalAsc', 'alphabeticalDesc'.""",
        typeConverter=TypeConverters.toString,
    )

    maskToken = Param(
        Params._dummy(),
        "maskToken",
        "Mask token to use for string indexing",
        typeConverter=TypeConverters.toString,
    )

    numOOVIndices = Param(
        Params._dummy(),
        "numOOVIndices",
        "Number of out of vocabulary indices to use",
        typeConverter=TypeConverters.toInt,
    )

    maxNumLabels = Param(
        Params._dummy(),
        "maxNumLabels",
        "Max number of labels to use",
        typeConverter=TypeConverters.toInt,
    )

    def setStringOrderType(self, value: str) -> "StringIndexParams":
        """
        Sets the stringOrderType parameter to the given value.
        Must be one of:
        - 'frequencyAsc'
        - 'frequencyDesc'
        - 'alphabeticalAsc'
        - 'alphabeticalDesc'.

        :param value: String to set the stringOrderType parameter to.
        :returns: Instance of class mixed in.
        """
        possible_order_options = [
            "frequencyAsc",
            "frequencyDesc",
            "alphabeticalAsc",
            "alphabeticalDesc",
        ]
        if value not in possible_order_options:
            raise ValueError(
                f"stringOrderType must be one of {', '.join(possible_order_options)}"
            )
        return self._set(stringOrderType=value)

    def getStringOrderType(self) -> str:
        """
        Gets the stringOrderType parameter.

        :returns: String value of the stringOrderType parameter.
        """
        return self.getOrDefault(self.stringOrderType)

    def setMaskToken(self, value: str) -> "StringIndexParams":
        """
        Sets the maskToken parameter.

        :param value: String value for the mask token to use for string indexing.
        :returns: Instance of class mixed in.
        """
        return self._set(maskToken=value)

    def getMaskToken(self) -> str:
        """
        Gets the maskToken parameter.

        :returns: String value for the mask token to use for string indexing.
        """
        return self.getOrDefault(self.maskToken)

    def setNumOOVIndices(self, value: int) -> "StringIndexParams":
        """
        Sets the numOOVIndices parameter.

        :param value: Int value for the number of OOV indices to use for string
        indexing.
        :returns: Instance of class mixed in.
        """
        if value <= 0:
            raise ValueError("numOOVIndices must be a positive integer")
        return self._set(numOOVIndices=value)

    def getNumOOVIndices(self) -> int:
        """
        Gets the numOOVIndices parameter.

        :returns: Int value for the number of OOV indices to use for string
        indexing.
        """
        return self.getOrDefault(self.numOOVIndices)

    def setMaxNumLabels(self, value: int) -> "StringIndexParams":
        """
        Sets the maxNumLabels parameter.

        :param value: Int value for the max number of labels to use for string
        indexing.
        :returns: Instance of class mixed in.
        """
        if value <= 0:
            raise ValueError("maxNumLabels must be a positive integer")
        return self._set(maxNumLabels=value)

    def getMaxNumLabels(self) -> int:
        """
        Gets the maxNumLabels parameter.

        :returns: Int value for the max number of labels to use for string
        indexing.
        """
        return self.getOrDefault(self.maxNumLabels)


class HashIndexParams(Params):
    """
    Mixin class containing bin parameter needed for hash indexing layers.
    """

    numBins = Param(
        Params._dummy(),
        "numBins",
        "Number of bins to use for hash indexing",
        typeConverter=TypeConverters.toInt,
    )

    maskValue = Param(
        Params._dummy(),
        "maskValue",
        "Mask value to use for hash indexing",
        typeConverter=TypeConverters.toString,
    )

    def setNumBins(self, value: int) -> "HashIndexParams":
        """
        Sets the numBins parameter.

        :param value: Integer value for the number of bins to use for hash indexing.
        :returns: Instance of class mixed in.
        """
        if value <= 0:
            raise ValueError("Number of bins must be greater than 0.")
        return self._set(numBins=value)

    def getNumBins(self) -> int:
        """
        Gets the numBins parameter.

        :returns: Integer value for the number of bins to use for hash indexing.
        """
        return self.getOrDefault(self.numBins)

    def setMaskValue(self, value: str) -> "HashIndexParams":
        """
        Sets the maskValue parameter.

        :param value: String value for the mask value to use for hash indexing.
        :returns: Instance of class mixed in.
        """
        return self._set(maskValue=value)

    def getMaskValue(self) -> str:
        """
        Gets the maskValue parameter.

        :returns: String value for the mask value to use for hash indexing.
        """
        return self.getOrDefault(self.maskValue)


class PadValueParams(Params):
    """
    Mixin class containing pad value parameters needed
    for ordinal array encoder transformers and array crop transformers.
    """

    padValue = Param(
        Params._dummy(),
        "padValue",
        "The value to be considered as padding.",
        typeConverter=TypeConverters.identity,
    )

    def setPadValue(self, value: Union[str, int, float]) -> "PadValueParams":
        """
        Sets the parameter pad value to the given value.
        :param value: pad value.
        :returns: Instance of class mixed in.
        """
        return self._set(padValue=value)

    def getPadValue(self) -> Union[str, int, float]:
        """
        Gets the pad value parameter.
        :returns: string pad value.
        """
        return self.getOrDefault(self.padValue)


class ListwiseStatisticsParams(Params):
    """
    Mixin class containing the parameters needed for Listwise Statistics transformers.
    """

    queryIdCol = Param(
        Params._dummy(),
        "queryIdCol",
        """Column name to aggregate summary statistics upon,
        such as 'search_id'.""",
        typeConverter=TypeConverters.toString,
    )
    topN = Param(
        Params._dummy(),
        "topN",
        "Limit to how far into the list to aggregate.",
        typeConverter=TypeConverters.toInt,
    )
    minFilterValue = Param(
        Params._dummy(),
        "minFilterValue",
        """Value which equal to or greater than will be aggregated
         upon, anything less will be removed - this is primarily to deal
         with padded features.""",
        typeConverter=TypeConverters.toInt,
    )
    sortOrder = Param(
        Params._dummy(),
        "sortOrder",
        """Either leave as blank for default ordering or 'asc'
        for ascending values or 'desc' for descending values.""",
        typeConverter=TypeConverters.toString,
    )

    def setInputCols(self, value: List[str]) -> "ListwiseStatisticsParams":
        """
        Overrides setting the input columns for the transformer.
        Throws an error if we do not have exactly two input columns.
        :returns: Class instance.
        """
        if len(value) != 2:
            raise ValueError(
                """
                Arg inputCols must contain exactly two columns.
                The first is the value column, on which to calculate the statistic.
                The second is the sort column, based on which to sort the items.
                If you don't need sorting, use setInputCol instead.
                """
            )
        if self.getTopN() is None:
            raise ValueError("topN must be specified when using a sort column.")
        return self._set(inputCols=value)

    def setQueryIdCol(self, value: str) -> "ListwiseStatisticsParams":
        """
        Sets the query id parameter.

        :param value: String for column name to aggregate upon.
        :returns: Instance of class mixed in.
        """
        return self._set(queryIdCol=value)

    def getQueryIdCol(self) -> str:
        """
        Gets the query id parameter.

        :returns:  String for column name to aggregate upon.
        """
        return self.getOrDefault(self.queryIdCol)

    def setTopN(self, value: int) -> "ListwiseStatisticsParams":
        """
        Sets the top N parameter.

        :param value: Filter to limit length of input feature. Should be ordered first.
        :returns: Instance of class mixed in.
        """
        return self._set(topN=value)

    def getTopN(self) -> int:
        """
        Gets the top N parameter.

        :returns:  Filter to limit length of input feature. Should be ordered first.
        """
        return self.getOrDefault(self.topN)

    def setMinFilterValue(self, value: int) -> "ListwiseStatisticsParams":
        """
        Sets the min filter value parameter.

        :param value: Minimum value to remove padded values. Defaults to 0.
        :returns: Instance of class mixed in.
        """
        return self._set(minFilterValue=value)

    def getMinFilterValue(self) -> int:
        """
        Gets the min filter value parameter.

        :returns: Minimum value to remove padded values - defaults to >= 0.
        """
        return self.getOrDefault(self.minFilterValue)

    def setDefaultValue(self, value: str) -> "ListwiseStatisticsParams":
        """
        Sets the default value parameter.

        :param value: Int defaults to 0, should a method only have
        padded data this value will be returned.
        :returns: Instance of class mixed in.
        """
        return self._set(defaultValue=value)

    def getDefaultValue(self) -> int:
        """
        Gets the default value parameter.

        :returns: Int defaults to 0, should a method only have
        padded data this value will be returned.
        """
        return self.getOrDefault(self.defaultValue)

    def setSortOrder(self, value: str) -> "ListwiseStatisticsParams":
        """
        Sets the sortOrder parameter to the given value.
        Must be one of 'asc' or 'desc'.

        :param value: String to set the stringOrderType parameter to.
        :returns: Instance of class mixed in.
        """
        possible_order_options = [
            "asc",
            "desc",
        ]
        if value not in possible_order_options:
            raise ValueError(
                f"sortOrderType must be one of {', '.join(possible_order_options)}"
            )
        return self._set(sortOrder=value)

    def getSortOrder(self) -> str:
        """
        Gets the sort order parameter.

        :returns: String to set the stringOrderType parameter to.
        """
        return self.getOrDefault(self.sortOrder)


class MaskValueParams(Params):
    """
    Mixin class containing maskValue parameter needed
    for standard scale and imputation layers.
    This parameter is used to ignore certain values in the scaling process.
    For imputation, the value is ignored by the estimator and imputed over at
    training and inference.
    """

    maskValue = Param(
        Params._dummy(),
        "maskValue",
        """
        Value to be used as a mask by the transformer.
        """,
        typeConverter=TypeConverters.toFloat,
    )

    def setMaskValue(self, value: float) -> "MaskValueParams":
        """
        Sets the maskValue parameter.
        :param value: Float value to use as the mask value.
        :returns: Instance of class mixed in.
        """
        return self._set(maskValue=value)

    def getMaskValue(self) -> float:
        """
        Gets the maskValue parameter.
        :returns: Float value of the mask value.
        """
        return self.getOrDefault(self.maskValue)


class ImputeMethodParams(Params):
    """
    Mixin class containing imputeParam parameter for imputation layer.
    This parameter is used to select the method to estimate the value
    that should be imputed over the mask.
    """

    imputeMethod = Param(
        Params._dummy(),
        "imputeMethod",
        """
        Method with which to compute the value that is imputed over the mask value.
        """,
        typeConverter=TypeConverters.toString,
    )

    def __init__(self) -> None:
        super().__init__()
        self.valid_impute_methods = ["mean", "median"]

    def setImputeMethod(self, value: str) -> "ImputeMethodParams":
        """
        Sets the imputeParam parameter for the imputation layer.
        """
        if value not in self.valid_impute_methods:
            raise ValueError(
                "Imputation method should be one of: [{}], but received {}".format(
                    ",".join(self.valid_impute_methods), value
                )
            )
        return self._set(imputeMethod=value)

    def getImputeMethod(self) -> str:
        """
        Gets the imputeParam parameter value.
        :returns: Str value of the method to use to compute the value to impute.
        """
        return self.getOrDefault(self.imputeMethod)


class NanFillValueParams(Params):
    nanFillValue = Param(
        Params._dummy(),
        "nanFillValue",
        """
        The value to fill Nan with.
        """,
        typeConverter=TypeConverters.toFloat,
    )

    def setNanFillValue(self, value: float) -> "NanFillValueParams":
        """
        Sets the nanFillValue parameter.
        :param value: Float value to use as the fill value.
        :returns: Instance of class mixed in.
        """
        if value is None:
            raise ValueError("nanFillValue cannot be None")
        return self._set(nanFillValue=value)

    def getNanFillValue(self) -> float:
        """
        Gets the nanFillValue parameter.
        :returns: Float value of the fill value.
        """
        return self.getOrDefault(self.nanFillValue)


class StandardScaleSkipZerosParams(Params):
    """
    Mixin class containing maskValue parameter needed for conditional standard scale
    layers. This parameter is used to ignore zeros when scaling.
    """

    skipZeros = Param(
        Params._dummy(),
        "skipZeros",
        """
        If True, during spark transform and keras inference, do not apply the
        scaling when the values to scale are equal to zero.
        """,
        typeConverter=TypeConverters.toBoolean,
    )

    epsilon = Param(
        Params._dummy(),
        "epsilon",
        """
        Epsilon value to use when checking the skipZeros condition.
        """,
        typeConverter=TypeConverters.toFloat,
    )

    def setSkipZeros(self, value: bool) -> "StandardScaleSkipZerosParams":
        """
        Sets the skipZeros parameter.

        :param value: Boolean value to use as the mask value.
        :returns: Instance of class mixed in.
        """
        return self._set(skipZeros=value)

    def getSkipZeros(self) -> bool:
        """
        Gets the skipZeros parameter.

        :returns: Boolean value of the mask value.
        """
        return self.getOrDefault(self.skipZeros)

    def setEpsilon(self, value: float) -> "StandardScaleSkipZerosParams":
        """
        Sets the epsilon parameter.

        :param value: Float value to use as the epsilon value.
        :returns: Instance of class mixed in.
        """
        return self._set(epsilon=value)

    def getEpsilon(self) -> float:
        """
        Gets the epsilon parameter.

        :returns: Float value of the epsilon value.
        """
        return self.getOrDefault(self.epsilon)


class StandardScaleParams(MaskValueParams):
    """
    Mixin class containing mean and standard deviation parameters needed
    for standard scaler layers.
    """

    mean = Param(
        Params._dummy(),
        "mean",
        "Mean of the feature values.",
        typeConverter=TypeConverters.toListFloat,
    )

    stddev = Param(
        Params._dummy(),
        "stddev",
        "Standard deviation of the feature values.",
        typeConverter=TypeConverters.toListFloat,
    )

    def setMean(self, value: List[float]) -> "StandardScaleParams":
        """
        Sets the parameter mean to the given Vector value.
        Saves the mean as a list of floats.

        :param value: List of mean values.
        :returns: Instance of class mixed in.
        """
        if None in set(value):
            ids = [i for i, x in enumerate(value) if x is None]
            raise ValueError("Got null Mean values at positions: ", ids)
        return self._set(mean=value)

    def getMean(self) -> List[float]:
        """
        Gets the mean parameter.

        :returns: List of float mean values.
        """
        return self.getOrDefault(self.mean)

    def setStddev(self, value: List[float]) -> "StandardScaleParams":
        """
        Sets the parameter stddev to the given Vector value.
        Saves the standard deviation as a list of floats.

        :param value: Vector of standard deviation values.
        :returns: Instance of class mixed in.
        """
        if None in set(value):
            ids = [i for i, x in enumerate(value) if x is None]
            raise ValueError("Got null Stddev values at positions: ", ids)
        return self._set(stddev=value)

    def getStddev(self) -> List[float]:
        """
        Gets the stddev parameter.

        :returns: List of float standard deviation values.
        """
        return self.getOrDefault(self.stddev)


class UnixTimestampParams(Params):
    """
    Mixin class for a unix timestamp
    """

    unit = Param(
        Params._dummy(),
        "unit",
        """Unit of the timestamp.
        Can be `milliseconds` or `seconds`. Can also use short-hand `ms` and `s`.
        Default is `s` (seconds)""",
        typeConverter=TypeConverters.toString,
    )

    def setUnit(self, value: str) -> "UnixTimestampParams":
        """
        Sets the unit parameter.
        :param value: unit.
        :returns: Instance of class mixed in.
        """
        allowed_units = ["milliseconds", "seconds", "ms", "s"]
        if value not in allowed_units:
            raise ValueError(f"Unit must be one of {allowed_units}")

        if value == "milliseconds":
            value = "ms"
        if value == "seconds":
            value = "s"

        return self._set(unit=value)

    def getUnit(self) -> str:
        """
        Gets the unit parameter.
        :returns: unit.
        """
        return self.getOrDefault(self.unit)


class DateTimeParams(Params):
    """
    Mixin class for a datetime transformation
    """

    includeTime = Param(
        Params._dummy(),
        "includeTime",
        """Whether to include the time in the output datetime.
        If False, only the date is included in the format yyyy-MM-dd. If True,
        the time is also included in the format yyyy-MM-dd HH:mm:ss.SSS""",
        typeConverter=TypeConverters.toBoolean,
    )

    def setIncludeTime(self, value: bool) -> "DateTimeParams":
        """
        Sets the includeTime parameter.
        :param value: includeTime.
        :returns: Instance of class mixed in.
        """
        return self._set(includeTime=value)

    def getIncludeTime(self) -> bool:
        """
        Gets the includeTime parameter.
        :returns: includeTime.
        """
        return self.getOrDefault(self.includeTime)


class AutoBroadcastParams(Params):
    """
    Mixin class for the auto broadcast parameter.
    """

    autoBroadcast = Param(
        Params._dummy(),
        "autoBroadcast",
        """Whether to enable auto broadcast for the layer.
        If `True`, will broadcast the input tensors to the biggest rank before
        concatenating. Defaults to `False`.
        """,
        typeConverter=TypeConverters.toBoolean,
    )

    def setAutoBroadcast(self, value: bool) -> "AutoBroadcastParams":
        """
        Sets the autoBroadcast parameter.
        :param value: autoBroadcast.
        :returns: Instance of class mixed in.
        """
        return self._set(autoBroadcast=value)

    def getAutoBroadcast(self) -> bool:
        """
        Gets the autoBroadcast parameter.
        :returns: autoBroadcast.
        """
        return self.getOrDefault(self.autoBroadcast)


class LatLonConstantParams(Params):
    """
    Mixin class containing lat and lon constant parameters.
    """

    latLonConstant = Param(
        Params._dummy(),
        "latLonConstant",
        """Constant lat & lon to use in haversine distance calculation if only one
        input column is provided.""",
        typeConverter=TypeConverters.toListFloat,
    )

    def setLatLonConstant(self, value: List[float]) -> "LatLonConstantParams":
        """
        Sets the latLonConstant parameter.
        :param value: List of float lat and lon values.
        :returns: Instance of class mixed in.
        """
        if len(value) != 2:
            raise ValueError("latLonConstant must be a list of two floats: [lat, lon]")
        elif self.isDefined("inputCols") and len(self.getInputCols()) != 2:
            raise ValueError(
                f"""In order to set latLonConstant, there must be exactly two
                input columns. Found {len(self.getInputCols())} columns."""
            )
        elif value[0] < -90.0 or value[0] > 90.0:
            raise ValueError("Latitude must be between -90 and 90")
        elif value[1] < -180.0 or value[1] > 180.0:
            raise ValueError("Longitude must be between -180 and 180")
        return self._set(latLonConstant=value)

    def getLatLonConstant(self) -> List[float]:
        """
        Gets the latLonConstant parameter.
        :returns: List of float value of lat and lon used in haversine distance
        and bearing angle calculation.
        """
        return self.getOrDefault(self.latLonConstant)


class DefaultIntValueParams(Params):
    """
    Mixin class containing default integer parameter.
    """

    defaultValue = Param(
        Params._dummy(),
        "defaultValue",
        """
        Default int value to use in the transformer.
        """,
        typeConverter=TypeConverters.toInt,
    )

    def setDefaultValue(self, value: int) -> "DefaultIntValueParams":
        """
        Sets the defaultValue parameter.
        :param value: Value to set the defaultValue parameter to.
        :returns: Instance of class mixed in.
        """
        return self._set(defaultValue=value)

    def getDefaultValue(self) -> int:
        """
        Gets the defaultValue parameter.
        :returns: defaultValue param value.
        """
        return self.getOrDefault(self.defaultValue)
