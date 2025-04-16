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
from typing import Optional

import tensorflow as tf


def add_missing_time_components_to_datetime_tensor(
    datetime_tensor: tf.Tensor, max_len: Optional[int] = None
) -> tf.Tensor:
    """
    Adds missing time components to a date string tensor.
    If the time components are missing, they will be added as zeros.

    :param datetime_tensor: date string tensor.
    Must be in yyyy-MM-dd (HH:mm:ss.SSS) format. Can be truncated, and missing time
    components will be added as zeros.
    :param max_len: Maximum length to append time to if the time is missing. Used to
    avoid unnecessary computation. E.g. if we only need hour, then don't add
    milliseconds. Default is None.
    :returns: Date string tensor with missing time components added as zeros.
    """
    if max_len is not None and max_len < 10:
        raise ValueError(
            """max_len must be at least 10, as this is the minimum length
            of a date string."""
        )
    # Add missing time components, these are at 10, 13, 16 and 19 characters
    # For hours, minutes, seconds and milliseconds respectively
    str_lens = [10, 13, 16, 19]
    str_suffixes = [" 00:00:00.000", ":00:00.000", ":00.000", ".000"]
    # Filter out the suffixes that are longer than the max_len. This allows us to not
    # add time components if we don't need them.
    str_loop = (
        filter(lambda x: x[0] <= max_len, zip(str_lens, str_suffixes))
        if max_len is not None
        else zip(str_lens, str_suffixes)
    )
    for str_len, str_suffix in str_loop:
        dynamic_str_len = tf.strings.length(datetime_tensor)
        datetime_tensor = tf.where(
            dynamic_str_len == str_len,
            tf.strings.join([datetime_tensor, str_suffix], ""),
            datetime_tensor,
        )
    return datetime_tensor


def datetime_days_to_month(datetime_tensor: tf.Tensor) -> tf.Tensor:
    """
    Helper function for some datetime functions.
    Gets the number of days to the month of the given datetime tensor.

    :param: datetime_tensor: date(time) string tensor.
    Must be in yyyy-MM-dd (HH:mm:ss.SSS) format.

    WARNING: Dates are not checked for validity, so if you pass in a date such
    as "2020-02-30" no errors will be thrown, and you will get a nonsense output.

    :returns: Number of days to month, stored as tf.int64.
    """
    # 30 days have September...
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    # Extract date parts
    year = datetime_year(datetime_tensor)
    month = datetime_month(datetime_tensor)
    days_to_month = tf.reduce_sum(
        tf.stack(
            [
                tf.where(month > idx + 1, 1, 0) * n_days
                for idx, n_days in enumerate(days_in_month)
            ],
            axis=-1,
        ),
        -1,
    ) + (
        tf.where(month > 2, 1, 0)
        * tf.where((year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0)), 1, 0)
    )

    days_to_month = tf.cast(days_to_month, tf.int64)

    return days_to_month


def datetime_year(datetime_tensor: tf.Tensor) -> tf.Tensor:
    """
    Utility function to parse a date(time) tensor into a year tensor.
    Uses native tf functions only to avoid serialization issues.

    :param: datetime_tensor: date(time) string tensor.
    Must be in yyyy-MM-dd (HH:mm:ss.SSS) format.

    WARNING: Dates are not checked for validity, so if you pass in a date such
    as "2020-02-30" no errors will be thrown, and you will get a nonsense output.

    :returns: Year tensor, stored as tf.int64.
    """
    year = tf.strings.to_number(
        tf.strings.substr(datetime_tensor, 0, 4), out_type=tf.int64
    )
    return year


def datetime_month(datetime_tensor: tf.Tensor) -> tf.Tensor:
    """
    Utility function to parse a date(time) tensor into a month tensor.
    Uses native tf functions only to avoid serialization issues.

    :param: datetime_tensor: date(time) string tensor.
    Must be in yyyy-MM-dd (HH:mm:ss.SSS) format.

    WARNING: Dates are not checked for validity, so if you pass in a date such
    as "2020-02-30" no errors will be thrown, and you will get a nonsense output.

    :returns: Month tensor, stored as tf.int64.
    """
    month = tf.strings.to_number(
        tf.strings.substr(datetime_tensor, 5, 2), out_type=tf.int64
    )
    return month


def datetime_day(datetime_tensor: tf.Tensor) -> tf.Tensor:
    """
    Utility function to parse a date(time) tensor into a day tensor.
    Uses native tf functions only to avoid serialization issues.

    :param: datetime_tensor: date(time) string tensor.
    Must be in yyyy-MM-dd (HH:mm:ss.SSS) format.

    WARNING: Dates are not checked for validity, so if you pass in a date such
    as "2020-02-30" no errors will be thrown, and you will get a nonsense output.

    :returns: Day tensor, stored as tf.int64.
    """
    day = tf.strings.to_number(
        tf.strings.substr(datetime_tensor, 8, 2), out_type=tf.int64
    )
    return day


def datetime_hour(datetime_tensor: tf.Tensor) -> tf.Tensor:
    """
    Utility function to parse a date(time) tensor into an hour tensor.
    Uses native tf functions only to avoid serialization issues.

    :param: datetime_tensor: date(time) string tensor.
    Must be in yyyy-MM-dd (HH:mm:ss.SSS) format.

    WARNING: Dates are not checked for validity, so if you pass in a date such
    as "2020-02-30" no errors will be thrown, and you will get a nonsense output.

    :returns: Hour tensor, stored as tf.int64.
    """
    datetime_tensor = add_missing_time_components_to_datetime_tensor(
        datetime_tensor, max_len=13
    )
    hour = tf.strings.to_number(
        tf.strings.substr(datetime_tensor, 11, 2), out_type=tf.int64
    )
    return hour


def datetime_minute(datetime_tensor: tf.Tensor) -> tf.Tensor:
    """
    Utility function to parse a date(time) tensor into a minute tensor.
    Uses native tf functions only to avoid serialization issues.

    :param: datetime_tensor: date(time) string tensor.
    Must be in yyyy-MM-dd (HH:mm:ss.SSS) format.

    WARNING: Dates are not checked for validity, so if you pass in a date such
    as "2020-02-30" no errors will be thrown, and you will get a nonsense output.

    :returns: Minute tensor, stored as tf.int64.
    """
    datetime_tensor = add_missing_time_components_to_datetime_tensor(
        datetime_tensor, max_len=16
    )
    minute = tf.strings.to_number(
        tf.strings.substr(datetime_tensor, 14, 2), out_type=tf.int64
    )
    return minute


def datetime_second(datetime_tensor: tf.Tensor) -> tf.Tensor:
    """
    Utility function to parse a date(time) tensor into a second tensor.
    Uses native tf functions only to avoid serialization issues.

    :param: datetime_tensor: date(time) string tensor.
    Must be in yyyy-MM-dd (HH:mm:ss.SSS) format.

    WARNING: Dates are not checked for validity, so if you pass in a date such
    as "2020-02-30" no errors will be thrown, and you will get a nonsense output.

    :returns: Second tensor, stored as tf.int64.
    """
    datetime_tensor = add_missing_time_components_to_datetime_tensor(
        datetime_tensor, max_len=19
    )
    second = tf.strings.to_number(
        tf.strings.substr(datetime_tensor, 17, 2), out_type=tf.int64
    )
    return second


def datetime_millisecond(datetime_tensor: tf.Tensor) -> tf.Tensor:
    """
    Utility function to parse a date(time) tensor into a millisecond tensor.
    Uses native tf functions only to avoid serialization issues.

    :param: datetime_tensor: date(time) string tensor.
    Must be in yyyy-MM-dd (HH:mm:ss.SSS) format.

    WARNING: Dates are not checked for validity, so if you pass in a date such
    as "2020-02-30" no errors will be thrown, and you will get a nonsense output.

    :returns: Millisecond tensor, stored as tf.int64.
    """
    datetime_tensor = add_missing_time_components_to_datetime_tensor(datetime_tensor)
    millisecond = tf.strings.to_number(
        tf.strings.substr(datetime_tensor, 20, 3), out_type=tf.int64
    )
    return millisecond


def datetime_total_days(datetime_tensor: tf.Tensor) -> tf.Tensor:
    """
    Utility function to parse a date(time) tensor into a total days tensor.
    Uses native tf functions only to avoid serialization issues.

    :param: datetime_tensor: date(time) string tensor.
    Must be in yyyy-MM-dd (HH:mm:ss.SSS) format.

    WARNING: Dates are not checked for validity, so if you pass in a date such
    as "2020-02-30" no errors will be thrown, and you will get a nonsense output.

    :returns: Total days tensor, stored as tf.int64.
    """
    year = datetime_year(datetime_tensor)
    day = datetime_day(datetime_tensor)
    first_century_year_post_1970 = tf.constant([2000], dtype=tf.int64)
    num_standard_days = (year - 1970) * 365
    # Compute the number of leap years to know if we need to add extra days.
    # We only consider year - 1, since if we are currently in a leap year, this will
    # be catered for in days_to_month.
    num_standard_leap_years = ((year - 1) - 1972) // 4
    num_century_years = tf.where(
        year > first_century_year_post_1970,
        ((year - 1) - first_century_year_post_1970) // 100,
        0,
    )
    num_century_leap_years = tf.where(
        year > first_century_year_post_1970,
        ((year - 1) - first_century_year_post_1970) // 400,
        0,
    )
    # Subtract all century years and add all century leap years.
    num_leap_years = (
        num_standard_leap_years - num_century_years + num_century_leap_years
    )
    # Days to year is the number of standard days across all the years plus the number
    # of leap years (as each leap year adds exactly 1 day)
    days_to_year = num_standard_days + num_leap_years
    days_to_month = datetime_days_to_month(datetime_tensor)
    # Add all the days together
    total_days = days_to_year + days_to_month + day

    return total_days


def datetime_total_seconds(datetime_tensor: tf.Tensor) -> tf.Tensor:
    """
    Utility function to parse a date(time) tensor into a total seconds tensor.
    Uses native tf functions only to avoid serialization issues.

    :param: datetime_tensor: date(time) string tensor.
    Must be in yyyy-MM-dd (HH:mm:ss.SSS) format.

    WARNING: Dates are not checked for validity, so if you pass in a date such
    as "2020-02-30" no errors will be thrown, and you will get a nonsense output.

    :returns: Total seconds tensor, stored as tf.int64.
    """
    # Extract date parts
    total_days = tf.cast(datetime_total_days(datetime_tensor), dtype=tf.float64)
    hour = tf.cast(datetime_hour(datetime_tensor), dtype=tf.float64)
    minute = tf.cast(datetime_minute(datetime_tensor), dtype=tf.float64)
    second = tf.cast(datetime_second(datetime_tensor), dtype=tf.float64)
    milliseconds = tf.cast(datetime_millisecond(datetime_tensor), dtype=tf.float64)
    # Add all the seconds together
    total_seconds = (
        (total_days * 24 * 60 * 60)
        + (hour * 60 * 60)
        + (minute * 60)
        + second
        + (milliseconds / tf.constant(1000.0, dtype=tf.float64))
    )
    return total_seconds


def datetime_total_milliseconds(datetime_tensor: tf.Tensor) -> tf.Tensor:
    """
    Utility function to parse a date(time) tensor into a total milliseconds tensor.
    Uses native tf functions only to avoid serialization issues.

    :param: datetime_tensor: date(time) string tensor.
    Must be in yyyy-MM-dd (HH:mm:ss.SSS) format.

    WARNING: Dates are not checked for validity, so if you pass in a date such
    as "2020-02-30" no errors will be thrown, and you will get a nonsense output.

    :returns: Total milliseconds tensor, stored as tf.int64.
    """
    # Extract date parts
    total_days = datetime_total_days(datetime_tensor)
    hour = datetime_hour(datetime_tensor)
    minute = datetime_minute(datetime_tensor)
    second = datetime_second(datetime_tensor)
    millisecond = datetime_millisecond(datetime_tensor)
    # Add all the milliseconds together
    total_milliseconds = (
        (total_days * 24 * 60 * 60 * 1000)
        + (hour * 60 * 60 * 1000)
        + (minute * 60 * 1000)
        + (second * 1000)
        + millisecond
    )
    return total_milliseconds


def datetime_weekday(datetime_tensor: tf.Tensor) -> tf.Tensor:
    """
    Utility function to parse a date(time) tensor into a weekday tensor.
    Uses native tf functions only to avoid serialization issues.

    :param: datetime_tensor: date(time) string tensor.
    Must be in yyyy-MM-dd (HH:mm:ss.SSS) format.

    WARNING: Dates are not checked for validity, so if you pass in a date such
    as "2020-02-30" no errors will be thrown, and you will get a nonsense output.

    :returns: Weekday tensor, stored as tf.int64.
    """
    total_days = datetime_total_days(datetime_tensor)
    # Compute the weekday
    week_day = (total_days - 4) % 7 + 1
    return week_day


def datetime_is_weekend(datetime_tensor: tf.Tensor) -> tf.Tensor:
    """
    Utility function to parse a date(time) tensor into a weekend tensor.
    Uses native tf functions only to avoid serialization issues.

    :param: datetime_tensor: date(time) string tensor.
    Must be in yyyy-MM-dd (HH:mm:ss.SSS) format.

    WARNING: Dates are not checked for validity, so if you pass in a date such
    as "2020-02-30" no errors will be thrown, and you will get a nonsense output.

    :returns: Weekend tensor, stored as tf.int64.
    """
    week_day = datetime_weekday(datetime_tensor)
    # Compute the weekend
    is_weekend = tf.cast(tf.where(week_day > 5, 1, 0), tf.int64)
    return is_weekend


def datetime_day_of_year(datetime_tensor: tf.Tensor) -> tf.Tensor:
    """
    Utility function to parse a date(time) tensor into a day of year tensor.
    Uses native tf functions only to avoid serialization issues.

    :param: datetime_tensor: date(time) string tensor.
    Must be in yyyy-MM-dd (HH:mm:ss.SSS) format.

    WARNING: Dates are not checked for validity, so if you pass in a date such
    as "2020-02-30" no errors will be thrown, and you will get a nonsense output.

    :returns: Day of year tensor, stored as tf.int64.
    """
    day = datetime_day(datetime_tensor)
    days_to_month = datetime_days_to_month(datetime_tensor)
    # Add all the days together
    day_of_year = days_to_month + day

    return day_of_year


def datetime_add_days(
    datetime_tensor: tf.Tensor, num_days: tf.Tensor, include_time: bool = True
) -> tf.Tensor:
    """
    Adds a number of days to a date(time) string tensor.

    :param datetime_tensor: date(time) string tensor.
    Must be in yyyy-MM-dd (HH:mm:ss.SSS) format.
    :param num_days: Number of days to add.
    :param include_time: Whether to include the time in the output. If True, the output
    will be in yyyy-MM-dd HH:mm:ss.SSS format. If False, the output will be in
    yyyy-MM-dd format. Default is True.
    :returns: Date(time) string tensor with num_days added.
    """
    total_seconds = datetime_total_seconds(datetime_tensor)
    num_days_seconds = num_days * tf.constant(24 * 60 * 60, dtype=num_days.dtype)
    total_seconds += num_days_seconds
    return unix_timestamp_to_datetime(
        tf.cast(total_seconds, dtype=tf.float64), include_time=include_time
    )


def unix_timestamp_to_datetime(
    timestamp_tensor: tf.Tensor, include_time: bool = True
) -> tf.Tensor:
    """
    Converts a timestamp tensor (seconds since Unix Epoch) into a datetime string
    tensor. If include_time is False, the output will be in yyyy-MM-dd, if include_time
    is True, the output will be in yyyy-MM-dd HH:mm:ss.SSS format.

    :param timestamp_tensor: the timestamp tensor to convert.
    Timestamps must be in seconds since unix epoch.
    :param include_time: Whether to include the time in the output. If True, the output
    will be in yyyy-MM-dd HH:mm:ss.SSS format. If False, the output will be in
    yyyy-MM-dd format. Default is True.
    :returns: Datetime string tensor in either yyyy-MM-dd or yyyy-MM-dd HH:mm:ss.SSS
    format.
    """

    # Days, hours, minutes and seconds since Unix Epoch
    seconds_in_one_minute = tf.constant(60.0, dtype=tf.float64)
    seconds_in_one_hour = tf.math.multiply(seconds_in_one_minute, 60.0)
    seconds_in_one_day = tf.math.multiply(seconds_in_one_hour, 24.0)
    total_days = tf.math.floordiv(timestamp_tensor, seconds_in_one_day)

    # Initialise the remainder days variable
    remainder_days = total_days
    days_in_4_years = tf.constant(1461.0, dtype=tf.float64)
    year = tf.add(
        tf.constant(1970.0, dtype=tf.float64),
        tf.multiply(
            tf.math.floordiv(remainder_days, days_in_4_years),
            tf.constant(4.0, dtype=tf.float64),
        ),
    )
    remainder_days = tf.math.mod(remainder_days, days_in_4_years)

    # Let k = the number of 4 year chunks since 1970
    # We count from 1970 + 4k, so every 3rd year is a leap year
    # (e.g. 1970 + 4k, 1971 + 4k, ^^1972 + 4^^)
    # We don't need to count the last year as the remainder will get
    # carried on to the next loop where the month is computed
    # TODO: Is there a better abstraction instead of for loops?
    #  These are O(1) operations, but feel clunky and also not very clear
    year_days = [
        tf.constant(365.0, dtype=tf.float64),
        tf.constant(365.0, dtype=tf.float64),
        tf.constant(366.0, dtype=tf.float64),
    ]
    for d in year_days:
        year_passed = tf.where(
            remainder_days >= d,
            tf.constant(1.0, dtype=tf.float64),
            tf.constant(0.0, dtype=tf.float64),
        )
        year += year_passed
        remainder_days -= year_passed * d

    # The full days in year that have been realised
    full_days_in_year = remainder_days

    # Initialise month loop variables
    # Days in the month (we treat leap years in the loop)
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    months_to_month = tf.zeros_like(total_days)
    remainder_days = full_days_in_year

    # First loop starts from December and works backwards
    for idx, _ in enumerate(month_days):
        n_months = 12 - idx

        cumulative_days_to_month = (
            # Leap year treatment (if we are in a leap year)
            # A leap year is one that is divisible by 4, unless it is divisible by 100
            # but not divisible by 400
            (
                tf.where(
                    (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0)),
                    tf.constant(1.0, dtype=tf.float64),
                    tf.constant(0.0, dtype=tf.float64),
                )
                * tf.where(
                    n_months >= 2,
                    tf.constant(1.0, dtype=tf.float64),
                    tf.constant(0.0, dtype=tf.float64),
                )
            )
            # Cumulative days in a normal year
            + sum(month_days[:n_months])
        )

        # Elements will be zero unless ALL cumulative_days_to_month have been realised,
        # in which case the element will be 1
        month_has_been_realised = remainder_days // cumulative_days_to_month
        remainder_days -= month_has_been_realised * cumulative_days_to_month
        months_to_month += n_months * month_has_been_realised

    # The month we are in hasn't been realised fully, but we are in it (so +1)
    month = months_to_month + 1
    # The day we are in has not been realised fully, but we are in it (so +1)
    day = remainder_days + 1

    year_str = tf.strings.as_string(tf.cast(year, dtype=tf.int64))
    month_str = tf.strings.as_string(tf.cast(month, dtype=tf.int64), width=2, fill="0")
    day_str = tf.strings.as_string(tf.cast(day, dtype=tf.int64), width=2, fill="0")
    date = tf.strings.join([year_str, month_str, day_str], "-")

    if include_time:
        leftover_seconds = timestamp_tensor - tf.math.multiply(
            total_days, seconds_in_one_day
        )
        total_hours = tf.math.floordiv(leftover_seconds, seconds_in_one_hour)
        leftover_seconds -= tf.math.multiply(total_hours, seconds_in_one_hour)

        total_mins = tf.math.floordiv(leftover_seconds, seconds_in_one_minute)
        leftover_seconds -= tf.math.multiply(total_mins, seconds_in_one_minute)
        total_seconds = tf.math.floor(leftover_seconds)
        total_milliseconds = leftover_seconds - total_seconds

        hours_str = tf.strings.as_string(
            tf.cast(total_hours, dtype=tf.int64), width=2, fill="0"
        )
        minutes_str = tf.strings.as_string(
            tf.cast(total_mins, dtype=tf.int64), width=2, fill="0"
        )
        seconds_str = tf.strings.as_string(
            tf.cast(total_seconds, dtype=tf.int64), width=2, fill="0"
        )
        milliseconds_str = tf.strings.as_string(
            # We need to round the milliseconds to fix them to 3 decimal places
            tf.cast(tf.math.round(total_milliseconds * 1000.0), tf.int64),
            width=3,
            fill="0",
        )

        time = tf.strings.join(
            [
                tf.strings.join([hours_str, minutes_str, seconds_str], ":"),
                milliseconds_str,
            ],
            ".",
        )
        datetime = tf.strings.join([date, time], " ")
        return datetime

    return date


def datetime_to_unix_timestamp(datetime_tensor: tf.Tensor) -> tf.Tensor:
    """
    Converts a date string tensor into a timestamp tensor (seconds since Unix Epoch).

    :param datetime_tensor: the date tensor to convert.
        Must be in yyyy-MM-dd (HH:mm:ss.SSS) format.
    :returns: Timestamp tensor in seconds since Unix Epoch
    """
    return datetime_total_seconds(datetime_tensor)
