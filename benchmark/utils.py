# -*- coding: utf-8 -*-
"""
| **@created on:** 08/06/18,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
| 
|
| **Sphinx Documentation Status:** --
|
..todo::
"""

import datetime
from typeguard import typechecked


class TimeUtils(object):
    """
    | **@author:** Prathyush SP
    |
    | Time Utils
    """

    class DateTime(object):
        """
        | **@author:** Prathyush SP
        |
        | Time Utils for DateTime objects
        """

        @staticmethod
        def calculate_seconds(dt: datetime):
            """
            | **@author:** Prathyush SP
            |
            | Calculate Seconds
            :param dt: Datetime Object
            :return: seconds
            """
            return float(
                dt.year * 12 * 30 * 24 * 60 * 60 + dt.month * 30 * 24 * 60 * 60 + dt.day * 24 * 60 * 60 + dt.hour * 60 * 60 + dt.minute * 60 + dt.second + dt.microsecond / 10 ** 6)

        @staticmethod
        def calculate_minutes(dt: datetime):
            """
            | **@author:** Prathyush SP
            |
            | Calculate Minutes
            :param dt: Datetime Object
            :return: Minutes
            """
            return float(TimeUtils.DateTime.calculate_seconds(dt) / 60)

        @staticmethod
        def calculate_hours(dt: datetime):
            """
            | **@author:** Prathyush SP
            |
            | Calculate Hours
            :param dt: Datetime Object
            :return: Hours
            """
            return float(TimeUtils.DateTime.calculate_seconds(dt) / (60 * 60))

        @staticmethod
        def calculate_days(dt: datetime):
            """
            | **@author:** Prathyush SP
            |
            | Calculate Days
            :param dt: Datetime Object
            :return: Days
            """
            return float(TimeUtils.DateTime.calculate_seconds(dt) / (24 * 60 * 60))

        @staticmethod
        def calculate_months(dt: datetime):
            """
            | **@author:** Prathyush SP
            |
            | Calculate Months
            :param dt: Datetime Object
            :return: Months
            """
            return float(TimeUtils.DateTime.calculate_seconds(dt) / (30 * 24 * 60 * 60))

        @staticmethod
        def calculate_years(dt: datetime):
            """
            | **@author:** Prathyush SP
            |
            | Calculate Years
            :param dt: Datetime Object
            :return: Years
            """
            return float(TimeUtils.DateTime.calculate_seconds(dt) / (12 * 24 * 60 * 60))

    class TimeDelta(object):
        """
        | **@author:** Prathyush SP
        |
        | Time Utils for TimeDelta objects
        """

        @staticmethod
        def calculate_seconds(dt: datetime):
            """
            | **@author:** Prathyush SP
            |
            | Calculate Seconds
            :param dt: TimeDelta Object
            :return: Seconds
            """
            return float(dt.hour * 60 * 60 + dt.minute * 60 + dt.second + dt.microsecond / 10 ** 6)

        @staticmethod
        def calculate_minutes(dt: datetime):
            """
            | **@author:** Prathyush SP
            |
            | Calculate Minutes
            :param dt: TimeDelta Object
            :return: Minutes
            """
            return float(TimeUtils.TimeDelta.calculate_seconds(dt) / 60)

        @staticmethod
        def calculate_hours(dt: datetime):
            """
            | **@author:** Prathyush SP
            |
            | Calculate Hours
            :param dt: TimeDelta Object
            :return: Hours
            """
            return float(TimeUtils.TimeDelta.calculate_seconds(dt) / (60 * 60))


@typechecked
def generate_timestamp() -> str:
    """
    | **@author:** Prathyush SP
    |
    | Genetate Timestamp
    :return: Timestamp in String : Format(YYYY-MM-DD_HH:mm:SEC)
    .. todo::
        Prathyush SP:
            1. Support for custom formatting

    """
    # todo: Prathyush SP: Support for custom formatting
    return str(datetime.datetime.now().replace(microsecond=0)).replace(' ', '_').replace(':', '').replace('-', '')


class DLTimer(object):
    """
    | **@author:** Prathyush SP
    |
    | DL Timer Class
    """

    def __init__(self):
        """
        | **@author:** Prathyush SP
        |
        | DL Timer Class Constructor
        """
        self.start_time = None
        pass

    def set_time(self):
        """
        | **@author:** Prathyush SP
        |
        |Set Time
        :return: DLTimer Object
        """
        self.start_time = datetime.datetime.now()
        return self

    def get_elapsed_time(self):
        """
        | **@author:** Prathyush SP
        |
        Get Elapsed Time
        :return: Time Elapsed in hh:mm:sec.milli_sec
        """
        return datetime.datetime.now() - self.start_time
