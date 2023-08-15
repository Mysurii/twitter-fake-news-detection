import time
from calendar import timegm


def get_day(time_stamp):
    """
    returns the day from a timestamp formatted like it was returned out of the Twitter API.
    :param time_stamp:
    :return: day string
    """
    return time_stamp[8:10]


def get_month(time_stamp):
    """
    Return an integer for the corresponding month in a timestamp string.
    :param time_stamp:
    :return: month integer
    """

    if 'Jan' in time_stamp:
        return 1
    elif 'Feb' in time_stamp:
        return 2
    elif 'Mar' in time_stamp:
        return 3
    elif 'Apr' in time_stamp:
        return 4
    elif 'May' in time_stamp:
        return 5
    elif 'Jun' in time_stamp:
        return 6
    elif 'Jul' in time_stamp:
        return 7
    elif 'Aug' in time_stamp:
        return 8
    elif 'Sep' in time_stamp:
        return 9
    elif 'Oct' in time_stamp:
        return 10
    elif 'Nov' in time_stamp:
        return 11
    elif 'Dec' in time_stamp:
        return 12


def get_year(time_stamp):
    """
    returns year from a timestamp formatted like it was returned out of the Twitter API.
    :param time_stamp:
    :return: year string
    """
    return time_stamp[26:30]


def convert_date_time_to_unix(time_stamp):
    """
    returns epoch time from a timestamp formatted like it was returned out of the Twitter API.
    :param time_stamp:
    :return: epoch time derived from time_stamp
    """
    month = get_month(time_stamp)
    day = get_day(time_stamp)
    year = get_year(time_stamp)
    time_hhmmss = time_stamp[11:19]
    utc_time = time.strptime("{}-{}-{}T{}".format(year, month, day, time_hhmmss), "%Y-%m-%dT%H:%M:%S")

    return timegm(utc_time)


if __name__ == '__main__':
    convert_date_time_to_unix("Tue Mar 10 14:38:00 +0000 2020")






