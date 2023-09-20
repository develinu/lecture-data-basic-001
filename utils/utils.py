import re
import datetime


def get_hour(dt: str):
    pattern = "\d{4}-\d{2}-\d{2} (\d{2}):\d{2}"
    result = re.findall(pattern, dt)
    return result[-1]


def convert_dt(dt: str) -> datetime.datetime:
    dt_format = "%Y-%m-%d %H:%M"
    pattern = "(\d{4}-\d{2}-\d{2}) \d{2}:(\d{2})"
    prev_hours = re.sub(pattern, r"\1 23:\2", dt)
    return datetime.datetime.strptime(prev_hours, dt_format) + datetime.timedelta(hours=1)


def safe_cast(val, to_type, default="null"):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default
