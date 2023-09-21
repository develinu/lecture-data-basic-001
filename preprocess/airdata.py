import datetime

import pandas as pd


def airdata_to_df(data: list) -> pd.DataFrame:
    return pd.DataFrame(data)


def preprocessing_for_model_ver1(df: pd.DataFrame):
    """
    groupby 기준을 시작 item으로부터 (24 - 1)개씩 그룹핑한다.

    이슈 : 중간에 데이터 누락되는 경우 기준이 틀어질 수 있음

    :param df: 수집된 에어코리아 데이터
    :return: 일별 평균 pm_10 데이터
    """

    agg_df = df.groupby(df.index // 24).agg({"pm_10": "mean"}).reset_index()
    return agg_df


def preprocessing_for_model_ver2(df: pd.DataFrame):
    """
    ver1의 이슈를 해결한 버전.
    groupby 기준을 날짜 기준으로 한다.

    :param df: 수집된 에어코리아 데이터
    :return: 일별 평균 pm_10 데이터
    """

    df["event_time"] = df["event_time"].apply(lambda x: datetime.datetime.utcfromtimestamp(x))
    df["event_time"] = df["event_time"].apply(lambda x: x.strftime("%Y-%m-%d"))
    agg_df = df.groupby(by="event_time").agg({"pm_10": "mean"}).reset_index()
    return agg_df
