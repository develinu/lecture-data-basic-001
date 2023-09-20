import json
from dotenv import load_dotenv

from collect.airdata import request_airkorea_api, parse_airdata
from preprocess.airdata import airdata_to_df, preprocessing_for_model_ver1, preprocessing_for_model_ver2


load_dotenv()


def test_request_airkorea_api():
    response = request_airkorea_api(station_name="서초구", page_no=1)
    assert response.status_code == 200


def test_parse_airdata():
    response = request_airkorea_api(station_name="서초구", page_no=1, data_term="MONTH")

    if response.status_code != 200:
        return json.dumps(response)

    airdata = parse_airdata(response.content)

    assert True == ("event_time" in airdata[-1].keys())


def test_preprocessing_airdata_for_model_ver1():
    response = request_airkorea_api(station_name="서초구", page_no=1, data_term="MONTH")

    if response.status_code != 200:
        return json.dumps(response)

    airdata = parse_airdata(response.content)
    df = airdata_to_df(airdata)
    agg_df = preprocessing_for_model_ver1(df)
    assert len(agg_df) > 20


def test_preprocessing_airdata_for_model_ver2():
    response = request_airkorea_api(station_name="서초구", page_no=1, data_term="MONTH")

    if response.status_code != 200:
        return json.dumps(response)

    airdata = parse_airdata(response.content)
    df = airdata_to_df(airdata)
    agg_df = preprocessing_for_model_ver2(df)
    assert len(agg_df) > 20
