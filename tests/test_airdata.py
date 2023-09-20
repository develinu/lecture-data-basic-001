import json
from dotenv import load_dotenv

from collect.airdata import request_airkorea_api, parse_airdata


load_dotenv()


def test_request_airkorea_api():
    response = request_airkorea_api(station_name="서초구", page_no=1)
    assert response.status_code == 200


def test_parse_airdata():
    response = request_airkorea_api(station_name="서초구", page_no=1, data_term="DAILY")

    if response.status_code != 200:
        return json.dumps(response)

    airdata = parse_airdata(response.content)

    assert True == ("event_time" in airdata[-1].keys())
