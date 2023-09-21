import os

import pandas as pd
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from collect.airdata import request_airkorea_api, parse_airdata
from preprocess.airdata import airdata_to_df, preprocessing_for_model_ver2
from dataset.airdata import AirdataDataset


if __name__ == '__main__':
    load_dotenv()

    test_df = "tests/data/clean_df.csv"
    if not os.path.exists(test_df):
        response = request_airkorea_api(station_name="금천구", page_no=1, data_term="3MONTH")
        airdata = parse_airdata(response.content)
        air_df = airdata_to_df(airdata)
        clean_df = preprocessing_for_model_ver2(air_df)
        clean_df.to_csv(test_df, header=True, index=False)
    else:
        clean_df = pd.read_csv(test_df, header=0)
    airdata_dataset = AirdataDataset(clean_df)
    dl = DataLoader(
        airdata_dataset,
        batch_size=32,
        shuffle=True,
        sampler=None,
        num_workers=0,
        pin_memory=True,
    )

    print(next(iter(dl)))
    print(clean_df)