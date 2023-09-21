import os

import pandas as pd
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader

from collect.airdata import request_airkorea_api, parse_airdata
from preprocess.airdata import airdata_to_df, preprocessing_for_model_ver2
from dataset.airdata import AirdataDataset
from utils.utils import split_df
from model.af_gru import AirdataForecastGRU


def train():
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_loss = 0.0
    iteration = 0

    for iteration, (item) in enumerate(train_dataloader):
        model.zero_grad()
        prediction = model(item)
        loss = criterion(prediction, item)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if iteration == 9:
            print(f"item : {item}")
            print(f"prediction : {prediction}")
            print(f"loss : {loss}")

    print(total_loss, iteration)

    return total_loss / (iteration+1)


if __name__ == '__main__':
    batch_size = 8
    input_size = 1
    num_classes = 1
    hidden_size = 1
    num_layers = 1
    seq_len = 1
    learning_rate = 0.001
    num_epoch = 3000

    load_dotenv()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_df = "tests/data/clean_df.csv"
    if not os.path.exists(test_df):
        response = request_airkorea_api(station_name="금천구", page_no=1, data_term="3MONTH")
        airdata = parse_airdata(response.content)
        air_df = airdata_to_df(airdata)
        clean_df = preprocessing_for_model_ver2(air_df)
        clean_df.to_csv(test_df, header=True, index=False)
    else:
        clean_df = pd.read_csv(test_df, header=0)

    train_df, test_df = split_df(clean_df[["pm_10"]], rate=0.2)

    train_dataset = AirdataDataset(train_df)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        num_workers=0,
        pin_memory=True,
    )

    test_dataset = AirdataDataset(test_df)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        num_workers=0,
        pin_memory=True,
    )

    model = AirdataForecastGRU(
        num_classes=num_classes, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, seq_len=seq_len
    )

    for epoch in range(num_epoch):
        loss = train()
        print(f"[{epoch+1}] avg loss : {loss}")