import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader

from collect.airdata import request_airkorea_api, parse_airdata
from preprocess.airdata import airdata_to_df, preprocessing_for_model_ver2
from dataset.airdata import AirdataDataset
from utils.utils import split_df
from model.af_gru import AirdataForecastGRU


def train():
    total_loss = 0.0
    iteration = 0

    model.train()
    for iteration, (item) in enumerate(train_dataloader):
        model.zero_grad()
        prediction = model(item)
        loss = criterion(prediction, item)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss / (iteration+1)


def evaluation():
    eval_loss_logs = []

    model.eval()
    with torch.no_grad():
        for iteration, (item) in enumerate(test_dataloader):
            prediction = model(item)
            loss = criterion(prediction, item)
            eval_loss_logs.append([iteration+1, loss.item()])
    return eval_loss_logs


if __name__ == '__main__':
    batch_size = 4
    input_size = 1
    num_classes = 1
    hidden_size = 1
    num_layers = 1
    seq_len = 1
    learning_rate = 0.001
    num_epoch = 1000
    seed = 2023

    load_dotenv()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    test_df = "tests/data/clean_df.csv"
    if not os.path.exists(test_df):
        response = request_airkorea_api(station_name="금천구", page_no=1, data_term="3MONTH")
        airdata = parse_airdata(response.content)
        air_df = airdata_to_df(airdata)
        clean_df = preprocessing_for_model_ver2(air_df)
        clean_df.to_csv(test_df, header=True, index=False)
    else:
        clean_df = pd.read_csv(test_df, header=0)

    train_df, test_df = split_df(clean_df[["pm_10"]], rate=0.3)

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

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_logs = []
    for epoch in range(num_epoch):
        loss = train()

        if epoch % 50 == 0:
            print(f"[{epoch+1}] avg loss : {loss}")

        train_loss_logs.append([epoch+1, loss])

    save_dst = ""
    torch.save(model, "artifact/af_gru_001.pth")

    # loss_df = pd.DataFrame(loss_logs, columns=["epoch", "loss"])
    # loss_df = loss_df.set_index(keys="epoch")
    # loss_df.plot()
    # plt.title("Train loss(MSE)")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.show()

    eval_loss_logs = evaluation()
    loss_df = pd.DataFrame(eval_loss_logs, columns=["step", "loss"])
    loss_df = loss_df.set_index(keys="step")
    loss_df.plot()
    plt.title("Evaluation loss(MSE)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.show()