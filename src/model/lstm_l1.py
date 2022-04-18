# https://towardsdatascience.com/lstm-time-series-forecasting-predicting-stock-prices-using-an-lstm-model-6223e9644a2f
import json
import os
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras.losses
from matplotlib import pyplot as plt
from scipy.io import savemat

from src.dataset.merge_data import FINAL_DATASET
from src.helper.path_function import get_relative_path

RESULT_DIR = Path(get_relative_path(__file__, f"../../_data/results_l1/"))


def get_lstm_model(input_shape, features, num_of_lstm=4):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=tuple(list(input_shape)[1:])))

    for i in range(num_of_lstm - 1):
        model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, dropout=0.33))

    model.add(tf.keras.layers.LSTM(units=50, dropout=0.33))
    model.add(tf.keras.layers.Dense(units=features))

    # model.summary()

    return model


def get_train_and_testing_dataset(ticker: str, col=None, start_date=None, end_date=None, sentiment=True, divide=0.8):
    filepath = FINAL_DATASET.joinpath(f"{ticker}.csv")
    if not (os.path.exists(filepath) and os.path.isfile(filepath)):
        raise Exception(f"Stock data for {ticker} not found")

    df = pd.read_csv(filepath)
    if start_date is not None:
        df = df[df['Date'] >= start_date]
    if end_date is not None:
        df = df[df['Date'] <= end_date]
    if col is not None:
        df = df.loc[:, ["Date", col, "positive", "negative", "neutral"]]

    dates = df.loc[:, "Date"]
    dates = dates.to_list()
    df = df.drop("Date", axis='columns')
    if not sentiment:
        df = df.drop("positive", axis='columns')
        df = df.drop("negative", axis='columns')
        df = df.drop("neutral", axis='columns')

    df = df.to_numpy()
    training_range = int(df.shape[0] * divide)
    training = df[:training_range, :]
    testing = df[training_range:, :]

    dates = [datetime.strptime(x, "%Y-%m-%d") for x in dates]
    return dates, training, testing


def normalize_dataset(training_data, test_data):
    data = np.concatenate((training_data, test_data), axis=0)

    scaled_data = tf.keras.utils.normalize(data[:, 0], axis=-1, order=1)
    scaled_data = scaled_data.T
    scaled_data = np.concatenate((scaled_data, data[:, 1:]), axis=1)

    training_data = scaled_data[:training_data.shape[0], :]
    test_data = scaled_data[training_data.shape[0]:, :]
    return training_data, test_data


def add_lag_to_data(training_data, test_data, time_steps=50):
    data = np.concatenate((training_data, test_data), axis=0)
    training_size = training_data.shape[0]
    training_data_x = []
    training_data_y = []
    test_data_x = []
    test_data_y = []

    for i in range(time_steps + 1, data.shape[0]):
        x_data = data[i - time_steps - 1: i - 1, :]
        y_data = data[i - 1: i, 0]
        if i < training_size:
            training_data_x.append(x_data)
            training_data_y.append(y_data)
        else:
            test_data_x.append(x_data)
            test_data_y.append(y_data)

    training_data_x = np.array(training_data_x)
    training_data_y = np.array(training_data_y)
    test_data_x = np.array(test_data_x)
    test_data_y = np.array(test_data_y)

    training_data_x = tf.convert_to_tensor(training_data_x)
    training_data_y = tf.convert_to_tensor(training_data_y)
    test_data_x = tf.convert_to_tensor(test_data_x)
    test_data_y = tf.convert_to_tensor(test_data_y)
    return (training_data_x, training_data_y), (test_data_x, test_data_y)


def run_model(
        ticker="GOOGL",
        start_date="2014-01-01",
        end_date="2019-12-31",
        layers=4,
        sentiment=True,
        epochs=50,
        to_dir=RESULT_DIR
):
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)

    filename = f"{ticker}_{start_date}_{end_date}_{epochs}_{sentiment}_{layers}"
    filepath = to_dir.joinpath(f"{filename}")

    if os.path.exists(f"{filepath}.json"):
        return

    print(filename)

    try:
        dates, training_data, test_data = get_train_and_testing_dataset(
            ticker=ticker,
            col="Close",
            start_date=start_date,
            end_date=end_date,
            sentiment=sentiment
        )
        training_data, test_data = normalize_dataset(training_data, test_data)
        training_data, test_data = add_lag_to_data(training_data, test_data)
        train_x, train_y = training_data
        test_x, test_y = test_data
        train_dates = dates[-train_x.shape[0] - test_x.shape[0]:-test_x.shape[0]]
        test_dates = dates[-test_x.shape[0]:]
        print(f"{len(train_dates)=}, {len(test_dates)=}")

        print(f"{train_x.shape=}, {train_y.shape=}, {test_x.shape=}, {test_y.shape=}")

        model = get_lstm_model(input_shape=train_x.shape, features=train_y.shape[-1], num_of_lstm=layers)
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(train_x, train_y, epochs=epochs, batch_size=64)

        prediction_train = model.predict(train_x)
        prediction_test = model.predict(test_x)
        print(f"{prediction_train.shape=}, {prediction_test.shape=}")

        save_data = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "epochs": epochs,
            "sentiment": sentiment,
            "layers": layers,
            "train_dates": [x.strftime("%Y-%m-%d") for x in train_dates],
            "test_dates": [x.strftime("%Y-%m-%d") for x in test_dates],
            "train_y": train_y[:, 0].numpy().tolist(),
            "test_y": test_y[:, 0].numpy().tolist(),
            "prediction_train": prediction_train[:, 0].tolist(),
            "prediction_test": prediction_test[:, 0].tolist(),

            "mean_absolute_error_train": float(keras.losses.mean_absolute_error(
                train_y[:, 0], prediction_train[:, 0]
            ).numpy()),
            "mean_squared_error_train": float(keras.losses.mean_squared_error(
                train_y[:, 0], prediction_train[:, 0]
            ).numpy()),
            "mean_absolute_percentage_error_train": float(keras.losses.mean_absolute_percentage_error(
                train_y[:, 0], prediction_train[:, 0]
            ).numpy()),
            "mean_squared_logarithmic_error_train": float(keras.losses.mean_squared_logarithmic_error(
                train_y[:, 0], prediction_train[:, 0]
            ).numpy()),

            "mean_absolute_error_test": float(keras.losses.mean_absolute_error(
                test_y[:, 0], prediction_test[:, 0]
            ).numpy()),
            "mean_squared_error_test": float(keras.losses.mean_squared_error(
                test_y[:, 0], prediction_test[:, 0]
            ).numpy()),
            "mean_absolute_percentage_error_test": float(keras.losses.mean_absolute_percentage_error(
                test_y[:, 0], prediction_test[:, 0]
            ).numpy()),
            "mean_squared_logarithmic_error_test": float(keras.losses.mean_squared_logarithmic_error(
                test_y[:, 0], prediction_test[:, 0]
            ).numpy()),
        }
        with open(f"{filepath}.json", "w") as file:
            file.write(json.dumps(save_data, indent=2))
        savemat(f"{filepath}.mat", save_data)

        plt.close()
        plt.plot(train_dates, train_y[:, 0], color='red', label='Real')
        plt.plot(train_dates, prediction_train[:, 0], color='blue', label='Predicted')
        plt.title(f"{ticker=}, {sentiment=}, LSTM layers={layers}, {epochs=} Train")
        plt.xlabel("Time")
        plt.ylabel("Normalised Close")
        plt.legend()
        plt.savefig(f"{filepath}_train.png")

        plt.close()
        plt.plot(test_dates, test_y[:, 0], color='red', label='Real')
        plt.plot(test_dates, prediction_test[:, 0], color='blue', label='Predicted')
        plt.title(f"{ticker=}, {sentiment=}, LSTM layers={layers}, {epochs=} Test")
        plt.xlabel("Time")
        plt.ylabel("Normalised Close")
        plt.legend()
        plt.savefig(f"{filepath}_test.png")
    except Exception as e:
        with open(f"{filepath}.log.txt", "w") as file:
            file.write(str(e))
            file.write(str(traceback.format_exc()))


def run_all_models():
    tickers = [
        "AAPL",
        "MSFT", "DUK", "NVDA", "FB", "UNH",
        "PG", "V", "COST", "HD", "MA", "ABBV", "AVGO",
        "KO", "PEP", "LLY", "TMO", "CRM", "ABT", "WMT",
        "ADBE", "AMD", "MCD", "UNP", "NKE", "TXN",
        "NEE", "RTX", "LOW", "SPGI", "MDT", "SCHW",
        "HON", "AMGN", "INTU", "ORCL", "DE", "NOW",
        "AXP", "PLD", "GS", "AMT", "TGT", "SBUX", "CB",
        "ADP", "ZTS", "SYK", "ADI", "MDLZ"
    ]
    sentiment = [True, False]
    num_layers = [1, 2, 3, 4]
    epochs = [50, 100, 150]

    parameters = []
    for t in tickers:
        for s in sentiment:
            for n in num_layers:
                for e in epochs:
                    parameters.append({
                        "ticker": t,
                        "layers": n,
                        "sentiment": s,
                        "epochs": e,
                    })

    for i, p in enumerate(parameters):
        print(f"Running {i}/{len(parameters)}")
        run_model(**p)


if __name__ == '__main__':
    run_all_models()
