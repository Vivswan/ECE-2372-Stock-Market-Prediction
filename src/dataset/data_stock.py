import csv
import os
from datetime import datetime
from operator import itemgetter
from pathlib import Path

import simplejson as simplejson
import yfinance
from pandas import DataFrame, Series, Timestamp

from src.dataset.data_news import list_stocks_tickers
from src.helper.path_function import get_relative_path

DATA_STOCK_CSV = get_relative_path(__file__, f"../../_data/stock_csv/")
DATA_STOCK_CSV_SANITIZE = get_relative_path(__file__, f"../../_data/stock_csv_sanitized/")
DATA_STOCK_CSV_SANITIZE_CRON = get_relative_path(__file__, f"../../_data/stock_csv_sanitized_cron/")


def dataframe_to_dict(dataframe: DataFrame):
    results = {
        "index": [(str(i) if isinstance(i, Timestamp) else i) for i in list(dataframe.index)]
    }

    for col, col_name in enumerate(dataframe):
        results[col_name] = [(str(i) if isinstance(i, Timestamp) else i) for i in dataframe.iloc[:, col].to_list()]

    return results


def get_ticker_data(ticker_id, period="max", interval="1d"):
    filename = f"{ticker_id}"
    filepath = get_relative_path(__file__, f"../../_data/stock_json/{filename}.json")
    results = None
    try:
        with open(filepath, "r") as file:
            results = simplejson.loads(file.read())
    except:
        pass

    if results is None:
        ticker = yfinance.Ticker(ticker_id)
        attributes = [
            "info",
            "actions",
            "dividends",
            "splits",
            "financials",
            "quarterly_financials",
            "major_holders",
            "institutional_holders",
            "balance_sheet",
            "quarterly_balance_sheet",
            "cashflow",
            "quarterly_cashflow",
            "earnings",
            "quarterly_earnings",
            "sustainability",
            "recommendations",
            "calendar",
            "isin",
            "options",
            "news",
        ]
        data = {
            "stock": dataframe_to_dict(ticker.history(period=period, interval=interval)),
        }

        for attr in attributes:
            value = getattr(ticker, attr)

            if isinstance(value, DataFrame):
                value = dataframe_to_dict(value)

            if isinstance(value, Series):
                value = value.to_list()

            if isinstance(value, dict):
                for key in list(value.keys()):
                    if isinstance(key, Timestamp):
                        value[str(key)] = value.pop(key)

            data[attr] = value

        with open(filepath, "w") as file:
            file.write(simplejson.dumps(data, ignore_nan=True, indent=2))

    return results


def download_all_stock_data_json(tickers=None):
    if tickers is None:
        tickers = list_stocks_tickers()

    for i in tickers:
        try:
            print(f"Downloading {i}...")
            get_ticker_data(i)
        except Exception as e:
            print(f"Error with {i}...")

    print("Downloading Completed...")


def download_all_stock_data_csv(tickers=None, to_dir=DATA_STOCK_CSV):
    if tickers is None:
        tickers = list_stocks_tickers()

    if not os.path.exists(to_dir):
        os.mkdir(to_dir)

    ticker = " ".join(tickers)

    data = yfinance.download(ticker, group_by="ticker")
    for i in tickers:
        filepath = Path(to_dir).joinpath(f"{i}.csv")
        data[i].to_csv(filepath)

        if i % 25 == 0:
            print(f"{i}/{len(tickers)}: {i/len(tickers) * 100:0.4f}%")

    print("Downloading Completed...")


def sanitize_stock_data_csv(from_dir=DATA_STOCK_CSV, to_dir=DATA_STOCK_CSV_SANITIZE):
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)

    from_files = os.listdir(from_dir)
    for i, file in enumerate(from_files):
        from_filepath = Path(from_dir).joinpath(file)
        to_filepath = Path(to_dir).joinpath(file)

        if not os.path.isfile(from_filepath):
            continue

        rows = []
        with open(from_filepath, "r", encoding="utf8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) == 0:
                    continue

                if len(row[1]) == 0:
                    continue

                rows.append(row)

        with open(to_filepath, "w", encoding="utf8", newline="") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerows(rows)

        if i % 25 == 0:
            print(f"{i}/{len(from_files)}: {i/len(from_files) * 100:0.4f}%")


def cron_stock_data_csv(from_dir=DATA_STOCK_CSV_SANITIZE, to_dir=DATA_STOCK_CSV_SANITIZE_CRON):
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)

    from_files = os.listdir(from_dir)
    for i, file in enumerate(from_files):
        from_filepath = Path(from_dir).joinpath(file)
        to_filepath = Path(to_dir).joinpath(file)

        if not os.path.isfile(from_filepath):
            continue

        with open(from_filepath, "r", encoding="utf8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            rows = [row for row in csv_reader]

        header = rows[0]
        rows = rows[1:]
        rows = [([int(datetime.strptime(x[0], "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d"))] + x[1:]) for x in rows]
        rows = sorted(rows, key=itemgetter(0))
        rows = [([datetime.strptime(str(x[0]), "%Y%m%d").strftime("%Y-%m-%d")] + x[1:]) for x in rows]

        with open(to_filepath, "w", encoding="utf8", newline="") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(header)
            csv_writer.writerows(rows)

        if i % 25 == 0:
            print(f"{i}/{len(from_files)}: {i/len(from_files) * 100:0.4f}%")


if __name__ == '__main__':
    # download_all_stock_data_csv()
    # sanitize_stock_data_csv()
    cron_stock_data_csv()
    pass
