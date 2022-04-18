# https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests
import csv
import math
import os
from datetime import datetime
from operator import itemgetter
from pathlib import Path

import numpy as np

from src.helper.path_function import get_relative_path

NEWS_CSV = get_relative_path(__file__, "../../_data/analyst_ratings_processed.csv")
SANITIZED_CSV = get_relative_path(__file__, "../../_data/analyst_ratings_sanitized.csv")
DISTRIBUTED_NEWS_CSV = get_relative_path(__file__, f"../../_data/news_csv/")
CRON_NEWS_CSV = get_relative_path(__file__, f"../../_data/news_csv_cron/")
SENTIMENT_NEWS_CSV = get_relative_path(__file__, f"../../_data/news_csv_cron_sentiment/")
DATE_MERGE_NEWS_CSV = get_relative_path(__file__, f"../../_data/news_csv_cron_sentiment_date_merge/")


def sanitize_news_csv(filename=NEWS_CSV, sanitize_file=SANITIZED_CSV):
    with open(filename, "r", encoding='utf-8') as reader, open(sanitize_file, "w", encoding='utf-8') as writer:
        lines = reader.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            while "  " in line:
                line = line.replace("  ", " ")

            if len(line) == 0:
                continue

            if len(lines) > i + 1 and lines[i + 1][0] == ",":
                writer.write(line)
            else:
                writer.write(line + "\n")


def list_stocks_tickers(filename=SANITIZED_CSV, header="stock"):
    set_of_tickers = set()
    header_rows = True
    stock_column = -1

    with open(filename, "r", encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if header_rows:
                stock_column = row.index(header)
                header_rows = False
            else:
                set_of_tickers.add(row[stock_column])

    return sorted(list(set_of_tickers))


def distribute_news_data_csv(from_file=SANITIZED_CSV, to_dir=DISTRIBUTED_NEWS_CSV, header="stock"):
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)

    all_news = {}
    header_row = None
    stock_column = None
    with open(from_file, "r", encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if header_row is None:
                header_row = row
                stock_column = header_row.index(header)
            else:
                ticker = row[stock_column].strip()
                if ticker not in all_news:
                    all_news[ticker] = []
                all_news[ticker].append(row)

    for key in all_news:
        with open(Path(to_dir).joinpath(f"{key}.csv"), "w", encoding="utf8", newline="") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow([header_row[2], header_row[1]])
            for i in all_news[key]:
                csv_writer.writerow([i[2], i[1]])


def cron_news_data_csv(from_dir=DISTRIBUTED_NEWS_CSV, to_dir=CRON_NEWS_CSV):
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
        rows = [([datetime.strptime(x[0][:x[0].rfind("-")], "%Y-%m-%d %H:%M:%S").timestamp()] + x[1:]) for x in rows]
        rows = sorted(rows, key=itemgetter(0))
        rows = [([datetime.fromtimestamp(x[0]).strftime("%Y-%m-%d %H:%M:%S")] + x[1:]) for x in rows]

        with open(to_filepath, "w", encoding="utf8", newline="") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(header)
            csv_writer.writerows(rows)

        if i % 25 == 0:
            print(f"{i}/{len(from_files)}: {i / len(from_files) * 100:0.4f}%")


def date_merge_news_data_csv(from_dir=SENTIMENT_NEWS_CSV, to_dir=DATE_MERGE_NEWS_CSV):
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

        header_row = rows[0]
        rows = rows[1:]
        rows = [([datetime.strptime(x[0], "%Y-%m-%d %H:%M:%S").timestamp()] + x[1:]) for x in rows]
        rows = [([datetime.fromtimestamp(x[0]).strftime("%Y-%m-%d")] + x[1:]) for x in rows]
        data = {}
        for row in rows:
            if row[0] not in data:
                data[row[0]] = {}
                data[row[0]]['positive'] = []
                data[row[0]]['negative'] = []
                data[row[0]]['neutral'] = []

            data[row[0]]['positive'].append(float(row[2]))
            data[row[0]]['negative'].append(float(row[3]))
            data[row[0]]['neutral'].append(float(row[4]))

        for key in data:
            data[key]['positive'] = np.mean(data[key]['positive'])
            data[key]['negative'] = np.mean(data[key]['negative'])
            data[key]['neutral'] = np.mean(data[key]['neutral'])

        rows = [
            [
                x,
                data[x]['positive'],
                data[x]['negative'],
                data[x]['neutral'],
            ] for x in data
        ]
        rows = [([datetime.strptime(x[0], "%Y-%m-%d").timestamp()] + x[1:]) for x in rows]
        rows = sorted(rows, key=itemgetter(0))
        rows = [([datetime.fromtimestamp(x[0]).strftime("%Y-%m-%d")] + x[1:]) for x in rows]
        header_row = ["date", "positive", "negative", "neutral"]

        with open(to_filepath, "w", encoding="utf8", newline="") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(header_row)
            csv_writer.writerows(rows)

        if i % 25 == 0:
            print(f"{i}/{len(from_files)}: {i / len(from_files) * 100:0.4f}%")


if __name__ == '__main__':
    # sanitize_news_csv()
    # distribute_news_data_csv()
    # cron_news_data_csv()
    # date_merge_news_data_csv()
    with open("tickers.csv", "w") as file:
        file.write("\n".join(list_stocks_tickers()))
