import csv
import os
from datetime import datetime
from operator import itemgetter
from pathlib import Path

import numpy as np

from src.dataset.data_news import DATE_MERGE_NEWS_CSV
from src.dataset.data_stock import DATA_STOCK_CSV_SANITIZE_CRON
from src.helper.path_function import get_relative_path

DATA_MERGE_CSV = get_relative_path(__file__, f"../../_data/merge_csv/")
FILL_MERGE_CSV = get_relative_path(__file__, f"../../_data/merge_csv_fill/")
FILTER_MERGE_CSV = get_relative_path(__file__, f"../../_data/merge_csv_fill_filter/")
FINAL_DATASET = Path(FILTER_MERGE_CSV)

def merge_data_csv(
        from_stock=DATA_STOCK_CSV_SANITIZE_CRON,
        from_news=DATE_MERGE_NEWS_CSV,
        to_dir=DATA_MERGE_CSV
):
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)

    from_files = os.listdir(from_news)
    for i, file in enumerate(from_files):
        from_stock_filepath = Path(from_stock).joinpath(file)
        from_news_filepath = Path(from_news).joinpath(file)
        to_filepath = Path(to_dir).joinpath(file)

        if not os.path.isfile(from_stock_filepath):
            continue

        if not os.path.isfile(from_news_filepath):
            continue

        with open(from_stock_filepath, "r", encoding="utf8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            stock_data = [row for row in csv_reader]

        with open(from_news_filepath, "r", encoding="utf8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            news_data = [row for row in csv_reader]

        stock_data_header = stock_data[0]
        news_data_header = news_data[0][1:]
        merge_data_header = stock_data_header + news_data_header
        stock_data = stock_data[1:]
        news_data = news_data[1:]

        data = {}
        for row in stock_data:
            if row[0] not in data:
                data[row[0]] = {}
                for head in merge_data_header[1:]:
                    data[row[0]][head] = ""

            for head in stock_data_header[1:]:
                data[row[0]][head] = row[stock_data_header.index(head)]

        for row in news_data:
            if row[0] not in data:
                data[row[0]] = {}
                for head in merge_data_header[1:]:
                    data[row[0]][head] = ""

            for head in news_data_header[1:]:
                data[row[0]][head] = row[news_data_header.index(head)]

        rows = []
        for key, value in data.items():
            row = [key]
            for head in merge_data_header[1:]:
                row.append(value[head])
            rows.append(row)

        rows = [([int(datetime.strptime(x[0], "%Y-%m-%d").strftime("%Y%m%d"))] + x[1:]) for x in rows]
        rows = sorted(rows, key=itemgetter(0))
        rows = [([datetime.strptime(str(x[0]), "%Y%m%d").strftime("%Y-%m-%d")] + x[1:]) for x in rows]

        with open(to_filepath, "w", encoding="utf8", newline="") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(merge_data_header)
            csv_writer.writerows(rows)

        if i % 25 == 0:
            print(f"{i}/{len(from_files)}: {i / len(from_files) * 100:0.4f}%")


def fill_stock_data_csv(from_dir=DATA_MERGE_CSV, to_dir=FILL_MERGE_CSV, alpha=0.5):
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
        for index, value in enumerate(rows):
            if len(rows[index][-3]) > 0:
                continue

            if len(rows[index - 1][-3]) == 0:
                continue

            days = int(rows[index][0].replace("-", "")) - int(rows[index - 1][0].replace("-", ""))

            new_positive = np.log(float(rows[index - 1][-3])) / (-alpha)
            new_positive = np.exp(- alpha * (new_positive + days))

            new_negative = np.log(float(rows[index - 1][-2])) / (-alpha)
            new_negative = np.exp(- alpha * (new_negative + days))

            new_neutral = 1 - new_positive - new_negative

            new_positive = 0. if np.isclose(new_positive, 0.) else new_positive
            new_negative = 0. if np.isclose(new_negative, 0.) else new_positive

            rows[index][-1] = str(float(new_neutral))
            rows[index][-2] = str(float(new_negative))
            rows[index][-3] = str(float(new_positive))

        for index, value in enumerate(rows):
            if len(rows[index][-3]) > 0:
                continue

            rows[index][-1] = str(float(1.))
            rows[index][-2] = str(float(0.))
            rows[index][-3] = str(float(0.))

        with open(to_filepath, "w", encoding="utf8", newline="") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(header)
            csv_writer.writerows(rows)

        if i % 25 == 0:
            print(f"{i}/{len(from_files)}: {i / len(from_files) * 100:0.4f}%")


def filter_stock_data_csv(from_dir=FILL_MERGE_CSV, to_dir=FILTER_MERGE_CSV):
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

        with open(to_filepath, "w", encoding="utf8", newline="") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            for row in rows:
                if len(row[1]) == 0:
                    continue

                csv_writer.writerow(row)

        if i % 25 == 0:
            print(f"{i}/{len(from_files)}: {i / len(from_files) * 100:0.4f}%")


if __name__ == '__main__':
    # merge_data_csv()
    # fill_stock_data_csv()
    filter_stock_data_csv()
