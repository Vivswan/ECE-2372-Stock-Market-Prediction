# https://huggingface.co/ProsusAI/finbert

import csv
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.dataset.data_news import CRON_NEWS_CSV, SENTIMENT_NEWS_CSV

bert_tokenizer = None
bert_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def initialise_bert():
    global bert_tokenizer, bert_model
    if bert_tokenizer is None:
        bert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    if bert_model is None:
        bert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        bert_model.to(device)


@torch.no_grad()
def sentiment_analysis_bert(text):
    global bert_tokenizer, bert_model
    if not isinstance(text, list):
        text = [text]
    initialise_bert()

    inputs = bert_tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    inputs = inputs.to(device)

    outputs = bert_model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)

    return predictions


def analysis_news_csv(from_dir=CRON_NEWS_CSV, to_dir=SENTIMENT_NEWS_CSV, header="title"):
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)

    print(device)
    global bert_model
    initialise_bert()

    from_files = os.listdir(from_dir)
    for i, file in enumerate(from_files):
        from_filepath = Path(from_dir).joinpath(file)
        to_filepath = Path(to_dir).joinpath(file)

        if not os.path.isfile(from_filepath):
            continue

        with open(from_filepath, "r", encoding="utf8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            rows = [row for row in csv_reader]

        header_row = rows[0] + list(bert_model.config.id2label.values())
        rows = rows[1:]
        stock_column = header_row.index(header)
        headlines = [x[stock_column] for x in rows]
        sentiments = sentiment_analysis_bert(headlines).tolist()
        rows = [(v + sentiments[x]) for x, v in enumerate(rows)]

        with open(to_filepath, "w", encoding="utf8", newline="") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(header_row)
            csv_writer.writerows(rows)

        if i % 25 == 0:
            print(f"{i}/{len(from_files)}: {i / len(from_files) * 100:0.4f}%")


if __name__ == '__main__':
    analysis_news_csv()
