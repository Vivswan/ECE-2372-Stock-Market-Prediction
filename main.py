from src.dataset.data_news import sanitize_news_csv, distribute_news_data_csv, cron_news_data_csv, \
    date_merge_news_data_csv
from src.dataset.data_stock import download_all_stock_data_csv, sanitize_stock_data_csv, cron_stock_data_csv
from src.dataset.merge_data import merge_data_csv, fill_stock_data_csv, filter_stock_data_csv
from src.model.bert import analysis_news_csv
from src.model.lstm import run_all_models

if __name__ == '__main__':
    # News and Sentiment Data
    sanitize_news_csv()
    distribute_news_data_csv()
    cron_news_data_csv()
    analysis_news_csv()
    date_merge_news_data_csv()

    # Stock Data
    download_all_stock_data_csv()
    sanitize_stock_data_csv()
    cron_stock_data_csv()

    # Merge News and Stock data
    merge_data_csv()
    fill_stock_data_csv()
    filter_stock_data_csv()

    # Run, train and test Model
    run_all_models()
