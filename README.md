Requirements:
1. "analyst_ratings_processed.csv" from https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests which need to be directly present in "_data" folder.
2. Python enviroment with Pytorch, Tensorflow and other requirements mentioned in "requirements.txt"

Run the main.py to preform the following process incuding training and testing:

1. Sanitize and Filter News data
2. Run Sentiment analysis on News Data
3. Merge all sentiment from same day
4. Download stock data
5. Sanitize and Filter Stock data
6. Merge Sentiment and Stock data
7. Filter the data
8. Train model using L2 (to change the normalisation, change the normalize_dataset function in model/lstm.py)

Then use analysis.m with matlab to analysis the results. 
