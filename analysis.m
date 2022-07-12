clear;
close all;

directory = "_data\results\";
files = dir(fullfile(directory, '*.mat'));

combined_filepath = [""];
combined_ticker = [""];
combined_ticker_set = [""];
combined_ticker_count = [];
combined_error = [];
combined_epochs = [];
combined_layers = [];
combined_sentiment = [];

str_vals={'False','True'};

combined_ticker = combined_ticker(2:end);
combined_filepath = combined_filepath(2:end);
combined_ticker_set = combined_ticker_set(2:end);

for i=1:length(files)
    filename = files(i).name;
    load(fullfile(directory, filename));
    
    combined_error(end + 1) = mean_absolute_percentage_error_test;
    combined_ticker(end + 1) = ticker;
    combined_filepath(end + 1) = fullfile(directory, filename);
    combined_epochs(end+1) = epochs;
    combined_layers(end+1) = layers;
    combined_sentiment(end+1) = sentiment;

    if not(ismember(ticker, combined_ticker_set))
        combined_ticker_set(end + 1) = ticker;
        combined_ticker_count(end + 1) = 0;
    end 

    index = find(combined_ticker_set == ticker);
    combined_ticker_count(index) = combined_ticker_count(index) + 1;
end

shape = [length(combined_ticker_set) max(combined_ticker_count)];
combined_error_matrix = zeros(shape);
combined_epochs_matrix = zeros(shape);
combined_layers_matrix = zeros(shape);
combined_sentiment_matrix = zeros(shape);
counter_matrix = zeros([1 max(combined_ticker_count)]);
for i=1:length(combined_ticker)
    ticker = combined_ticker(i);
    index = find(combined_ticker_set == ticker);
    counter_matrix(index) = counter_matrix(index) + 1;
    counter = counter_matrix(index);

    combined_error_matrix(index, counter) = combined_error(i);
    combined_epochs_matrix(index, counter) = combined_epochs(i);
    combined_layers_matrix(index, counter) = combined_layers(i);
    combined_sentiment_matrix(index, counter) = combined_sentiment(i);
end

num_errors_analysis = 3;
[~, worst_error] = maxk(combined_error, num_errors_analysis);
[~, best_error] = mink(combined_error, num_errors_analysis);

figure(Name="Best-Worst");
for i=1:num_errors_analysis
    index_value = best_error(i);
    
    filepath = combined_filepath(index_value);
    load(filepath);
    
    subplot(2, num_errors_analysis, i);
    hold on;
    test_dates = datetime(test_dates, 'InputFormat', 'yyyy-MM-dd');
    test_dates = datetime(test_dates, 'InputFormat', 'yyyy-MM-dd');
    plot(test_dates, prediction_test, DisplayName="Prediction");
    plot(test_dates, test_y, DisplayName="Real");
    xlabel("Time");
    ylabel("Normalised Price");
    title(append("Best ", num2str(i), ", ", ticker, " (", num2str(layers), ", ", str_vals(sentiment + 1), ", ", num2str(epochs), ")", ", Error= ", num2str(combined_error(index_value)), "%"));
    legend;
    hold off;
end
for i=1:num_errors_analysis
    index_value = worst_error(i);
    
    filepath = combined_filepath(index_value);
    load(filepath);
    
    subplot(2, num_errors_analysis, num_errors_analysis + i);
    hold on;
    test_dates = datetime(test_dates, 'InputFormat', 'yyyy-MM-dd');
    test_dates = datetime(test_dates, 'InputFormat', 'yyyy-MM-dd');
    plot(test_dates, prediction_test, DisplayName="Prediction");
    plot(test_dates, test_y, DisplayName="Real");
    xlabel("Time");
    ylabel("Normalised Price");
    title(append("Worst ", num2str(i), ", ", ticker, " (", num2str(layers), ", ", str_vals(sentiment + 1), ", ", num2str(epochs), ")", ", Error= ", num2str(combined_error(index_value)), "%"));
    legend;
    hold off;
end

figure(Name="Error");

subplot(2, 2, 1);
scatter(categorical(combined_ticker), combined_error);
ylabel("Error");
xlabel("Tickers");

subplot(2, 2, 2);
scatter(combined_layers, combined_error);
ylabel("Error");
xlabel("#LSTM Layers");
xlim([0.9 4.1]);

subplot(2, 2, 3);
scatter(combined_sentiment, combined_error);
ylabel("Error");
xlabel("Sentiment");
xlim([-0.1 1.1]);

subplot(2, 2, 4);
scatter(combined_epochs, combined_error);
ylabel("Error");
xlabel("Epochs");
xlim([45 155]);
