 Crypto Price Prediction Using PySpark & LSTM

 Overview

This project explores the use of both traditional machine learning and deep learning techniques to predict the price of Bitcoin (BTC) against Tether (USDT) using historical minute-level trading data. It includes feature engineering with PySpark, data visualization, and price forecasting with Gradient Boosted Trees (GBTRegressor) and Long Short-Term Memory (LSTM) neural networks.

The dataset used is aggregated from cryptodatadownload.com for multiple coins, processed and stored in a unified CSV format. The pipeline covers data acquisition, visualization, model training, and evaluation.

 Key Components

1. Data Acquisition & Preprocessing

Downloaded minute-level historical data for 17 popular cryptocurrencies.

Unified the datasets into a single CSV: binance_multicoin_dataset.csv.

Used PySpark for loading, filtering, and transforming data efficiently at scale.

2. Visualization

Used Matplotlib and Pandas to:

Plot BTC close prices over time.

Compare price trends of BTC, ETH, and LTC.

Visualize 50-minute moving average and rolling volatility.

3. Feature Engineering (PySpark)

Created rolling features:

5-min and 30-min moving averages

Rolling max, min, and standard deviation (volatility)

Generated lag features (t-1, t-2, t-3 closes)

Computed percentage returns

Added time-based features (hour, day of the week)

4. Machine Learning Model - GBTRegressor

Assembled features using VectorAssembler and scaled them.

Trained a Gradient Boosted Trees Regressor.

Evaluated using RMSE, MAE, and R².

Plotted predicted vs actual prices.

Displayed feature importances.

5. Deep Learning Model - LSTM (TensorFlow)

Preprocessed close prices using MinMaxScaler.

Created sequences of 60-minute windows for LSTM input.

Built a 2-layer LSTM model with dropout to prevent overfitting.

Used early stopping to optimize training.

Predicted on test data and visualized actual vs predicted prices.

