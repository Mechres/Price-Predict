import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import yfinance as yf
import pandas as pd
import ta
from ta.momentum import RSIIndicator
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import yaml
from pathlib import Path


class LGBMRegressorModel:
    """
    A class for training and using a LightGBM regressor model for stock price prediction.

    This class handles data preprocessing, model training, hyperparameter tuning,
    prediction, and visualization for stock price data using LightGBM.
    """

    def __init__(self, config_path='configs/LGBM_Config.yaml'):
        """
        Initialize the LGBMRegressorModel.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        self.config = self.load_config(config_path)
        self.ensure_directories()

    @staticmethod
    def load_config(config_path):
        """
        Load configuration from a YAML file.

        Args:
            config_path (str): Path to the configuration YAML file.

        Returns:
            dict: Loaded configuration.
        """
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def ensure_directories(self):
        """Create necessary directories for plots and models if they don't exist."""
        for dir_name in ['plots', 'models']:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

    def yfdown(self, ticker, start, end):
        """
        Download stock data and prepare features for model training.

        Args:
            ticker (str): Stock ticker symbol.
            start (str): Start date for data download.
            end (str): End date for data download.

        Returns:
            tuple: X_train, X_test, y_train, y_test for model training.
        """
        df = yf.download(ticker, start=start, end=end)
        df = df.dropna()

        # Technical Indicators
        sma_window = self.config['technical_indicators']['sma_window']
        ema_window = self.config['technical_indicators']['ema_window']
        rsi_window = self.config['technical_indicators']['rsi_window']

        df['SMA_20'] = df['Close'].rolling(window=sma_window).mean()
        df['EMA_12'] = df['Close'].ewm(span=ema_window, adjust=False).mean()
        rsi_indicator = RSIIndicator(close=df["Close"], window=rsi_window)
        df['RSI'] = rsi_indicator.rsi()

        df['Day_of_Week'] = pd.to_datetime(df.index).dayofweek

        # Shift for Previous Values
        df['Prev_Close'] = df['Close'].shift(1)
        df['Prev_SMA_20'] = df['SMA_20'].shift(1)
        df['Prev_EMA_12'] = df['EMA_12'].shift(1)
        df['Prev_RSI'] = df['RSI'].shift(1)

        df = df.dropna()

        # Separate scalers for each column
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        x = self.scaler_x.fit_transform(df[['Prev_Close', 'Prev_SMA_20', 'Prev_EMA_12', 'Prev_RSI', 'Day_of_Week',
                                            'Volume', 'Open', 'High', 'Low']])
        y = self.scaler_y.fit_transform(df[['Close']])

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.config['test_size'], shuffle=False)
        return X_train, X_test, y_train, y_test

    def model(self, X_train, y_train, X_test, y_test, best_params):
        """
        Train the LightGBM model with the best parameters.

        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training target.
            X_test (np.array): Testing features.
            y_test (np.array): Testing target.
            best_params (dict): Best parameters from grid search.

        Returns:
            lgb.Booster: Trained LightGBM model.
        """
        params = self.config['lgbm_params'].copy()
        params.update(best_params)

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        early_stopping_callback = lgb.early_stopping(stopping_rounds=self.config['early_stopping_rounds'], verbose=True)

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=self.config['num_boost_round'],
            valid_sets=lgb_eval,
            callbacks=[early_stopping_callback]
        )

        return model

    def grid(self, X_train, y_train, X_test, y_test):
        """
        Perform grid search for hyperparameter tuning.

        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training target.
            X_test (np.array): Testing features.
            y_test (np.array): Testing target.

        Returns:
            tuple: GridSearchCV object and best parameters.
        """
        param_grid = self.config['param_grid']

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        early_stopping_callback = lgb.early_stopping(stopping_rounds=self.config['early_stopping_rounds'], verbose=True)

        grid_search = GridSearchCV(
            estimator=lgb.LGBMRegressor(**self.config['lgbm_params']),
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=self.config['cv_folds'],
            verbose=2
        )

        grid_search.fit(X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        callbacks=[early_stopping_callback])

        best_model = grid_search.best_estimator_
        print("Best Parameters:", grid_search.best_params_)

        return grid_search, grid_search.best_params_

    def yhat(self, ticker, model, X_test, y_test):
        """
        Make predictions and calculate RMSE.

        Args:
            ticker (str): Stock ticker symbol.
            model (lgb.Booster): Trained LightGBM model.
            X_test (np.array): Testing features.
            y_test (np.array): Testing target.

        Returns:
            float: Root Mean Squared Error (RMSE).
        """
        yhat = model.predict(X_test)
        y_test = self.scaler_y.inverse_transform(y_test)
        yhat = self.scaler_y.inverse_transform(yhat.reshape(-1, 1))

        rmse = math.sqrt(mean_squared_error(y_test, yhat))

        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Actual Price')
        plt.plot(yhat, label='Predicted Price')
        plt.title(f'{ticker} Price Prediction - LGBM Model')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig(f'plots/LGBM_{ticker}_prediction.png')
        plt.close()

        return rmse

    def save_model(self, model, ticker):
        """
        Save the trained model and scalers.

        Args:
            model (lgb.Booster): Trained LightGBM model.
            ticker (str): Stock ticker symbol.
        """
        joblib.dump(model, f'models/{ticker}_lgbm_model.joblib')
        joblib.dump(self.scaler_x, f'models/{ticker}_lgbm_scaler_x.joblib')
        joblib.dump(self.scaler_y, f'models/{ticker}_lgbm_scaler_y.joblib')

    def load_model(self, ticker):
        """
        Load a saved model and scalers.

        Args:
            ticker (str): Stock ticker symbol.

        Returns:
            lgb.Booster: Loaded LightGBM model.
        """
        model = joblib.load(f'models/{ticker}_lgbm_model.joblib')
        self.scaler_x = joblib.load(f'models/{ticker}_lgbm_scaler_x.joblib')
        self.scaler_y = joblib.load(f'models/{ticker}_lgbm_scaler_y.joblib')
        return model

    def run(self, ticker, start_date, end_date):
        """
        Run the entire modeling process: data preparation, training, and evaluation.

        Args:
            ticker (str): Stock ticker symbol.
            start_date (str): Start date for data download.
            end_date (str): End date for data download.

        Returns:
            tuple: Trained model and RMSE.
        """
        X_train, X_test, y_train, y_test = self.yfdown(ticker, start_date, end_date)
        grid_search, best_params = self.grid(X_train, y_train, X_test, y_test)
        model = self.model(X_train, y_train, X_test, y_test, best_params)
        rmse = self.yhat(ticker, model, X_test, y_test)
        print(f'RMSE: {rmse}')
        self.save_model(model, ticker)
        return model, rmse

    def predict_new_data(self, model, ticker, start_date, end_date):
        """
        Make predictions on new data.

        Args:
            model (lgb.Booster): Trained LightGBM model.
            ticker (str): Stock ticker symbol.
            start_date (str): Start date for new data.
            end_date (str): End date for new data.

        Returns:
            tuple: RMSE, dates, actual prices, and predicted prices.
        """
        df = yf.download(ticker, start=start_date, end=end_date)
        df = df.dropna()

        sma_window = self.config['technical_indicators']['sma_window']
        ema_window = self.config['technical_indicators']['ema_window']
        rsi_window = self.config['technical_indicators']['rsi_window']

        df['SMA_20'] = df['Close'].rolling(window=sma_window).mean()
        df['EMA_12'] = df['Close'].ewm(span=ema_window, adjust=False).mean()
        rsi_indicator = RSIIndicator(close=df["Close"], window=rsi_window)
        df['RSI'] = rsi_indicator.rsi()

        df['Day_of_Week'] = pd.to_datetime(df.index).dayofweek

        # Shift for Previous Values
        df['Prev_Close'] = df['Close'].shift(1)
        df['Prev_SMA_20'] = df['SMA_20'].shift(1)
        df['Prev_EMA_12'] = df['EMA_12'].shift(1)
        df['Prev_RSI'] = df['RSI'].shift(1)

        df = df.dropna()

        # Prepare features
        X = df[['Prev_Close', 'Prev_SMA_20', 'Prev_EMA_12', 'Prev_RSI', 'Day_of_Week', 'Volume', 'Open', 'High', 'Low']]
        X_scaled = self.scaler_x.transform(X)

        # Make predictions
        y_pred = model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1))

        # Calculate RMSE
        y_true = df['Close'].values.reshape(-1, 1)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, y_true, label='Actual Price')
        plt.plot(df.index, y_pred, label='Predicted Price')
        plt.title(f'{ticker} Price Prediction - LGBM Model (New Data)')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig(f'plots/LGBM_{ticker}_prediction_new_data.png')
        plt.close()

        return rmse, df.index, y_true, y_pred


if __name__ == "__main__":
    lgbm_model = LGBMRegressorModel()
    model, rmse = lgbm_model.run('BTC-USD', '2018-05-22', '2024-06-22')
    print(f'Final RMSE: {rmse}')