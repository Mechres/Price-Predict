import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import yfinance as yf
import ta
from bayes_opt import BayesianOptimization
import yaml
import os


class XGBoost_Predictor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.ticker = self.config['ticker']
        self.start_date = self.config['start_date']
        self.end_date = self.config['end_date']
        self.test_size = self.config['test_size']
        self.plot_dir = self.config['plot_dir']
        self.hyperparameter_tuning = self.config['hyperparameter_tuning']
        self.scaler_y = MinMaxScaler()

        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        self.model = None

    def download_data(self):
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        df = df.dropna()

        # Technical Indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['RSI'] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
        df['MACD'] = ta.trend.MACD(close=df["Close"]).macd()
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.volatility.BollingerBands(
            close=df["Close"]).bollinger_hband(), ta.volatility.BollingerBands(
            close=df["Close"]).bollinger_mavg(), ta.volatility.BollingerBands(close=df["Close"]).bollinger_lband()
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()

        df['Day_of_Week'] = pd.to_datetime(df.index).dayofweek
        df['Month'] = pd.to_datetime(df.index).month

        # Shift for Previous Values
        for col in ['Close', 'SMA_20', 'EMA_12', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'OBV']:
            df[f'Prev_{col}'] = df[col].shift(1)

        df = df.dropna()

        return df

    def prepare_data(self, df):
        feature_columns = ['Prev_Close', 'Prev_SMA_20', 'Prev_EMA_12', 'Prev_RSI', 'Prev_MACD',
                           'Prev_BB_upper', 'Prev_BB_lower', 'Prev_OBV', 'Day_of_Week', 'Month', 'Volume']

        # Use TimeSeriesSplit for data splitting and get the last split for training and testing
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(df):
            pass  # Iterate until the last split

        # Fit and transform scaler for the final split
        scaler_x = MinMaxScaler()
        X_train = scaler_x.fit_transform(df[feature_columns].iloc[train_index])
        X_test = scaler_x.transform(df[feature_columns].iloc[test_index])

        self.scaler_y.fit(df[['Close']].iloc[train_index])  # Fit the scaler to the last split
        y_train = self.scaler_y.transform(df[['Close']].iloc[train_index])
        y_test = self.scaler_y.transform(df[['Close']].iloc[test_index])

        return X_train, X_test, y_train, y_test

    def optimize_xgb(self, X_train, y_train):
        def xgb_evaluate(max_depth, n_estimators, learning_rate, subsample, colsample_bytree):
            params = {
                'max_depth': int(max_depth),
                'n_estimators': int(n_estimators),
                'learning_rate': learning_rate,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
            }

            model = xgb.XGBRegressor(**params)
            tscv = TimeSeriesSplit(n_splits=5)

            scores = []
            for train_index, val_index in tscv.split(X_train):
                X_train_split, X_val_split = X_train[train_index], X_train[val_index]
                y_train_split, y_val_split = y_train[train_index], y_train[val_index]

                model.fit(X_train_split, y_train_split)
                predictions = model.predict(X_val_split)
                mse = mean_squared_error(y_val_split, predictions)
                scores.append(-mse)  # Negative MSE for maximization

            return np.mean(scores)

        optimizer = BayesianOptimization(
            f=xgb_evaluate,
            pbounds={
                'max_depth': (
                self.hyperparameter_tuning['max_depth']['min'], self.hyperparameter_tuning['max_depth']['max']),
                'n_estimators': (
                self.hyperparameter_tuning['n_estimators']['min'], self.hyperparameter_tuning['n_estimators']['max']),
                'learning_rate': (
                self.hyperparameter_tuning['learning_rate']['min'], self.hyperparameter_tuning['learning_rate']['max']),
                'subsample': (
                self.hyperparameter_tuning['subsample']['min'], self.hyperparameter_tuning['subsample']['max']),
                'colsample_bytree': (self.hyperparameter_tuning['colsample_bytree']['min'],
                                     self.hyperparameter_tuning['colsample_bytree']['max'])
            },
            random_state=42
        )

        optimizer.maximize(n_iter=self.hyperparameter_tuning['n_iter'])

        return optimizer.max['params']

    def train_model(self, X_train, y_train, params):

        params['max_depth'] = int(params['max_depth'])
        params['n_estimators'] = int(params['n_estimators'])
        params['learning_rate'] = float(params['learning_rate'])
        params['subsample'] = float(params['subsample'])
        params['colsample_bytree'] = float(params['colsample_bytree'])

        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred):
        y_true_real = self.scaler_y.inverse_transform(y_true)
        y_pred_real = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1))

        mse = mean_squared_error(y_true_real, y_pred_real)
        rmse = np.sqrt(mse)

        return rmse, y_true_real, y_pred_real


    def plot_results(self, y_true, y_pred):
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label="Actual Price", color="blue")
        plt.plot(y_pred, label="Predicted Price", color="red")
        plt.title(f"{self.ticker} Actual vs. Predicted Prices - XGBoost Model")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.savefig(os.path.join(self.plot_dir, f"{self.ticker}_prediction.png"))
        plt.close()

    def save_model(self, filename):
        self.model.save_model(filename)

    def load_model(self, filename):
        self.model = xgb.XGBRegressor()
        self.model.load_model(filename)

    def run(self):
        df = self.download_data()
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()

        # Prepare data using the scalers
        X_train, X_test, y_train, y_test = self.prepare_data(df)

        # Now use scaler_y in evaluate
        best_params = self.optimize_xgb(X_train, y_train)
        print("Best Parameters:", best_params)

        self.train_model(X_train, y_train, best_params)

        y_pred_train = self.predict(X_train)
        y_pred_test = self.predict(X_test)

        train_rmse, y_train_real, y_pred_train_real = self.evaluate(y_train, y_pred_train)
        test_rmse, y_test_real, y_pred_test_real = self.evaluate(y_test, y_pred_test)

        print("Train RMSE:", train_rmse)
        print("Test RMSE:", test_rmse)

        self.plot_results(y_test_real, y_pred_test_real)
        self.save_model("models/xgboost_model.json")

if __name__ == "__main__":
    predictor = XGBoost_Predictor("configs/xgboost_config.yaml")
    predictor.run()