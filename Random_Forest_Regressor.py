import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import logging
import warnings
from joblib import dump, load
import yaml
import os

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RandomForestPredictor:
    def __init__(self, config_path='configs/random_forest_config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.model = None
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def download_and_prepare_data(self, symbol, start_date, end_date):
        try:
            logging.info(f"Downloading data for {symbol} from {start_date} to {end_date}")
            df = yf.download(symbol, start=start_date, end=end_date)

            # Add technical indicators
            df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
            df['EMA_12'] = EMAIndicator(close=df['Close'], window=12).ema_indicator()
            df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()

            # Add Bollinger Bands
            bollinger = BollingerBands(close=df['Close'], window=20, window_dev=2)
            df['BB_upper'] = bollinger.bollinger_hband()
            df['BB_lower'] = bollinger.bollinger_lband()

            # Add volatility
            df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()

            # Lag Features (Shifted values)
            for feature in ['Close', 'SMA_20', 'EMA_12', 'RSI', 'BB_upper', 'BB_lower', 'Volatility']:
                df[f'Prev_{feature}'] = df[feature].shift(1)

            # Define Target Variable
            df['target'] = df['Close'].shift(-1)

            # Drop NaN values
            df.dropna(inplace=True)

            logging.info("Data preparation completed successfully")
            return df
        except Exception as e:
            logging.error(f"Error in data preparation: {str(e)}")
            raise

    def prepare_features_and_target(self, df):
        features = ['Close', 'SMA_20', 'EMA_12', 'RSI', 'BB_upper', 'BB_lower', 'Volatility',
                    'Prev_Close', 'Prev_SMA_20', 'Prev_EMA_12', 'Prev_RSI', 'Prev_BB_upper', 'Prev_BB_lower',
                    'Prev_Volatility']
        X = df[features]
        y = df['target']
        return X, y

    def train_and_evaluate_model(self, X, y):
        # Split into Train and Test Sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config['test_size'], random_state=42,
                                                            shuffle=False)

        # Scaling
        X_train_scaled = self.scaler_x.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        X_test_scaled = self.scaler_x.transform(X_test)

        # Hyperparameter tuning
        param_dist = self.config['hyperparameter_tuning']

        rf = RandomForestRegressor(random_state=42)

        # Random search of parameters
        random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                           n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

        # Fit the random search model
        random_search.fit(X_train_scaled, y_train_scaled.ravel())

        logging.info(f"Best parameters found: {random_search.best_params_}")

        # Get the best model
        self.model = random_search.best_estimator_

        # Make Predictions
        predictions_scaled = self.model.predict(X_test_scaled)
        predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))

        # Evaluate Model
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        return X_test, y_test, predictions, mse, mae, r2

    def plot_results(self, y_test, predictions, ticker):
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test, label='Actual', marker='o')
        plt.plot(y_test.index, predictions, label='Predicted', marker='x')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{ticker} Price Prediction')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.config['plot_dir'], f'{ticker}_price_prediction.png'))
        plt.close()
        logging.info(f"Results plot saved as '{ticker}_price_prediction.png'")

    def save_model(self):
        if not os.path.exists(self.config['model_dir']):
            os.makedirs(self.config['model_dir'])
        model_path = os.path.join(self.config['model_dir'], 'random_forest_model.joblib')
        scaler_path = os.path.join(self.config['model_dir'], 'random_forest_scalers.joblib')
        dump(self.model, model_path)
        dump((self.scaler_x, self.scaler_y), scaler_path)
        logging.info(f"Model and scalers saved successfully to {self.config['model_dir']}")

    def load_model(self):
        model_path = os.path.join(self.config['model_dir'], 'random_forest_model.joblib')
        scaler_path = os.path.join(self.config['model_dir'], 'random_forest_scalers.joblib')
        self.model = load(model_path)
        self.scaler_x, self.scaler_y = load(scaler_path)
        logging.info(f"Model and scalers loaded successfully from {self.config['model_dir']}")


    def predict_new_data(self, ticker, start_date, end_date):
        df = self.download_and_prepare_data(ticker, start_date, end_date)
        X, y = self.prepare_features_and_target(df)
        X_scaled = self.scaler_x.transform(X)
        predictions_scaled = self.model.predict(X_scaled)
        predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))

        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        self.plot_results(y, predictions, ticker)

        return mse, mae, r2, y.index, y, predictions

    def run(self, ticker, start_date, end_date):
        df = self.download_and_prepare_data(ticker, start_date, end_date)
        X, y = self.prepare_features_and_target(df)
        X_test, y_test, predictions, mse, mae, r2 = self.train_and_evaluate_model(X, y)
        self.plot_results(y_test, predictions, ticker)
        return mse, mae, r2


def main():
    predictor = RandomForestPredictor()
    ticker = "BTC-USD"
    start_date = "2016-05-22"
    end_date = "2024-07-24"

    mse, mae, r2 = predictor.run(ticker, start_date, end_date)

    logging.info(f"Mean Squared Error: {mse}")
    logging.info(f"Mean Absolute Error: {mae}")
    logging.info(f"R-squared Score: {r2}")

    save_model = input("Do you want to save the model? (y/n): ").lower()
    if save_model == 'y':
        predictor.save_model()


if __name__ == "__main__":
    main()