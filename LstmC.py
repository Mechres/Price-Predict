import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import ta
from ta.momentum import RSIIndicator
import pandas as pd
import pickle
from typing import Tuple, Dict, Any
import logging
from configs.LstmConfig import Config
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LSTMPredictor:
    def __init__(self):
        self.model = None
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    @staticmethod
    def yf_Down(ticker: str, start: str, end: str) -> pd.DataFrame:
        try:
            df = yf.download(ticker, start=start, end=end)
            if df.empty:
                raise ValueError(f"No data available for {ticker} between {start} and {end}")
            df = df.dropna()

            # Technical Indicators
            df['SMA_20'] = df['Close'].rolling(window=Config.SMA_WINDOW).mean()
            df['EMA_12'] = df['Close'].ewm(span=Config.EMA_WINDOW, adjust=False).mean()
            rsi_indicator = RSIIndicator(close=df["Close"], window=Config.RSI_WINDOW)
            df['RSI'] = rsi_indicator.rsi()

            df['Day_of_Week'] = pd.to_datetime(df.index).dayofweek

            # Shift for Previous Values
            for col in ['Close', 'SMA_20', 'EMA_12', 'RSI']:
                df[f'Prev_{col}'] = df[col].shift(1)

            return df.dropna()
        except Exception as e:
            logging.error(f"Error downloading stock data: {str(e)}")
            raise

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        try:
            x = self.scaler_x.fit_transform(df[Config.FEATURE_COLUMNS])
            y = self.scaler_y.fit_transform(df[[Config.TARGET_COLUMN]])
            return train_test_split(x, y, test_size=Config.VALIDATION_SPLIT, shuffle=False)
        except Exception as e:
            logging.error(f"Error preparing data: {str(e)}")
            raise

    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        try:
            inputs = Input(shape=input_shape)
            x = inputs
            for units in Config.LSTM_UNITS:
                x = Bidirectional(LSTM(units=units, return_sequences=True))(x)
                x = Dropout(Config.DROPOUT_RATE)(x)
            x = LSTM(units=Config.LSTM_UNITS[-1])(x)
            for units in Config.DENSE_UNITS:
                x = Dense(units, activation='relu')(x)
            outputs = Dense(1)(x)

            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=Config.OPTIMIZER, loss='mse')
            return model
        except Exception as e:
            logging.error(f"Error building model: {str(e)}")
            raise

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> None:
        try:
            self.model = self.build_model((X_train.shape[1], 1))

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

            history = self.model.fit(
                X_train, y_train,
                batch_size=Config.BATCH_SIZE,
                epochs=Config.EPOCHS,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping, reduce_lr],
                verbose=2
            )
            return history
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        try:
            yhat = self.model.predict(X_test, verbose=0)
            return self.scaler_y.inverse_transform(yhat)
        except Exception as e:
            logging.error(f"Error making predictions: {str(e)}")
            raise

    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def plot_results(y_true: np.ndarray, y_pred: np.ndarray, ticker: str) -> None:
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(y_true, label='Actual Price')
            plt.plot(y_pred, label='Predicted Price')
            plt.title(f'{ticker} Price Prediction - LSTM Model')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.show()
            plt.savefig(os.path.join(os.getenv('PLOT_DIR', '.'), f'{ticker}_prediction_plot.png'))
        except Exception as e:
            logging.error(f"Error plotting results: {str(e)}")
            raise

    def save_model(self, filename: str) -> None:
        try:
            with open(filename, 'wb') as f:
                pickle.dump({'model': self.model, 'scaler_x': self.scaler_x, 'scaler_y': self.scaler_y}, f)
            logging.info(f"Model saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise

    @classmethod
    def load_model(cls, filename: str) -> 'LSTMPredictor':
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            predictor = cls()
            predictor.model = data['model']
            predictor.scaler_x = data['scaler_x']
            predictor.scaler_y = data['scaler_y']
            logging.info(f"Model loaded from {filename}")
            return predictor
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    @staticmethod
    def run(ticker: str) -> None:
        print("LSTM selected.")
        predictor = LSTMPredictor()

        print("Load saved model? (Must be in same directory.)")
        selection_c = input("Y/N: ").upper()

        try:
            if selection_c == "Y":
                predictor = LSTMPredictor.load_model(Config.MODEL_SAVE_PATH)
                logging.info("Model loaded successfully.")
                Config.START_DATE = input("Start Date (YYYY-MM-DD): ")
                Config.END_DATE = input("End Date (YYYY-MM-DD): ")
                Config.TICKER = ticker
                df = predictor.yf_Down(Config.TICKER, Config.START_DATE, Config.END_DATE)
                X_train, X_test, y_train, y_test = predictor.prepare_data(df)
                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                y_pred = predictor.predict(X_test)
                y_true = predictor.scaler_y.inverse_transform(y_test)
                rmse = predictor.evaluate_model(y_true, y_pred)
                logging.info(f'RMSE: {rmse}')
                predictor.plot_results(y_true, y_pred, Config.TICKER)

            elif selection_c == "N":
                Config.START_DATE = input("Start Date (YYYY-MM-DD): ")
                Config.END_DATE = input("End Date (YYYY-MM-DD): ")
                Config.TICKER = ticker

                # Download and prepare data
                df = predictor.yf_Down(Config.TICKER, Config.START_DATE, Config.END_DATE)
                X_train, X_test, y_train, y_test = predictor.prepare_data(df)

                # Reshape data for LSTM input
                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                # Train model
                history = predictor.train_model(X_train, y_train, X_test, y_test)

                # Make predictions
                y_pred = predictor.predict(X_test)
                y_true = predictor.scaler_y.inverse_transform(y_test)

                # Evaluate model
                rmse = predictor.evaluate_model(y_true, y_pred)
                logging.info(f'RMSE: {rmse}')

                # Plot results
                predictor.plot_results(y_true, y_pred, Config.TICKER)

                # Ask to save model
                save_model = input("Save model? Y/N: ").upper()
                if save_model == "Y":
                    predictor.save_model(Config.MODEL_SAVE_PATH)
                    logging.info("Model saved successfully.")
            else:
                logging.warning("Invalid selection. Please choose Y or N.")

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            print("An error occurred. Please check the logs for more information.")


"""
# Test
if __name__ == "__main__":
    try:
        predictor = LSTMPredictor()

        # Download and prepare data
        df = predictor.download_stock_data(Config.TICKER, Config.START_DATE, Config.END_DATE)
        X_train, X_test, y_train, y_test = predictor.prepare_data(df)

        # Train model
        history = predictor.train_model(X_train, y_train, X_test, y_test)

        # Make predictions
        y_pred = predictor.predict(X_test)
        y_true = predictor.scaler_y.inverse_transform(y_test)

        # Evaluate and plot results
        rmse = predictor.evaluate_model(y_true, y_pred)
        logging.info(f"RMSE: {rmse}")
        predictor.plot_results(y_true, y_pred, Config.TICKER)

        # Save model
        predictor.save_model(Config.MODEL_SAVE_PATH)

        # Load model
        loaded_predictor = LSTMPredictor.load_model(Config.MODEL_SAVE_PATH)
    except Exception as e:
        logging.error(f"An error occurred in the main execution: {str(e)}")"""