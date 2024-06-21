import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import numpy as np
import matplotlib.pyplot as plt
import ta
from ta.momentum import RSIIndicator
import pandas as pd
import pickle

class Lstm:
    def __init__(self):
        pass

    @staticmethod
    def yfdown(ticker, start, end):
        df = yf.download(ticker, start=start, end=end)
        df = df[['Close']].dropna()

        # Technical Indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        rsi_indicator = RSIIndicator(close=df["Close"], window=14)  # RSI indicator
        df['RSI'] = rsi_indicator.rsi()

        df['Day_of_Week'] = pd.to_datetime(df.index).dayofweek  # 0: Monday, ..., 6: Sunday

        # Shift for Previous Values
        df['Prev_Close'] = df['Close'].shift(1)
        df['Prev_SMA_20'] = df['SMA_20'].shift(1)
        df['Prev_EMA_12'] = df['EMA_12'].shift(1)
        df['Prev_RSI'] = df['RSI'].shift(1)

        df = df.dropna()  # Drop rows with NaN values

        # Separate scalers for each column
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        x = scaler_x.fit_transform(df[['Prev_Close', 'Prev_SMA_20', 'Prev_EMA_12', 'Prev_RSI', 'Day_of_Week']])
        y = scaler_y.fit_transform(df[['Close']])

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
        return X_train, X_test, y_train, y_test, scaler_y

    @staticmethod
    def model(X_train, y_train, X_test, y_test):
        inputs = Input(shape=(X_train.shape[1], 1))  # Adjust input shape
        x = Bidirectional(LSTM(units=128, return_sequences=True))(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(units=64)(x)
        x = Dense(32)(x)
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
        model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), verbose=2)
        return model

    @staticmethod
    def yhat(model, X_test, y_test, scaler):
        yhat = model.predict(X_test, verbose=0)

        y_test = scaler.inverse_transform(y_test)  # Use scaler_y
        yhat = scaler.inverse_transform(yhat)

        rmse = math.sqrt(mean_squared_error(y_test, yhat))

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Actual Price')
        plt.plot(yhat, label='Predicted Price')
        plt.title('Bitcoin Price Prediction - LSTM Model')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

        return rmse

    def save_model(model, scaler, filename):
        with open(filename, 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler}, f)

    def load_model(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            return data['model'], data['scaler']


''''' Test
lstm = Lstm()
X_train, X_test, y_train, y_test, scaler = lstm.yfdown('BTC-USD', '2020-05-24', '2024-06-02')
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
model = lstm.model(X_train, y_train, X_test, y_test)
rmse = lstm.yhat(model, X_test, y_test, scaler)
print(f'RMSE: {rmse}')
'''''
