import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import ta
from ta.momentum import RSIIndicator
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


@staticmethod
def yfdown(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df.dropna()

    # Technical Indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    rsi_indicator = RSIIndicator(close=df["Close"], window=14)  # RSI indicator
    df['RSI'] = rsi_indicator.rsi()

    df['Day_of_Week'] = pd.to_datetime(df.index).dayofweek

    # Shift for Previous Values
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_SMA_20'] = df['SMA_20'].shift(1)
    df['Prev_EMA_12'] = df['EMA_12'].shift(1)
    df['Prev_RSI'] = df['RSI'].shift(1)

    df = df.dropna()  # Drop NaN

    df['Day_of_Week'] = pd.to_datetime(df.index).dayofweek
    print(df)
    # Separate scalers for each column
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x = scaler_x.fit_transform(df[['Prev_Close', 'Prev_SMA_20', 'Prev_EMA_12', 'Prev_RSI', 'Day_of_Week', 'Volume']])
    y = scaler_y.fit_transform(df[['Close']])

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test, scaler_y


@staticmethod
def xgbst(X_train, X_test, y_train, y_test, scaler_y):
    tscv = TimeSeriesSplit(n_splits=5)
    xgb_model = xgb.XGBRegressor(tree_method="hist")  # Change to 'hist' for efficiency
    param_grid = {
        "max_depth": [3, 5, 7],
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    clf = GridSearchCV(xgb_model, param_grid, scoring="neg_mean_squared_error", cv=tscv, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("Best Parameters:", clf.best_params_)
    print("Best MSE:", -clf.best_score_)

    # Model training and evaluation
    model = xgb.XGBRegressor(**clf.best_params_)
    model.fit(X_train, y_train)

    # Predictions and metrics
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)  # Predictions on training set

    # Inverse transform for accurate metrics
    y_test_real = scaler_y.inverse_transform(y_test)
    y_pred_real = scaler_y.inverse_transform(y_pred.reshape(-1, 1))

    mse_test = mean_squared_error(y_test_real, y_pred_real)
    rmse_test = np.sqrt(mse_test)

    y_train_real = scaler_y.inverse_transform(y_train)
    y_pred_train_real = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1))
    mse_train = mean_squared_error(y_train_real, y_pred_train_real)
    rmse_train = np.sqrt(mse_train)

    print("Test RMSE:", rmse_test)
    print("Train RMSE:", rmse_train)
    return y_test_real, y_pred_real


@staticmethod
def plot(y_test_real, y_pred_real):
    # Plot actual vs. predicted prices
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_real, label="Actual Price", color="blue")
    plt.plot(y_pred_real, label="Predicted Price", color="red")
    plt.title("BTC-USD Actual vs. Predicted Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def savemodel():
    pass


def loadmodel():
    pass


'''
#Test
X_train, X_test, y_train, y_test, scaler_y = yfdown('BTC-USD', '2018-05-22', '2024-06-20')
y_test_real, y_pred_real = xgbst(X_train, X_test, y_train, y_test, scaler_y)
plot(y_test_real, y_pred_real)'''
