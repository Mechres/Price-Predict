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


class LGBMRegressorModel:
    def __init__(self):
        pass

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

        df = df.dropna()

        # Separate scalers for each column
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        x = scaler_x.fit_transform(df[['Prev_Close', 'Prev_SMA_20', 'Prev_EMA_12', 'Prev_RSI', 'Day_of_Week', 'Volume',
                                       'Open', 'High', 'Low']])
        y = scaler_y.fit_transform(df[['Close']])

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
        return X_train, X_test, y_train, y_test, scaler_y

    @staticmethod
    def model(X_train, y_train, X_test, y_test, best_params):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 16,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        # Early stopping callback
        early_stopping_callback = lgb.early_stopping(stopping_rounds=5, verbose=True)

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=100,
            valid_sets=lgb_eval,
            callbacks=[early_stopping_callback]
        )

        return model

    @staticmethod
    def grid(X_train, y_train, X_test, y_test, scaler_y):
        param_grid = {
            'num_leaves': [16, 31, 64, 128],
            'learning_rate': [0.1, 0.01, 0.05, 0.1],
            'max_depth': [-5, -1, 5, 10, 20],
        }

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        # Early stopping callback
        early_stopping_callback = lgb.early_stopping(stopping_rounds=5, verbose=True)

        # Grid Search with early stopping
        grid_search = GridSearchCV(
            estimator=lgb.LGBMRegressor(),
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=5,  # 5-fold cross-validation
            verbose=2
        )

        grid_search.fit(X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        callbacks=[early_stopping_callback])

        best_model = grid_search.best_estimator_
        rmse = LGBMRegressorModel.yhat(best_model, X_test, y_test, scaler_y)
        print(f'RMSE: {rmse}')
        print("Best Parameters:", grid_search.best_params_)

        return grid_search, grid_search.best_params_

    @staticmethod
    def yhat(model, X_test, y_test, scaler):
        yhat = model.predict(X_test)
        y_test = scaler.inverse_transform(y_test)
        yhat = scaler.inverse_transform(yhat.reshape(-1, 1))  # Reshape for scaler

        rmse = math.sqrt(mean_squared_error(y_test, yhat))

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Actual Price')
        plt.plot(yhat, label='Predicted Price')
        plt.title('Bitcoin Price Prediction - LGBM Model')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

        return rmse

    @staticmethod
    def savemodel():
        pass

    @staticmethod
    def loadmodel():
        pass


"""
#Test
X_train, X_test, y_train, y_test, scaler_y = LGBMRegressorModel.yfdown('BTC-USD', '2018-05-22', '2024-06-22')
grid_search, grid_search.best_params_ = LGBMRegressorModel.grid(X_train, y_train, X_test, y_test, scaler_y)
best_params = grid_search.best_params_
model = LGBMRegressorModel.model(X_train, y_train, X_test, y_test, best_params)
rmse = LGBMRegressorModel.yhat(model, X_test, y_test, scaler_y)
print(f'RMSE: {rmse}')
"""
