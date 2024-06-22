import math
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt


class CatboostPredictor:
    def __init__(self):
        pass

    @staticmethod
    def yfdown(ticker, start, end):
        df = yf.download(ticker, start=start, end=end)
        df = df[['Close']].dropna()
        df['Prev_Close'] = df['Close'].shift(1)
        df = df.dropna()
        x = df[['Prev_Close']]
        y = df['Close']
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def searchcatboost(X_train, y_train):
        train_pool = Pool(X_train, y_train)

        # Parameter grid
        grid = {'iterations': [500, 1000, 2000, 3000],
                'learning_rate': [0.1, 0.01, 0.001],
                'depth': [2, 4, 6, 8]}

        # Perform grid search
        model = CatBoostRegressor(loss_function='RMSE')  # Instantiate the model
        grid_search_result = model.grid_search(grid, train_pool)

        # Extract the best parameters and the RMSE values for each fold
        best_params = grid_search_result['params']
        rmse_values = grid_search_result['cv_results']['test-RMSE-mean']

        # Find the minimum RMSE value from the list
        best_score = min(rmse_values)

        print("Best parameters: ", best_params)
        print("Best CV score (RMSE): ", best_score)
        return best_params


    @staticmethod
    def catboost_model(X_train, y_train, X_test, y_test, best_params):
        model = CatBoostRegressor(**best_params)
        model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        print(f"Mean Squared Error: {mse}")
        print(f"RMSE: {math.sqrt(mean_squared_error(y_test, pred))}")
        print(f"MAE: {mean_absolute_error(y_test, pred)}")
        print(f"MAPE: {mean_absolute_percentage_error(y_test, pred)}")
        return pred

    @staticmethod
    def plot_catboost(pred, y_test):
        plt.figure(figsize=(14, 7))
        plt.plot(y_test.index, y_test, label='Real')
        plt.plot(y_test.index, pred, label='Prediction')
        plt.legend()
        plt.show()

    @staticmethod
    def savemodel():
        pass

    @staticmethod
    def usemodel():
        pass

#Add Save model & Use model!

'''''
#Test
X_train, X_test, y_train, y_test = CatboostPredictor.yfdown('BTC-USD', '2020-05-24', '2024-06-02')
best_params, best_score = CatboostPredictor.searchcatboost(X_train=X_train, y_train=y_train)
pred = CatboostPredictor.catboost_model(X_train, y_train, X_test, y_test, best_params)
CatboostPredictor.plot_catboost(pred, y_test)
'''''