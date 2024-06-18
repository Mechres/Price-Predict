import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import logging

# Suppress warnings
logging.getLogger('prophet').setLevel(logging.WARNING)


class MProphet:

    def __init__(self, ticker, start_date, end_date, hyperparams=None):  # Added hyperparams here
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.hyperparams = hyperparams if hyperparams else {}  # Handle empty case
        self.data = None
        self.model = None
        self.forecast = None

    def download_data(self):
        """Downloads price and volume data from Yahoo Finance."""
        self.data = yf.download(self.ticker, self.start_date, self.end_date)
        self.data.reset_index(inplace=True)
        self.data = self.data[["Date", "Close", "Volume"]].rename(
            columns={"Date": "ds", "Close": "y", "Volume": "volume"}
        )

    def fit_predict(self):
        """Fits the Prophet model and generates forecasts."""
        self.model = Prophet(
            interval_width=0.95,
            **self.hyperparams
        ).add_regressor("volume")
        self.model.fit(self.data)
        future = self.model.make_future_dataframe(periods=365)
        future["volume"] = self.data["volume"].mean()  # Fill future volume with average
        self.forecast = self.model.predict(future)

    def plot(self):
        """Plots the forecast, its components, and the volume data."""
        fig1 = self.model.plot(self.forecast)
        plt.title(f"{self.ticker} Price and Volume Forecast")
        plt.show()
        fig2 = self.model.plot_components(self.forecast)
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(self.data["ds"], self.data["volume"], label="Actual Volume")
        plt.title(f"{self.ticker} Volume Data")
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.legend()
        plt.show()

    def cross_validate(self):
        """Perform cross-validation and print performance metrics."""
        df_cv = cross_validation(
            self.model, initial="730 days", period="180 days", horizon="365 days"
        )
        performance = performance_metrics(df_cv)
        print(performance)

    def tune_hyperparameters(ticker, start_date, end_date, param_grid):
        """Tune hyperparameters using grid search and cross-validation."""
        best_params = None
        best_score = float("inf")

        for params in ParameterGrid(param_grid):
            mp = MProphet(ticker, start_date, end_date, hyperparams=params)
            mp.download_data()
            mp.fit_predict()
            df_cv = cross_validation(
                mp.model, initial="365 days", period="90 days", horizon="180 days")
            performance = performance_metrics(df_cv)
            rmse = performance["rmse"].mean()
            if rmse < best_score:
                best_score = rmse
                best_params = params

        return best_params

    def savemodel(self):
        pass

    def loadmodel(self):
        pass


'''
# Test
param_grid = {
    "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
    "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
}

best_params = MProphet.tune_hyperparameters("BTC-USD", "2018-05-22", "2024-06-17", param_grid)
print("Best hyperparameters:", best_params)

# Use best parameters for final model
mp_final = MProphet("BTC-USD", "2018-05-22", "2024-06-17", hyperparams=best_params)
mp_final.download_data()
mp_final.fit_predict()
mp_final.plot()
mp_final.cross_validate()'''
