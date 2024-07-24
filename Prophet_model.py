import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.serialize import model_to_json, model_from_json
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import logging
import os
from typing import Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml
import multiprocessing

# Suppress warnings
logging.getLogger('prophet').setLevel(logging.WARNING)


class MProphet:

    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.setup_logging()

        self.ticker = self.config['ticker']
        self.start_date = self.config['start_date']
        self.end_date = self.config['end_date']
        self.hyperparams = self.config['hyperparameters']

        self.data: Optional[pd.DataFrame] = None
        self.model: Optional[Prophet] = None
        self.forecast: Optional[pd.DataFrame] = None

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from a YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def setup_logging(self) -> None:
        """Set up logging based on the configuration."""
        logging.basicConfig(level=self.config['log_level'],
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def download_data(self) -> None:
        """Downloads price and volume data from Yahoo Finance."""
        self.logger.info(f"Downloading data for {self.ticker} from {self.start_date} to {self.end_date}")
        try:
            self.data = yf.download(self.ticker, self.start_date, self.end_date)
            self.data.reset_index(inplace=True)
            self.data = self.data[["Date", "Close", "Volume"]].rename(
                columns={"Date": "ds", "Close": "y", "Volume": "volume"}
            )
        except Exception as e:
            self.logger.error(f"Error downloading data: {str(e)}")
            raise

    def fit_predict(self) -> None:
        """Fits the Prophet model and generates forecasts."""
        if self.data is None:
            raise ValueError("Data has not been downloaded. Call download_data() first.")

        self.logger.info("Fitting Prophet model and generating forecasts")
        self.model = Prophet(interval_width=0.95, **self.hyperparams).add_regressor("volume")
        self.model.fit(self.data)
        future = self.model.make_future_dataframe(periods=self.config['forecast_periods'])
        future["volume"] = self.data["volume"].mean()  # Fill future volume with average
        self.forecast = self.model.predict(future)

    def plot(self) -> None:
        """Plots the forecast, its components, and the volume data."""
        if self.forecast is None:
            raise ValueError("Forecast has not been generated. Call fit_predict() first.")

        self.logger.info("Plotting forecast and components")
        fig1 = self.model.plot(self.forecast)
        plt.title(f"{self.ticker} Price and Volume Forecast")
        plt.savefig(os.path.join(self.config['plot_save_dir'], f'{self.ticker}_forecast.png'))
        plt.close()

        fig2 = self.model.plot_components(self.forecast)
        plt.savefig(os.path.join(self.config['plot_save_dir'], f'{self.ticker}_components.png'))
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(self.data["ds"], self.data["volume"], label="Actual Volume")
        plt.title(f"{self.ticker} Volume Data")
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.legend()
        plt.savefig(os.path.join(self.config['plot_save_dir'], f'{self.ticker}_volume.png'))
        plt.close()

    def cross_validate(self) -> pd.DataFrame:
        """Perform cross-validation and return performance metrics."""
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit_predict() first.")

        self.logger.info("Performing cross-validation")
        df_cv = cross_validation(self.model,
                                 initial=self.config['cv_initial'],
                                 period=self.config['cv_period'],
                                 horizon=self.config['cv_horizon'])
        performance = performance_metrics(df_cv)
        self.logger.info(f"Cross-validation performance metrics:\n{performance}")
        return performance

    @staticmethod
    def _evaluate_params(args: tuple) -> tuple:
        """Helper method to evaluate a set of hyperparameters."""
        config_path, params = args
        mp = MProphet(config_path)
        mp.hyperparams = params
        mp.download_data()
        mp.fit_predict()
        df_cv = cross_validation(mp.model,
                                 initial=mp.config['cv_initial'],
                                 period=mp.config['cv_period'],
                                 horizon=mp.config['cv_horizon'])
        performance = performance_metrics(df_cv)
        rmse = performance["rmse"].mean()
        return params, rmse

    @classmethod
    def tune_hyperparameters(cls, config_path: str, n_jobs: Optional[int] = None) -> Dict[str, Any]:
        """Tune hyperparameters using grid search and cross-validation with parallel processing."""
        config = cls.load_config(config_path)
        logger = logging.getLogger(__name__)
        logger.info("Tuning hyperparameters")
        best_params = None
        best_score = float("inf")

        # Determine the number of workers
        if n_jobs is None or n_jobs <= 0:
            n_jobs = multiprocessing.cpu_count()  # Use all available cores

        logger.info(f"Using {n_jobs} workers for hyperparameter tuning")

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(cls._evaluate_params, (config_path, params))
                       for params in ParameterGrid(config['param_grid'])]

            for future in as_completed(futures):
                params, rmse = future.result()
                if rmse < best_score:
                    best_score = rmse
                    best_params = params

        logger.info(f"Best hyperparameters: {best_params}")
        return best_params

    def save_model(self) -> None:
        """Save the trained model to a file."""
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit_predict() first.")

        os.makedirs(os.path.dirname(self.config['model_save_path']), exist_ok=True)
        with open(self.config['model_save_path'], "w") as f:
            f.write(model_to_json(self.model))
        self.logger.info(f"Model saved to {self.config['model_save_path']}")

    def load_model(self) -> None:
        """Load a trained model from a file."""
        if not os.path.exists(self.config['model_save_path']):
            raise FileNotFoundError(f"Model file not found: {self.config['model_save_path']}")

        with open(self.config['model_save_path'], "r") as f:
            self.model = model_from_json(f.read())
        self.logger.info(f"Model loaded from {self.config['model_save_path']}")

    def run(self) -> Dict[str, float]:
        """Run the entire modeling process."""
        self.download_data()
        self.fit_predict()
        self.plot()
        performance = self.cross_validate()
        rmse = performance["rmse"].mean()
        return {"rmse": rmse}


if __name__ == "__main__":
    config_path = 'configs/prophet_config.yaml'

    # Tune hyperparameters
    best_params = MProphet.tune_hyperparameters(config_path)
    print("Best hyperparameters:", best_params)

    # Run with best parameters
    mp = MProphet(config_path)
    mp.hyperparams = best_params
    results = mp.run()
    print(f"Final RMSE: {results['rmse']}")

    mp.save_model()

    # Load and use the saved model
    mp_loaded = MProphet(config_path)
    mp_loaded.load_model()
    mp_loaded.download_data()
    mp_loaded.fit_predict()
    mp_loaded.plot()