import math
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import yfinance as yf
import matplotlib.pyplot as plt
import yaml
from typing import Tuple, Dict
import logging
import os
import joblib
import datetime

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CatBoostPredictor:
    def __init__(self, config_path: str = 'configs/catboostconfig.yaml'):
        self.model = None
        self.config = self.load_config(config_path)

    @staticmethod
    def load_config(config_path: str) -> Dict:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def yfdown(self, ticker: str, start: str, end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        df = yf.download(ticker, start=start, end=end)
        df = df[['Close']].dropna()
        df['Prev_Close'] = df['Close'].shift(1)
        df = df.dropna()
        x = df[['Prev_Close']]
        y = df['Close']
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.config['test_size'], shuffle=False)
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def search_catboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        logger.info("Grid Search Starting.")
        train_pool = Pool(X_train, y_train)

        # Perform grid search with config file.
        model = CatBoostRegressor(loss_function=self.config['loss_function'])
        grid_search_result = model.grid_search(self.config['grid_search_params'], train_pool)

        # Extract the best parameters and the RMSE values for each fold
        best_params = grid_search_result['params']
        best_score = min(grid_search_result['cv_results']['test-RMSE-mean'])

        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV score (RMSE): {best_score}")
        return best_params

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
                    best_params: Dict) -> np.ndarray:
        logger.info("Model Training Starting.")
        self.model = CatBoostRegressor(**best_params)
        self.model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
        pred = self.model.predict(X_test)
        self._print_metrics(y_test, pred)
        return pred

    @staticmethod
    def _print_metrics(y_true: pd.Series, y_pred: np.ndarray) -> None:
        mse = mean_squared_error(y_true, y_pred)
        logger.info(f"Mean Squared Error: {mse}")
        logger.info(f"RMSE: {math.sqrt(mse)}")
        logger.info(f"MAE: {mean_absolute_error(y_true, y_pred)}")
        logger.info(f"MAPE: {mean_absolute_percentage_error(y_true, y_pred)}")

    @staticmethod
    def plot_catboost(ticker: str, pred: np.ndarray, y_test: pd.Series) -> None:
        plt.figure(figsize=(14, 7))
        plt.title(f"{ticker} Actual vs. Predicted Prices - CatBoost Model")
        plt.plot(y_test.index, y_test, label='Real')
        plt.plot(y_test.index, pred, label='Prediction')
        plt.legend()
        plt.show()
        plt.savefig(os.path.join(os.getenv('PLOT_DIR', '.'), f'{ticker}_prediction_plot.png'))
        plt.close()

    def save_model(self, filename: str) -> None:
        if self.model is None:
            raise ValueError("Model hasn't been trained yet.")

        # Create the model directory if it doesn't exist
        os.makedirs(self.config['model_dir'], exist_ok=True)

        full_path = os.path.join(self.config['model_dir'], filename)
        joblib.dump(self.model, full_path)
        logger.info(f"Model saved to {full_path}")

    def load_model(self, filename: str) -> None:
        full_path = os.path.join(self.config['model_dir'], filename)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found: {full_path}")

        self.model = joblib.load(full_path)
        logger.info(f"Model loaded from {full_path}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model hasn't been trained or loaded yet.")
        return self.model.predict(X)

    def catboost_prediction(ticker):
        predictor = CatBoostPredictor()
        print("Catboost Regressor selected.")
        print("Load saved model? (Must be in same directory.)")
        selection_c = input("Y/N: ").strip().upper()

        if selection_c == "Y":
            try:
                predictor.load_model('catboost_model.joblib')
                print("Model loaded successfully.")
            except FileNotFoundError:
                print("Model file not found. Please train a new model.")
                return

        start_Date = input("Start Date (YYYY-MM-DD): ")
        end_Date = input("End Date (YYYY-MM-DD): ")

        # Validate dates
        try:
            datetime.datetime.strptime(start_Date, '%Y-%m-%d')
            datetime.datetime.strptime(end_Date, '%Y-%m-%d')
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
            return

        try:
            X_train, X_test, y_train, y_test = predictor.yfdown(ticker, start_Date, end_Date)
        except Exception as e:
            print(f"Error downloading data: {e}")
            return

        if selection_c == "Y":
            try:
                new_predictions = predictor.predict(X_test)
                predictor.plot_catboost(ticker, new_predictions, y_test)
            except Exception as e:
                print(f"Error making predictions: {e}")
        else:
            try:
                best_params = predictor.search_catboost(X_train, y_train)
                pred = predictor.train_model(X_train, y_train, X_test, y_test, best_params)
                predictor.plot_catboost(ticker, pred, y_test)

                savemodel = input("Save model? Y/N: ").strip().upper()
                if savemodel == "Y":
                    predictor.save_model('catboost_model.joblib')
                    print("Model saved successfully.")
            except Exception as e:
                print(f"Error during model training or prediction: {e}")

    def loadget(self):
        pass


"""""
# Test
if __name__ == "__main__":
    predictor = CatBoostPredictor()
    X_train, X_test, y_train, y_test = predictor.yfdown('BTC-USD', '2020-05-24', '2024-06-02')
    best_params = predictor.search_catboost(X_train, y_train)
    pred = predictor.train_model(X_train, y_train, X_test, y_test, best_params)
    predictor.plot_catboost('BTC-USD', pred, y_test)

    predictor.save_model('catboost_model.joblib')
    predictor.load_model('catboost_model.joblib')

    new_predictions = predictor.predict(X_test)
"""""
'''''
#Test 
X_train, X_test, y_train, y_test = CatboostPredictor.yfdown('BTC-USD', '2020-05-24', '2024-06-02')
best_params, best_score = CatboostPredictor.searchcatboost(X_train=X_train, y_train=y_train)
pred = CatboostPredictor.catboost_model(X_train, y_train, X_test, y_test, best_params)
CatboostPredictor.plot_catboost(pred, y_test)
'''''
