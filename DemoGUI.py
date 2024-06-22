import tkinter as tk
from tkinter import ttk
from Kedi import CatboostPredictor
from LstmC import Lstm
from prop import MProphet
import xg
import numpy as np
import pickle
from lgb import LGBMRegressorModel


#This is a demo and it's not completed!

def run_selected_model(selection, ticker):
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()

    if selection == "1":
        # LSTM
        model_path = "lstm_model.pkl"
        if load_var.get():
            model, scaler = load_model(model_path)
        else:
            X_train, X_test, y_train, y_test, scaler = Lstm.yfdown(ticker, start_date, end_date)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            model = Lstm.model(X_train, y_train, X_test, y_test)
            rmse = Lstm.yhat(ticker, model, X_test, y_test, scaler)
            print(f'RMSE: {rmse}')
            if save_var.get():
                save_model(model, scaler, model_path)
                print("Model saved.")

    if selection == "2":
        #Catboost
        if load_var.get():
            pass
        else:
            X_train, X_test, y_train, y_test = CatboostPredictor().yfdown(ticker, start_date, end_date)
            best_params = CatboostPredictor.searchcatboost(X_train=X_train, y_train=y_train)
            pred = CatboostPredictor.catboost_model(X_train, y_train, X_test, y_test, best_params)
            CatboostPredictor.plot_catboost(ticker, pred, y_test)
            if save_var.get():
                pass
    if selection == "3":
        #Prophet
        if load_var.get():
            pass
        else:
            param_grid = {
                "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
                "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
            }

            best_params = MProphet.tune_hyperparameters(ticker, start_date, end_date, param_grid)
            print("Best hyperparameters:", best_params)

            mp_final = MProphet(ticker, start_date, end_date, hyperparams=best_params)
            mp_final.download_data()
            mp_final.fit_predict()
            mp_final.plot()
            mp_final.cross_validate()
            if save_var.get():
                pass
    if selection == "4":
        #XGBoost
        if load_var.get():
            xg.loadmodel()
        else:
            X_train, X_test, y_train, y_test, scaler_y = xg.yfdown(ticker, start_date, end_date)
            y_test_real, y_pred_real = xg.xgbst(X_train, X_test, y_train, y_test, scaler_y)
            xg.plot(ticker, y_test_real, y_pred_real)
            if save_var.get():
                xg.savemodel()
    if selection == "5":
        if load_var.get():
            pass
        else:
            X_train, X_test, y_train, y_test, scaler_y = LGBMRegressorModel.yfdown(ticker, start_date, end_date)
            grid_search, grid_search.best_params_ = LGBMRegressorModel.grid(ticker, X_train, y_train, X_test, y_test, scaler_y)
            best_params = grid_search.best_params_
            model = LGBMRegressorModel.model(X_train, y_train, X_test, y_test, best_params)
            rmse = LGBMRegressorModel.yhat(ticker, model, X_test, y_test, scaler_y)
            print(f'RMSE: {rmse}')

def save_model(model, scaler, filename):
    with open(filename, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)


def load_model(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        return data['model'], data['scaler']


window = tk.Tk()
window.title("Cryptocurrency & Stock Price Predictor")

# Ticker Label and Entry
ticker_label = ttk.Label(window, text="Ticker:")
ticker_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

ticker_entry = ttk.Entry(window)
ticker_entry.insert(0, "BTC-USD")  # Default ticker
ticker_entry.grid(row=0, column=1, padx=5, pady=5)

# Start and End Date Labels and Entries
start_date_label = ttk.Label(window, text="Start Date (YYYY-MM-DD):")
start_date_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
start_date_entry = ttk.Entry(window)
start_date_entry.grid(row=1, column=1, padx=5, pady=5)

end_date_label = ttk.Label(window, text="End Date (YYYY-MM-DD):")
end_date_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
end_date_entry = ttk.Entry(window)
end_date_entry.grid(row=2, column=1, padx=5, pady=5)

# Load/Save
load_var = tk.BooleanVar(value=False)
load_check = ttk.Checkbutton(window, text="Load Model", variable=load_var)
load_check.grid(row=3, column=0, padx=5, pady=5, sticky="w")
save_var = tk.BooleanVar(value=False)
save_check = ttk.Checkbutton(window, text="Save Model", variable=save_var)
save_check.grid(row=3, column=1, padx=5, pady=5, sticky="w")

# Model Selection
model_var = tk.StringVar(value="1")
model_frame = ttk.LabelFrame(window, text="Select Model")
model_frame.grid(row=4, columnspan=2, padx=5, pady=5, sticky="w")

ttk.Radiobutton(model_frame, text="LSTM", variable=model_var, value="1").pack(anchor="w")
ttk.Radiobutton(model_frame, text="Catboost", variable=model_var, value="2").pack(anchor="w")
ttk.Radiobutton(model_frame, text="Prophet", variable=model_var, value="3").pack(anchor="w")
ttk.Radiobutton(model_frame, text="Xgboost", variable=model_var, value="4").pack(anchor="w")
ttk.Radiobutton(model_frame, text="LGBM", variable=model_var, value="5").pack(anchor="w")

# Run Button
run_button = ttk.Button(window, text="Run", command=lambda: run_selected_model(model_var.get(), ticker_entry.get()))
run_button.grid(row=5, columnspan=2, pady=10)

window.mainloop()
