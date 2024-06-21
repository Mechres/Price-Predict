import tkinter as tk
from tkinter import ttk
from Kedi import CatboostPredictor
from LstmC import Lstm
from prop import MProphet
import xg
import numpy as np
import pickle

#Works only for LSTM!
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
            rmse = Lstm.yhat(model, X_test, y_test, scaler)
            print(f'RMSE: {rmse}')
            if save_var.get():
                save_model(model, scaler, model_path)
                print("Model saved.")

    # Other models will be added.


def save_model(model, scaler, filename):
    with open(filename, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)


def load_model(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        return data['model'], data['scaler']


window = tk.Tk()
window.title("Cryptocurrency Price Predictor")

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
model_var = tk.StringVar(value="1")  # Default selection
model_frame = ttk.LabelFrame(window, text="Select Model")
model_frame.grid(row=4, columnspan=2, padx=5, pady=5, sticky="w")

ttk.Radiobutton(model_frame, text="LSTM", variable=model_var, value="1").pack(anchor="w")
ttk.Radiobutton(model_frame, text="Catboost", variable=model_var, value="2").pack(anchor="w")
ttk.Radiobutton(model_frame, text="Prophet", variable=model_var, value="3").pack(anchor="w")
ttk.Radiobutton(model_frame, text="Xgboost", variable=model_var, value="4").pack(anchor="w")

# Run Button
run_button = ttk.Button(window, text="Run", command=lambda: run_selected_model(model_var.get(), ticker_entry.get()))
run_button.grid(row=5, columnspan=2, pady=10)

window.mainloop()
