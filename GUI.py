import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pickle
from Kedi import CatBoostPredictor
from lgbm_model import LGBMRegressorModel


def catboost_prediction(ticker, start_date, end_date, load_var, save_var):
    predictor = CatBoostPredictor()

    try:
        if load_var.get():
            try:
                predictor.load_model('catboost_model.joblib')
                messagebox.showinfo("Info", "Model loaded successfully.")
            except FileNotFoundError:
                messagebox.showerror("Error", "Model file not found. Please train a new model.")
                return

            X_train, X_test, y_train, y_test = predictor.yfdown(ticker, start_date, end_date)
            new_predictions = predictor.predict(X_test)
            predictor.plot_catboost(ticker, new_predictions, y_test)
        else:
            X_train, X_test, y_train, y_test = predictor.yfdown(ticker, start_date, end_date)
            best_params = predictor.search_catboost(X_train, y_train)
            pred = predictor.train_model(X_train, y_train, X_test, y_test, best_params)
            predictor.plot_catboost(ticker, pred, y_test)

            if save_var.get():
                predictor.save_model('catboost_model.joblib')
                messagebox.showinfo("Info", "Model saved successfully.")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


def lstm_prediction(ticker, start_date, end_date, load_var, save_var):
    from LstmC import LSTMPredictor
    from configs.LstmConfig import Config
    try:
        if load_var.get():
            try:
                predictor = LSTMPredictor.load_model(Config.MODEL_SAVE_PATH)
                messagebox.showinfo("Info", "Model loaded successfully.")
            except FileNotFoundError:
                messagebox.showerror("Error", "Model file not found. Please train a new model.")
                return

            df = predictor.yf_Down(ticker, start_date, end_date)
            X_train, X_test, y_train, y_test = predictor.prepare_data(df)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            y_pred = predictor.predict(X_test)
            y_true = predictor.scaler_y.inverse_transform(y_test)
            rmse = predictor.evaluate_model(y_true, y_pred)
            messagebox.showinfo("RMSE", f'RMSE: {rmse}')
            predictor.plot_results(y_true, y_pred, ticker)
        else:
            predictor = LSTMPredictor()
            Config.TICKER = ticker
            df = predictor.yf_Down(Config.TICKER, Config.START_DATE, Config.END_DATE)
            X_train, X_test, y_train, y_test = predictor.prepare_data(df)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            history = predictor.train_model(X_train, y_train, X_test, y_test)
            y_pred = predictor.predict(X_test)
            y_true = predictor.scaler_y.inverse_transform(y_test)
            rmse = predictor.evaluate_model(y_true, y_pred)
            messagebox.showinfo("RMSE", f'RMSE: {rmse}')
            predictor.plot_results(y_true, y_pred, Config.TICKER)

            if save_var.get():
                predictor.save_model(Config.MODEL_SAVE_PATH)
                messagebox.showinfo("Saved!", "Model saved successfully.")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


def MetaProphet(ticker, start_date, end_date, load_var, save_var):
    # Prophet
    from prop import MProphet
    import yaml
    try:
        # Load the config file
        config_path = 'configs/prophet_config.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        messagebox.showerror("Error", "Config file not found.")

    try:

        if load_var.get():
            try:
                # Update config with user input
                config['start_date'] = start_date
                config['end_date'] = end_date
                config['ticker'] = ticker
                # Save updated config
                with open(config_path, 'w') as file:
                    yaml.dump(config, file)
                mp_loaded = MProphet(config_path)
                mp_loaded.load_model()
                messagebox.showinfo("Info", "Model loaded successfully.")
            except FileNotFoundError:
                messagebox.showerror("Error", "Model file not found. Please train a new model.")
                return
            mp_loaded.download_data()
            mp_loaded.fit_predict()
            mp_loaded.plot()
            messagebox.showinfo("Info", "Plots saved to '/plots' directory.")

        else:
            # Update config with user input
            config['start_date'] = start_date
            config['end_date'] = end_date
            config['ticker'] = ticker
            # Save updated config
            with open(config_path, 'w') as file:
                yaml.dump(config, file)
            best_params = MProphet.tune_hyperparameters(config_path)
            print("Best hyperparameters:", best_params)
            mp_final = MProphet(config_path)
            mp_final.hyperparams = best_params
            results = mp_final.run()
            print(f"Final RMSE: {results['rmse']}")
            messagebox.showinfo("Info", "Plots saved to '/plots' directory.")

            if save_var.get():
                mp_final.save_model()
                messagebox.showinfo("Info", "Model saved successfully.")
            else:
                pass
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


def xgboost_prediction(ticker, start_date, end_date, load_var, save_var):
    from xg import XGBoost_Predictor
    import yaml
    config_path = "configs/Xgboost_config.yaml"

    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        config = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "test_size": 0.2,
            "plot_dir": "plots",
            "hyperparameter_tuning": {
                "max_depth": {"min": 3, "max": 10},
                "n_estimators": {"min": 100, "max": 500},
                "learning_rate": {"min": 0.01, "max": 0.3},
                "subsample": {"min": 0.8, "max": 1.0},
                "colsample_bytree": {"min": 0.8, "max": 1.0},
                "n_iter": 20
            }
        }

    config['start_date'] = start_date
    config['end_date'] = end_date
    config['ticker'] = ticker

    # Save updated config
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    predictor = XGBoost_Predictor(config_path)

    try:
        if load_var.get():
            model_path = "models/xgboost_model.json"
            try:
                predictor.load_model(model_path)
                messagebox.showinfo("Info", "Model loaded successfully.")
            except FileNotFoundError:
                messagebox.showerror("Error", "Model file not found. Please train a new model.")
                return

            df = predictor.download_data()
            X_test, _, _, _ = predictor.prepare_data(df)

            y_pred = predictor.predict(X_test)
            y_pred_real = predictor.scaler_y.inverse_transform(y_pred.reshape(-1, 1))

            predictor.plot_results(df['Close'].values[-len(y_pred_real):], y_pred_real)

            messagebox.showinfo("Info", "Prediction completed. Check the plots directory for visualization.")
        else:
            predictor.run()

            if save_var.get():
                model_filename = "models/xgboost_model.json"
                predictor.save_model(model_filename)
                messagebox.showinfo("Info", f"Model saved to {model_filename}.")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


def run_lgbm(ticker, start_date, end_date, load_var, save_var):
    lgbm_model = LGBMRegressorModel()

    if load_var.get():
        try:
            model = lgbm_model.load_model(ticker)
            rmse, dates, y_true, y_pred = lgbm_model.predict_new_data(model, ticker, start_date, end_date)
            messagebox.showinfo("LGBM Results",
                                f'RMSE on new data: {rmse}\n\nPredictions saved in plots/{ticker}_prediction_new_data.png')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model or predict: {str(e)}")
    else:
        try:
            model, rmse = lgbm_model.run(ticker, start_date, end_date)
            messagebox.showinfo("LGBM Results", f'RMSE: {rmse}')

            if save_var.get():
                lgbm_model.save_model(model, ticker)
                messagebox.showinfo("Save Successful", f"Model for {ticker} saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run LGBM model: {str(e)}")


def run_selected_model(selection, ticker):
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()

    if selection == "1":
        # LSTM
        lstm_prediction(ticker, start_date, end_date, load_var, save_var)

    if selection == "2":
        #Catboost
        catboost_prediction(ticker, start_date, end_date, load_var, save_var)

    if selection == "3":
        #Prophet
        MetaProphet(ticker, start_date, end_date, load_var, save_var)

    if selection == "4":
        #XGBoost
        xgboost_prediction(ticker, start_date, end_date, load_var, save_var)

    if selection == "5":
        #LGBM
        run_lgbm(ticker, start_date, end_date, load_var, save_var)


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
