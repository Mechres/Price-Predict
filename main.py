import numpy as np
import logging

def main():
    ticker = 'BTC-USD'

    print("1.LSTM")
    print("2.Catboost")
    print("3.Prophet")
    print("4.Xgboost")
    print("5.LGBM")
    selection = input("Select Model:    ")

    if selection == "1":
        # LSTM
        from LstmC import LSTMPredictor
        LSTMPredictor.run(ticker)


    elif selection == "2":
        #Catboost
        from Kedi import CatBoostPredictor
        CatBoostPredictor.catboost_prediction(ticker)

    elif selection == "3":
        # Prophet
        from prop import MProphet
        import yaml
        print("Prophet selected.")
        # Load the config file
        config_path = 'configs/prophet_config.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        print("Load saved model? (Must be in models folder.)")
        selection_c = input("Y/N: ").upper()
        if selection_c == "Y":
            # Update config with user input
            config['start_date'] = input("Start Date (YYYY-MM-DD): ")
            config['end_date'] = input("End Date (YYYY-MM-DD): ")
            config['ticker'] = ticker
            # Save updated config
            with open(config_path, 'w') as file:
                yaml.dump(config, file)
            mp_loaded = MProphet(config_path)
            mp_loaded.load_model()
            mp_loaded.download_data()
            mp_loaded.fit_predict()
            mp_loaded.plot()


        elif selection_c == "N":
            # Update config with user input
            config['start_date'] = input("Start Date (YYYY-MM-DD): ")
            config['end_date'] = input("End Date (YYYY-MM-DD): ")
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

            savemodel = input("Save model? Y/N: ").upper()
            if savemodel == "Y":
                mp_final.save_model()
                print("Model saved.")
            else:
                pass

    elif selection == "4":
        #XGBoost
        from xg import XGBoost_Predictor
        import yaml

        config_path = "configs/Xgboost_config.yaml"

        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            config = {
                "ticker": ticker,
                "start_date": "",
                "end_date": "",
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
        print("XgboostRegressor selected.")
        print("Load saved model? (Must be in models folder.)")
        selection_c = input("Y/N: ").upper()

        if selection_c == "Y":
            model_path = "models/xgboost_model.json"
            predictor = XGBoost_Predictor(config_path)
            predictor.load_model(model_path)
            print("Model loaded successfully.")

            config['start_date'] = input("Start Date (YYYY-MM-DD): ")
            config['end_date'] = input("End Date (YYYY-MM-DD): ")
            config['ticker'] = ticker

            # Use loaded model for prediction
            df = predictor.download_data()
            X_test, _, _, _ = predictor.prepare_data(df)

            y_pred = predictor.predict(X_test)
            y_pred_real = predictor.scaler_y.inverse_transform(y_pred.reshape(-1, 1))

            # Plot results
            predictor.plot_results(df['Close'].values[-len(y_pred_real):], y_pred_real)

            print("Prediction completed. Check the plots directory for visualization.")

        elif selection_c == "N":
            config['start_date'] = input("Start Date (YYYY-MM-DD): ")
            config['end_date'] = input("End Date (YYYY-MM-DD): ")
            config['ticker'] = ticker

            # Save updated config
            with open(config_path, 'w') as file:
                yaml.dump(config, file)

            # Create and run the predictor
            predictor = XGBoost_Predictor(config_path)
            predictor.run()

            save_model = input("Save model? Y/N: ").upper()
            if save_model == "Y":
                model_filename = "models/xgboost_model.json"
                predictor.save_model(model_filename)
                print(f"Model saved to {model_filename}.")
        else:
            print("Invalid selection.")

    elif selection == "5":
        #LGBM
        from lgb import LGBMRegressorModel
        print("LGBM selected.")
        print("Load saved model? (Must be in same directory.)")
        selection_c = input("Y/N:    ")
        if selection_c == "Y":
            LGBMRegressorModel.loadmodel()
        elif selection_c == "N":
            start_Date = input("Start Date (YYYY-MM-DD): ")
            end_Date = input("End Date (YYYY-MM-DD): ")
            X_train, X_test, y_train, y_test, scaler_y = LGBMRegressorModel.yfdown(ticker, start_Date, end_Date)
            grid_search, grid_search.best_params_ = LGBMRegressorModel.grid(ticker, X_train, y_train, X_test, y_test,
                                                                            scaler_y)
            best_params = grid_search.best_params_
            model = LGBMRegressorModel.model(X_train, y_train, X_test, y_test, best_params)
            rmse = LGBMRegressorModel.yhat(ticker, model, X_test, y_test, scaler_y)
            print(f'RMSE: {rmse}')

            savemodel = input("Save model? Y/N  ")
            if savemodel == "Y":
                LGBMRegressorModel.savemodel()
            else:
                pass
    else:
        print('Incorrect Input')


if __name__ == "__main__":
    main()
