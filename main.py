import numpy as np
import logging

#Add Ticker selection
#Load&Save
#Yfdown should be here and used once.
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

        print("Load saved model? (Must be in same directory.)")
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
        import xg
        print("XgboostRegressor selected.")
        print("Load saved model? (Must be in same directory.)")
        selection_c = input("Y/N:    ")
        if selection_c == "Y":
            xg.loadmodel()
        elif selection_c == "N":
            start_Date = input("Start Date (YYYY-MM-DD): ")
            end_Date = input("End Date (YYYY-MM-DD): ")
            X_train, X_test, y_train, y_test, scaler_y = xg.yfdown(ticker, start_Date, end_Date)
            y_test_real, y_pred_real, model = xg.xgbst(X_train, X_test, y_train, y_test, scaler_y)
            xg.plot(ticker, y_test_real, y_pred_real)

            savemodel = input("Save model? Y/N  ")
            if savemodel == "Y":
                xg.savemodel(model)
                print("Model saved.")
            else:
                pass
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
