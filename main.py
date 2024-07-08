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
        #Prophet
        from prop import MProphet
        print("Prophet selected.")
        print("Load saved model? (Must be in same directory.)")
        selection_c = input("Y/N:    ")
        if selection_c == "Y":
            #MProphet.savemodel()
            pass
        elif selection_c == "N":
            start_Date = input("Start Date (YYYY-MM-DD): ")
            end_Date = input("End Date (YYYY-MM-DD): ")
            param_grid = {
                "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
                "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
            }

            best_params = MProphet.tune_hyperparameters(ticker, start_Date, end_Date, param_grid)
            print("Best hyperparameters:", best_params)

            mp_final = MProphet(ticker, start_Date, end_Date, hyperparams=best_params)
            mp_final.download_data()
            mp_final.fit_predict()
            mp_final.plot()
            mp_final.cross_validate()

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
