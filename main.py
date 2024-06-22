from Kedi import CatboostPredictor
from LstmC import Lstm
from prop import MProphet
from lgb import LGBMRegressorModel
import xg
import numpy as np


#Add Ticker selection
#Add GUI
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
        model_path = "lstm_model.pkl"
        print("LSTM selected.")
        print("Load saved model? (Must be in same directory.)")
        selection_c = input("Y/N:    ")
        if selection_c == "Y":
            #model, scaler = Lstm.load_model(model_path)
            pass
        elif selection_c == "N":
            start_Date = input("Start Date (YYYY-MM-DD): ")
            end_Date = input("End Date (YYYY-MM-DD): ")
            X_train, X_test, y_train, y_test, scaler = Lstm.yfdown(ticker, start_Date, end_Date)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            model = Lstm.model(X_train, y_train, X_test, y_test)
            rmse = Lstm.yhat(ticker, model, X_test, y_test, scaler)
            print(f'RMSE: {rmse}')
            savemodel = input("Save model? Y/N  ")
            if savemodel == "Y":
                Lstm.save_model(model, scaler, model_path)
                print("Model saved.")
            else:
                pass

    elif selection == "2":
        print("Catboost Regressor selected.")
        print("Load saved model? (Must be in same directory.)")
        selection_c = input("Y/N:    ")
        if selection_c == "Y":
            CatboostPredictor.usemodel()
        elif selection_c == "N":
            start_Date = input("Start Date (YYYY-MM-DD): ")
            end_Date = input("End Date (YYYY-MM-DD): ")
            X_train, X_test, y_train, y_test = CatboostPredictor().yfdown(ticker, start_Date, end_Date)
            best_params = CatboostPredictor.searchcatboost(X_train=X_train, y_train=y_train)
            pred = CatboostPredictor.catboost_model(X_train, y_train, X_test, y_test, best_params)
            CatboostPredictor.plot_catboost(ticker, pred, y_test)
            savemodel = input("Save model? Y/N  ")
            if savemodel == "Y":
                CatboostPredictor.savemodel()
            else:
                pass

        else:
            print("")
    elif selection == "3":
        #Prophet
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
        print("XgboostRegressor selected.")
        print("Load saved model? (Must be in same directory.)")
        selection_c = input("Y/N:    ")
        if selection_c == "Y":
            xg.loadmodel()
        elif selection_c == "N":
            start_Date = input("Start Date (YYYY-MM-DD): ")
            end_Date = input("End Date (YYYY-MM-DD): ")
            X_train, X_test, y_train, y_test, scaler_y = xg.yfdown(ticker, start_Date, end_Date)
            y_test_real, y_pred_real = xg.xgbst(X_train, X_test, y_train, y_test, scaler_y)
            xg.plot(ticker, y_test_real, y_pred_real)

            savemodel = input("Save model? Y/N  ")
            if savemodel == "Y":
                xg.savemodel()
            else:
                pass
    elif selection == "5":
        print("LGBM selected.")
        print("Load saved model? (Must be in same directory.)")
        selection_c = input("Y/N:    ")
        if selection_c == "Y":
            LGBMRegressorModel.loadmodel()
        elif selection_c == "N":
            start_Date = input("Start Date (YYYY-MM-DD): ")
            end_Date = input("End Date (YYYY-MM-DD): ")
            X_train, X_test, y_train, y_test, scaler_y = LGBMRegressorModel.yfdown(ticker, start_Date, end_Date)
            grid_search, grid_search.best_params_ = LGBMRegressorModel.grid(ticker, X_train, y_train, X_test, y_test, scaler_y)
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
