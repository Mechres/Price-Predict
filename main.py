from Kedi import CatboostPredictor
from LstmC import Lstm
from prop import MProphet
import numpy as np


#Add Ticker selection
#Add GUI
#Load&Save

print("1.LSTM")
print("2.Catboost")
print("3.Prophet")
print("4.Xgboost")
selection = input("Select Model:    ")

if selection == "1":
    print("LSTM selected.")
    print("Load saved model? (Must be in same directory.)")
    selection_c = input("Y/N:    ")
    if selection_c == "Y":
        Lstm.load()
    elif selection_c == "N":
        start_Date = input("Start Date (YYYY-MM-DD): ")
        end_Date = input("End Date (YYYY-MM-DD): ")
        X_train, X_test, y_train, y_test, scaler = Lstm.yfdown('BTC-USD', start_Date, end_Date)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        model = Lstm.model(X_train, y_train, X_test, y_test)
        rmse = Lstm.yhat(model, X_test, y_test, scaler)
        print(f'RMSE: {rmse}')
        savemodel = input("Save model? Y/N  ")
        if savemodel == "Y":
            Lstm.save()
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
        X_train, X_test, y_train, y_test = CatboostPredictor().yfdown('BTC-USD', start_Date, end_Date)
        best_params = CatboostPredictor.searchcatboost(X_train=X_train, y_train=y_train)
        pred = CatboostPredictor.catboost_model(X_train, y_train, X_test, y_test, best_params)
        CatboostPredictor.plot_catboost(pred, y_test)
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
        pass
    elif selection_c == "N":
        start_Date = input("Start Date (YYYY-MM-DD): ")
        end_Date = input("End Date (YYYY-MM-DD): ")
        param_grid = {
            "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
            "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
        }

        best_params = MProphet.tune_hyperparameters("BTC-USD", start_Date, end_Date, param_grid)
        print("Best hyperparameters:", best_params)

        mp_final = MProphet("BTC-USD", start_Date, end_Date, hyperparams=best_params)
        mp_final.download_data()
        mp_final.fit_predict()
        mp_final.plot()
        mp_final.cross_validate()

elif selection == 4:
    pass
else:
    print('Incorrect Input')