from Kedi import CatboostPredictor
from LstmC import Lstm
selection_c = ""
#
print("1.LSTM")
print("2.Catboost")
print("3.Xgboost")
print("4.Xgboost")
selection = input("Select Model:    ")

if selection == "1":
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
elif selection == 3:
    pass
elif selection == 4:
    pass
else:
    print('Incorrect Input')