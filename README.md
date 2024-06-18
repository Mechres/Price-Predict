# Bitcoin Price Prediction with Various Neural Networks
Main file let's you select model, start & end date, currently Lstm, Catboost, Prophet is usable. Will update soon.
This is a side project made in free times, models made with this project shouldn't use for anything other than experimenting.


## LSTM
Long Short-Term Memory (LSTM). It leverages historical price data from Yahoo Finance and incorporates technical indicators (SMA, EMA, RSI) as well as time-based features to improve the prediction accuracy.

## CatBoost
Gradient boosting library. It leverages historical price data from Yahoo Finance and focuses on finding the optimal model parameters through grid search for improved prediction accuracy.

## Prophet
Meta's forecasting library designed for time series with seasonality and trend components, leveraging volume as a predictor.


### Dependencies:
pip install tensorflow yfinance ta sklearn pandas numpy matplotlib catboost prophet

### - Planned Features:
- Graphical User Interface (GUI)
- Save & Load
- XGBoost Integration
- LightGBM
- Random Forest Regressor
- Gradient Boosting Regressor

Important Notes:
-Experimental Project: This is a side project for educational purposes and should not be used as the sole basis for investment decisions.

# Old Files(For Reference):
These files provide basic implementations of the models, which you can reference for understanding the underlying algorithms and principles:

- Keras-LSTM.ipynb: Illustrates the fundamentals of LSTM models in Keras.
- CatBoostRegressor.ipynb: Shows a simple CatBoostRegressor example with grid search.
- Prophet.ipynb: Demonstrates the basic usage of Facebook's Prophet library.


#Contributing:

Contributions and feedback are welcome! Feel free to open issues or submit pull requests to help improve this project.
Let me know if you'd like any other modifications!