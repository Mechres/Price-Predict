# Crypto & Stock Price Prediction with Various Neural Networks
Crypto & Stock Price Prediction and Forecasting Toolkit.

Main.py & GUI.py files let's you select model, start & end date, available models are: Lstm, Catboost, Prophet, XGBoost, LGBM and Random Forest.


### Update Log:
```
→ 24.07.2024
- File names are updated and more readable now.
- Added Random Forest Regressor.
→ 16.07.2024
LightGBM Update:
- Load & Save added.
- Docstring added.
- Code improvements.
- Config file added.
- Models & plots now save to the corresponding folder.
→ 12.07.2024
Xgboost update:
- Load & Save added.
- Bayesian Optimization added.
- TimeSeriesSplit for data splitting added.
- Config file added.
- Plot save added.
→ 09.07.2024
Prophet Update
- Save & load works
- Multiprocessing added
- Config file added
- Plots now save to plots directory
04.07.2024 →→→ 08.07.2024
- Updated Lstm, added config file.
- Config files moved to a folder.
- Model save now saves to models folder.
- Old Jupyter notebook's deleted.
- Catboost - Save & Load added
```
### Dependencies:

```
pip install tensorflow yfinance ta scikit-learn pandas numpy matplotlib catboost prophet lightgbm tkinter pickle PyYAML bayesian-optimization
```
In case of an update breaks something:
- tensorflow = 2.16.1
- keras = 3.3.3
- prophet = 1.1.5
- yfinance = 0.2.40
- ta = 0.11.0
- scikit-learn = 1.5.0
- catboost = 1.2.5
- lightgbm = 4.4.0

## LSTM
Long Short-Term Memory (LSTM). It leverages historical price data from Yahoo Finance and incorporates technical indicators (SMA, EMA, RSI) as well as time-based features to improve the prediction accuracy.

## CatBoost
Gradient boosting library. It leverages historical price data from Yahoo Finance and focuses on finding the optimal model parameters through grid search for improved prediction accuracy.

## Prophet
Meta's forecasting library designed for time series with seasonality and trend components, leveraging volume as a predictor.

## XGBoost
XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. 

## LightGBM
LightGBM is a gradient boosting framework that uses tree based learning algorithms

## Planned Features:
-  ~~Graphical User Interface (GUI)~~
-  ~~Save & Load~~
- ~~Random Forest Regressor~~
- Gradient Boosting Regressor

## Important Notes:
- This is a side project made in free times, models made with this project shouldn't be used for anything other than experimenting.
- At current level models made with this "tool" is not usable for crpyto market but feel free to modify, use, experiment on it.
- Experimental Project: This is a side project for educational purposes and should not be used as the sole basis for investment decisions.

##  ~~Old Files(For Reference):~~
 ~~These files provide basic implementations of the models, which you can reference for understanding the underlying algorithms and principles:~~

- ~~Keras-LSTM.ipynb: Illustrates the fundamentals of LSTM models in Keras.~~
- ~~CatBoostRegressor.ipynb: Shows a simple CatBoostRegressor example with grid search.~~
- ~~Prophet.ipynb: Demonstrates the basic usage of Facebook's Prophet library.~~


## Screenshots

![Screenshot1](https://i.imgur.com/P6u9Eg1.png)
![Screenshot2](https://i.imgur.com/W8ucKjX.png)
![Screenshot3](https://i.imgur.com/4Frsf3w.png)
![Screenshot4](https://i.imgur.com/p4SZXei.png)

## Contributing:

Contributions and feedback are welcome! Feel free to open issues or submit pull requests to help improve this project.
Let me know if you'd like any other modifications!
