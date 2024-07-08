from tensorflow.keras.optimizers import Adam

class Config:
    # Data parameters
    TICKER = "BTC-USD"
    START_DATE = "2018-01-01"
    END_DATE = "2024-07-08"

    # Model parameters
    LSTM_UNITS = [128, 64]
    DENSE_UNITS = [32]
    DROPOUT_RATE = 0.2
    LEARNING_RATE = 1e-3
    OPTIMIZER = Adam(learning_rate=LEARNING_RATE)

    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    VALIDATION_SPLIT = 0.2

    # File paths
    MODEL_SAVE_PATH = "models/lstm_model.pkl"

    # Feature columns
    FEATURE_COLUMNS = ['Prev_Close', 'Prev_SMA_20', 'Prev_EMA_12', 'Prev_RSI', 'Day_of_Week']
    TARGET_COLUMN = 'Close'

    # Technical indicator parameters
    SMA_WINDOW = 20
    EMA_WINDOW = 12
    RSI_WINDOW = 14