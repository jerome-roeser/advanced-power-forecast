import pandas as pd

from tensorflow.keras import models, layers, optimizers, metrics
from tensorflow.keras.layers import Lambda   



# keras models
# =============================================================================

def init_RNN(X_train, y_train):

    # 1 - RNN architecture
    # ======================
    model = models.Sequential()

    ## 1.1 - Recurrent Layer
    model.add(layers.LSTM(24,
                          activation='tanh',
                          return_sequences = False,
                          input_shape=(X_train.shape[1],X_train.shape[2])
                          ))
    ## 1.2 - Predictive Dense Layers
    output_length = y_train.shape[1]
    model.add(layers.Dense(output_length, activation='linear'))

    # 2 - Compiler
    # ======================
    adam = optimizers.Adam(learning_rate=0.02)
    model.compile(loss='mse', optimizer=adam, metrics=["mae"])

    return model


def init_baseline_yesterday():

    ## architecture
    model = models.Sequential()
    model.add(layers.Lambda(lambda x: x[:,-25:-1,0,None]))  # all sequences, last day, 1 feature (pv_power)

    ## Compile
    adam = optimizers.Adam(learning_rate=0.02)
    model.compile(loss='mse', optimizer=adam, metrics=["mae"])

    return model


def init_baseline_mean():

    ## architecture
    model = models.Sequential()

    ## Compile
    model.compile()

    return model

# function models
# =============================================================================





def model_yesterday(X: pd.DataFrame, input_date: str) -> pd.DataFrame:
    """
    Returns a simple previous day model
    Input:
     - a clean DataFrame
     - a date with format: "YEAR-MONTH-DAY HOUR:MIN:SECONDS"
    Returns:
     - A dataFrame with the power production from the previous day
    """
    input_timestamp = pd.Timestamp(input_date, tz='UTC')
    idx = X[X.utc_time == input_timestamp].index[0]
    if idx <= 24:
        return X.iloc[0:idx,:]
    return X.iloc[idx-24:idx,:]
