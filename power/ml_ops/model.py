import pandas as pd
import datetime
import tensorflow as tf

from tensorflow.keras import models, layers, optimizers, metrics
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.callbacks import EarlyStopping



# keras models
# =============================================================================

def initialize_model(X_train, y_train, n_unit=24):

    # 1 - RNN architecture
    # ======================
    model = models.Sequential()

    ## 1.1 - Recurrent Layer
    model.add(layers.LSTM(n_unit,
                          activation='tanh',
                          return_sequences = False,
                          input_shape=(X_train.shape[1],X_train.shape[2])
                          ))
    ## 1.2 - Predictive Dense Layers
    output_length = y_train.shape[1]
    model.add(layers.Dense(output_length, activation='linear'))

    return model

def compile_model(model, learning_rate=0.02):

    # def r_squared(y_true, y_pred):
    #     ss_res = K.sum(K.square(y_true - y_pred))
    #     ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    #     return (1 - ss_res/(ss_tot + K.epsilon()))

    adam = optimizers.Adam(learning_rate=learning_rate)
    # model.compile(loss='mse', optimizer=adam, metrics=['mae', r_squared])
    model.compile(loss='mse', optimizer=adam, metrics=['mae'])

    return model

def train_model(model,
                X_train,
                y_train,
                validation_split = 0.3,
                batch_size = 32,
                epochs = 50):

    es = EarlyStopping(monitor = "val_mae",
                       mode = "min",
                       patience = 5,
                       restore_best_weights = True)

    history = model.fit(X_train, y_train,
                        validation_split=validation_split,
                        shuffle=False,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks = [es],
                        verbose = 0)
    return model, history


def evaluate_model(model,
                    X,
                    y,
                    batch_size=32
                    ):
    """ evaluate trained model """

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
            x=X,
            y=y,
            batch_size=batch_size,
            verbose=0,
            return_dict=True
        )

    loss = metrics["loss"]
    mae = metrics["mae"]

    print(f"✅ Model evaluated, MAE: {round(mae, 2)}")

    return metrics


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

    def mean_historical_power(X: pd.DataFrame, input_date: str):
            """
            takes a date as a string input
            returns the mean power produced on that day.
            Mean over the 40 years of training
            should remove 24 data points
            """
            input_date_dt = datetime.datetime.strptime(input_date, '%Y-%m-%d') + datetime.timedelta(days=1)
            #filter by month
            df_month = X[X.utc_time.dt.month == input_date_dt.month]
            #filter by day
            df_day = df_month[df_month.utc_time.dt.day == input_date_dt.day].reset_index()
            array = df_day['electricity'].groupby(df_day.utc_time.dt.hour).mean().to_numpy()
            tensor = tf.convert_to_tensor(array)
            tensor = tf.expand_dims(tensor, axis=0)
            return tensor

    model.add(layers.Lambda(mean_historical_power))
    ## Compile
    model.compile()

    adam = optimizers.Adam(learning_rate=0.02)
    model.compile(loss='mse', optimizer=adam, metrics=["mae"])

    return model


def init_baseline_keras():

    model = models.Sequential()
    # a layer to take the last value of the sequence and output it
    model.add(layers.Lambda(lambda x: x[:,-25:-1,0,None]))    # all sequences, last day, 1 feature (pv_power)


    adam = optimizers.Adam(learning_rate=0.02)
    model.compile(loss='mse', optimizer=adam, metrics=["mae"])

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

def mean_historical_power(X: pd.DataFrame, input_date: str):
    """
    takes a date as a string input
    returns the mean power produced on that day.
    Mean over the 40 years of training
    should remove 24 data points
    """
    input_date_dt = datetime.datetime.strptime(input_date, '%Y-%m-%d') + datetime.timedelta(days=1)
    #filter by month
    df_month = X[X.utc_time.dt.month == input_date_dt.month]
    #filter by day
    df_day = df_month[df_month.utc_time.dt.day == input_date_dt.day].reset_index()
    array = df_day['electricity'].groupby(df_day.utc_time.dt.hour).mean().to_numpy()
    tensor = tf.convert_to_tensor(array)
    tensor = tf.expand_dims(tensor, axis=0)
    return array
