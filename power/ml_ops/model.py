import pandas as pd
import datetime
import tensorflow as tf

from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.callbacks import EarlyStopping



# keras models
# =============================================================================


def initialize_model(X_train, y_train, n_unit=24):


    # 1 - RNN architecture
    # ======================
    output_length = y_train.shape[1]

    normalizer = layers.Normalization() # Instantiate a "normalizer" layer
    normalizer.adapt(X_train) # "Fit" it on the train set

    model = models.Sequential([
        ## 1.0 - Input Layer
        layers.Input(shape=(X_train.shape[1],X_train.shape[2])),
        ## 1.1 - Normalization Layer
        normalizer,
        ## 1.2 - Recurrent Layer
        layers.LSTM(units=24, activation='tanh', return_sequences = True),
        layers.LSTM(units=24, activation='tanh', return_sequences = False),
        ## 1.3 Dense Layer and Dropout
        layers.Dense(units=16, activation='relu'),
        layers.Dropout(0.5),
        ## 1.4 - Predictive Dense Layers
        layers.Dense(output_length, activation='linear'),
        ])

    return model

def compile_model(model, learning_rate=0.02):

    # def r_squared(y_true, y_pred):
    #     ss_res = K.sum(K.square(y_true - y_pred))
    #     ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    #     return (1 - ss_res/(ss_tot + K.epsilon()))

    adam = optimizers.Adam(learning_rate=learning_rate)
    # model.compile(loss='mse', optimizer=adam, metrics=['mae', r_squared])
    model.compile(loss='mse', optimizer=adam, metrics=['mae'])
    # model.compile(loss='mse', optimizer=adam, metrics=[MeanAbsoluteError(),
    #                                                    MeanSquaredError(),
    #                                                   ])

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
