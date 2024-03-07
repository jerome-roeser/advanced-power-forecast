
# fold_length
# fold_stride
# train_test_ratio
# output_length
# n_targets
# sequence_stride (option1 for sequencing)

# input_length
# n_features



# importing ###################################################################

# Data manipulation
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

# Data Retrieval
from power.ml_ops.data import clean_pv_data, get_pv_data

# System
import os

# Manipulating temporal data and check the types of variables
from typing import Dict, List, Tuple, Sequence

# Early stopping
from tensorflow.keras.callbacks import EarlyStopping


# Creating FOLDs for cross validation #########################################

def get_folds(
    df: pd.DataFrame,
    fold_length: int,
    fold_stride: int) -> List[pd.DataFrame]:
    '''
    This function slides through the Time Series dataframe of shape (n_timesteps, n_features) to create folds
    - of equal `fold_length`
    - using `fold_stride` between each fold

    Returns a list of folds, each as a DataFrame
    '''
    folds = []
    for idx in range(0, len(df), fold_stride):
        if (idx + fold_length) > len(df):
            break
        fold = df.iloc[idx:idx + fold_length, :]  # select from row idx til last row of the fold (6 years), all the columns
        folds.append(fold)                        # append the 6 year fold to folds
    return folds



# Splitting folds #############################################################

def train_test_split(fold:pd.DataFrame,
                     train_test_ratio: float,
                     input_length: int) -> Tuple[pd.DataFrame]:
    '''
    Returns a train dataframe and a test dataframe (fold_train, fold_test)
    from which one can sample (X,y) sequences.
    df_train should contain all the timesteps until round(train_test_ratio * len(fold))
    '''
    # TRAIN SET
    # ======================
    last_train_idx = round(train_test_ratio * len(fold))    # 83% of the fold for train
    fold_train = fold.iloc[0:last_train_idx, :]             # 1st until last row of train set, all columns

    # TEST SET
    # ======================
    first_test_idx = last_train_idx - input_length          # last row of train set - 2 weeks --> test set starts 2 weeks
                                                            # before train set ends --> overlap (not a problem with X)
    fold_test = fold.iloc[first_test_idx:, :]               # 1st until last row of test set, all columns

    return (fold_train, fold_test)



# Craeting sequences: Option 1 (using strides) #################################

def get_X_y_strides(fold: pd.DataFrame, input_length: int, output_length: int, sequence_stride: int):
    '''
    - slides through a `fold` Time Series (2D array) to create sequences of equal
        * `input_length` for X,
        * `output_length` for y,
      using a temporal gap `sequence_stride` between each sequence
    - returns a list of sequences, each as a 2D-array time series
    '''

    X, y = [], []

    for i in range(0, len(fold), sequence_stride):
        # Exits the loop as soon as the last fold index would exceed the last index
        if (i + input_length + output_length) >= len(fold):
            break
        X_i = fold.iloc[i:i + input_length, :]
        y_i = fold.iloc[i + input_length:i + input_length + output_length, :][[TARGET]] # index + length of sequence until index + length of seq. + length of target
        X.append(X_i)
        y.append(y_i)

    return np.array(X), np.array(y)



# Craeting sequences: Option 2 (random sampling) ###############################

def get_Xi_yi(
    fold:pd.DataFrame,
    input_length:int,       # 48
    output_length:int):     # 24
    '''
    - given a fold, it returns one sequence (X_i, y_i)
    - with the starting point of the sequence being chosen at random
    '''
    first_possible_start = 0                                                    # the +1 accounts for the index, that is exclusive.
    last_possible_start = len(fold) - (input_length + output_length) + 1        # It can start as long as there are still
                                                                                # 48 + 1 days after the 1st day.
    random_start = np.random.randint(first_possible_start, last_possible_start) # np.random to pick a day inside the possible interval.

    X_i = fold.iloc[random_start:random_start+input_length]
    y_i = fold.iloc[random_start+input_length:
                  random_start+input_length+output_length][[TARGET]]            # creates a pd.DataFrame for the target y

    return (X_i, y_i)

def get_X_y(
    fold:pd.DataFrame,
    number_of_sequences:int,
    input_length:int,
    output_length:int):

    X, y = [], []                                                 # lists for the sequences for X and y

    for i in range(number_of_sequences):
        (Xi, yi) = get_Xi_yi(fold, input_length, output_length)   # calls the previous function to generate sequences X + y
        X.append(Xi)
        y.append(yi)

    return np.array(X), np.array(y)



# Cross validation ############################################################

def cross_validation(cross_val_dict):
    '''
    This function cross-validates
    - the "last seen value" baseline model
    - the RNN model

    input parameters:
    '''

    # fold_length
    # fold_stride
    # train_test_ratio
    # output_length
    # n_targets
    # sequence_stride (option1 for sequencing)

    if cross_val_dict == {}:
        cross_val_dict = {
            'fold_length': 52560,           # 6 years
            'fold_stride': 52560,           # 6 years
            'train_test_ratio': 0.83,       # 5 yrs/6 yrs
            'output_length': 24,            # Day-ahead predictions
            'sequence_stride': 72,          # 3 days
            'n_seq_train': 250,             # number_of_sequences_train
            'n_seq_test': 50                # number_of_sequences_test
        }

    fold_length = cross_val_dict['fold_length']
    fold_stride = cross_val_dict['fold_stride']
    train_test_ratio = cross_val_dict['train_test_ratio']
    output_length = cross_val_dict['output_length']
    sequence_stride = cross_val_dict['sequence_stride']
    n_seq_train = cross_val_dict['n_seq_train']
    n_seq_test = cross_val_dict['n_seq_test']


    list_of_mae_baseline_model = []
    list_of_mae_recurrent_model = []

    # 0 - Creating folds
    # =========================================
    folds = get_folds(pv_df, fold_length, fold_stride)  # function we coded to get the folds

    for fold_id, fold in enumerate(folds):

        # 1 - Train/Test split the current fold
        # =========================================
        (fold_train, fold_test) = train_test_split(fold, train_test_ratio, INPUT_LENGTH) # function we coded to split train/test

        n_seq_train = 250   # number_of_sequences_train
        n_seq_test =  50    # number_of_sequences_test

        X_train, y_train = get_X_y(fold_train, n_seq_train, INPUT_LENGTH, output_length)
        X_test, y_test = get_X_y(fold_test, n_seq_test, INPUT_LENGTH, output_length)

        X_train = np.delete(X_train, [0, 1], 2)
        X_test = np.delete(X_test, [0, 1], 2)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        # 2 - Modelling
        # =========================================

        ##### Baseline Model
        baseline_model = init_baseline()
        mae_baseline = baseline_model.evaluate(X_test, y_test, verbose=0)[1]   # evaluating baseline model (metric)
        list_of_mae_baseline_model.append(mae_baseline)
        print("-"*50)
        print(f"MAE baseline fold n°{fold_id} = {round(mae_baseline, 2)}")

        ##### LSTM Model
        model = init_model(X_train, y_train)
        es = EarlyStopping(monitor = "val_mae",
                           mode = "min",
                           patience = 5,
                           restore_best_weights = True)
        history = model.fit(X_train, y_train,
                            validation_split = 0.3,
                            shuffle = False,
                            batch_size = 32,
                            epochs = 50,
                            callbacks = [es],
                            verbose = 0)
        res = model.evaluate(X_test, y_test, verbose=0)    # evaluating LSTM (metric)
        mae_lstm = res[1]
        list_of_mae_recurrent_model.append(mae_lstm)
        print(f"MAE LSTM fold n°{fold_id} = {round(mae_lstm, 2)}")

        ##### Comparison LSTM vs Baseline for the current fold
        print(f"improvement over baseline: {round((1 - (mae_lstm/mae_baseline))*100,2)} % \n")

    return list_of_mae_baseline_model, list_of_mae_recurrent_model
