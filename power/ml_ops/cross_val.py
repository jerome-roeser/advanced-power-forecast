
# importing ###################################################################

# Data manipulation
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

# Manipulating temporal data and check the types of variables
from typing import Dict, List, Tuple, Sequence
from power.ml_ops.data import get_weather_forecast_features
from power.ml_ops.model import mean_historical_power


# Creating FOLDs ##############################################################

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

    TARGET = 'electricity'
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
    pv_fold:pd.DataFrame,
    forecast_fold:pd.DataFrame,
    input_length:int,       # 48
    output_length:int,      # 24
    gap_hours):
    '''
    - given a fold, it returns one sequence (X_i, y_i)
    - with the starting point of the sequence being chosen at random
    - TARGET is the variable(s) we want to predict (name of the column(s))
    '''
    TARGET = 'electricity'
    first_possible_start = 0
    last_possible_start = len(pv_fold) - (input_length + gap_hours + output_length) + 1

    random_start = np.random.randint(first_possible_start, last_possible_start)

    # input_start & input_end are the indexes of the 48h (training length) of training data
    # thus for weather forecast data from input_end should be used to add
    # the weather forecast features
    input_start = random_start
    input_end = random_start + input_length
    # here we extract the forecast date and hour
    forecast_date = pv_fold.iloc[input_end]['utc_time'].strftime('%Y-%m-%d')
    forecast_hour = pv_fold.iloc[input_end]['utc_time'].hour

    target_start = input_end + gap_hours
    target_end = target_start + output_length

    # first we parse the electricity feature
    # need to reset index only for X_i in order to be able to concat with X_weather later on
    X_i = pv_fold.iloc[input_start:input_end].reset_index()
    y_i = pv_fold.iloc[target_start:target_end][[TARGET]]    # creates a pd.DataFrame for the target y

    # then we parse/create the weather forecast features
    X_weather = get_weather_forecast_features(forecast_fold, forecast_date)

    #
    features = ['utc_time', 'prediction_utc_time','temperature', 'clouds', 'wind_speed']
    to_concat = [X_i[['utc_time','electricity']], X_weather[features]]
    X_i = pd.concat(to_concat, axis=1)
    return (X_i, y_i)

def get_X_y_seq(
    pv_fold:pd.DataFrame,
    forecast_fold:pd.DataFrame,
    number_of_sequences:int,
    input_length:int,
    output_length:int,
    gap_hours=0):
    '''
    Given a fold, it creates a series of sequences randomly
    as many as being specified
    '''

    X, y = [], []                                                 # lists for the sequences for X and y

    for i in range(number_of_sequences):
        (Xi, yi) = get_Xi_yi(pv_fold, forecast_fold, input_length, output_length, gap_hours)   # calls the previous function to generate sequences X + y
        X.append(Xi)
        y.append(yi)

    return np.array(X), np.array(y)


def get_Xi_yi_mean_baseline(
    fold:pd.DataFrame,
    input_length:int,       # 48
    output_length:int,      # 24
    gap_hours,
    processed_pv: pd.DataFrame):
    '''
    - given a fold, it returns one sequence (X_i, y_i)
    - with the starting point of the sequence being chosen at random
    - TARGET is the variable(s) we want to predict (name of the column(s))
    '''
    TARGET = 'electricity'
    first_possible_start = 0
    last_possible_start = len(fold) - (input_length + gap_hours + output_length) + 1

    random_start = np.random.randint(first_possible_start, last_possible_start)

    input_start = random_start
    input_end = random_start + input_length
    target_start = input_end + gap_hours
    target_end = target_start + output_length

    X_i = fold.iloc[input_start:input_end]
    y_i = fold.iloc[target_start:target_end][[TARGET]]
    input_date = fold.iloc[target_start:target_end].utc_time[0][:10]
    y_i['electricity'] = mean_historical_power(processed_pv, input_date)

    return (X_i, y_i)

def get_X_y_seq_mean_baseline(
    fold:pd.DataFrame,
    number_of_sequences:int,
    input_length:int,
    output_length:int,
    gap_hours=0):
    '''
    Given a fold, it creates a series of sequences randomly
    as many as being specified
    '''

    X, y = [], []                                                 # lists for the sequences for X and y

    for i in range(number_of_sequences):
        (Xi, yi) = get_Xi_yi(fold, input_length, output_length, gap_hours)   # calls the previous function to generate sequences X + y
        X.append(Xi)
        y.append(yi)

    return np.array(X), np.array(y)
