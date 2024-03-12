import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse
from typing import Dict, List, Tuple, Sequence
from datetime import datetime

from power.params import *
from power.ml_ops.data import get_data_with_cache, load_data_to_bq, clean_pv_data
from power.ml_ops.model import initialize_model, compile_model, train_model, evaluate_model
from power.ml_ops.registry import load_model, save_model, save_results
from power.ml_ops.cross_val import get_X_y_seq

def preprocess(min_date = '1980-01-01 00:00:00',
               max_date = '2022-12-31 23:00:00') -> None:
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Process query data
    - Store processed data on your personal BQ (truncate existing table if it exists)
    - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Query raw data from BUCKET BigQuery using `get_data_with_cache`
    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.raw_pv
        ORDER BY _0
    """

    # Retrieve data using `get_data_with_cache`
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"raw_pv.csv")
    data_query = get_data_with_cache(
        query=query,
        gcp_project=GCP_PROJECT,
        cache_path=data_query_cache_path,
        data_has_header=True
    )

    # Process data
    data_clean = clean_pv_data(data_query)


    load_data_to_bq(
        data_clean,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'processed_pv',
        truncate=True
    )

    print("✅ preprocess() done \n")


def train(
        min_date = '1980-01-01 00:00:00',
        max_date = '2019-12-31 23:00:00',
        split_ratio: float = 0.02, # 0.02 represents ~ 1 month of validation data on a 2009-2015 train set
        learning_rate=0.02,
        batch_size = 32,
        patience = 5
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)


    # Load processed data using `get_data_with_cache` in chronological order
    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_pv
        ORDER BY utc_time
    """

    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_pv.csv")
    data_processed = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=True
    )

    # the processed data from bq needs to be converted to datetime object
    data_processed.utc_time = pd.to_datetime(data_processed.utc_time,utc=True)

    if data_processed.shape[0] < 240:
        print("❌ Not enough processed data retrieved to train on")
        return None

    # Split the data into training and testing sets
    train = data_processed[data_processed['utc_time'] < max_date]

    train = train[['electricity']]

    X_train, y_train = get_X_y_seq(train,
                                   number_of_sequences=10_000,
                                   input_length=48,
                                   output_length=24,
                                   gap_hours=12)


    # Train model using `model.py`
    model = load_model()

    if model is None:
        model = initialize_model(X_train, y_train, n_unit=24)

    model = compile_model(model, learning_rate=learning_rate)
    model, history = train_model(model,
                                X_train,
                                y_train,
                                validation_split = 0.3,
                                batch_size = 32,
                                epochs = 50
                                )

    val_mae = np.min(history.history['val_mae'])

    params = dict(
        context="train",
        training_set_size=f'Training data from {min_date} to {max_date}',
        row_count=len(X_train),
    )

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(mae=val_mae))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    print("✅ train() done \n")

    return val_mae


def evaluate(
        stage: str = "Production"
    ) -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return MAE as a float
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    model = load_model(stage=stage)
    assert model is not None


    # Query your BigQuery processed table and get data_processed using `get_data_with_cache`
    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_pv
        ORDER BY utc_time
    """

    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_pv.csv")
    data_processed = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=True
    )

    if data_processed.shape[0] == 0:
        print("❌ No data to evaluate on")
        return None

    # y is a sequence of 24 observations after a 12 observation gap -> 36 rows
    # X is a sequence of 48 observations before Gap & y
    X_new = data_processed[(-48 - 36): -36]
    y_new = data_processed[-24:]

    # convert X_pred to a tensorflow object
    X_new = X_new[['electricity']].to_numpy()
    X_new_tf = tf.convert_to_tensor(X_new)
    X_new_tf = tf.expand_dims(X_new_tf, axis=0)

    # convert X_pred to a tensorflow object
    y_new = y_new[['electricity']].to_numpy()
    y_new_tf = tf.convert_to_tensor(y_new)
    y_new_tf = tf.expand_dims(y_new_tf, axis=0)


    metrics_dict = evaluate_model(model=model, X=X_new_tf, y=y_new_tf)
    mae = metrics_dict["mae"]

    params = dict(
        context="evaluate", # Package behavior
        training_set_size='"40 years"',
        row_count=len(X_new)
    )

    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")

    return mae


def pred(input_pred:str = '2013-05-08 12:00:00',
         min_date = '2020-01-01 00:00:00',
         max_date = '2022-12-29 23:00:00') -> pd.DataFrame:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_pv
        ORDER BY utc_time
    """

    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_pv.csv")
    data_processed = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=True
    )

    # X_pred should be the 48 hours before the input date
    X_pred = data_processed[data_processed['utc_time'] < input_pred][-48:]

    # we have to rename columns because model is using 'power' as coulumns name
    # X_pred= X_pred.rename(columns={'electricity': 'power'})

    # convert X_pred to a tensorflow object
    X_pred = X_pred[['electricity']].to_numpy()
    X_pred_tf = tf.convert_to_tensor(X_pred)
    X_pred_tf = tf.expand_dims(X_pred_tf, axis=0)

    model = load_model()
    assert model is not None

    y_pred = model.predict(X_pred_tf)

    # y_pred dates shoud be the 24hours after a 12 hour gap
    y_pred_df = data_processed[data_processed['utc_time'] > input_pred][12:36]
    y_pred_df['pred'] = y_pred[0]

    # y_pred_df should have only two columns: 'utc_time', 'pred'; utc_time
    # should be datetime object
    # # please improve code above!
    y_pred_df = y_pred_df.drop(columns='local_time')
    y_pred_df = y_pred_df.drop(columns='electricity')
    y_pred_df.reset_index(drop=True, inplace=True)
    y_pred_df.utc_time = pd.to_datetime(y_pred_df.utc_time,utc=True)

    # Cut-off predictions that are negative or bigger than max capacity
    def cutoff_func(x):
      if x < 0.0:
        return 0
      if x > 0.9:
        return 0.9
      return x

    y_pred_df.pred = y_pred_df.pred.apply(cutoff_func)

    print(y_pred_df.pred)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")

    return y_pred_df


if __name__ == '__main__':
    preprocess(min_date = '1980-01-01 00:00:00',
               max_date = '2022-12-31 23:00:00')
    train(min_date = '1980-01-01 00:00:00',
          max_date = '2019-12-30 23:00:00')
    evaluate()
    pred('2013-05-08 12:00:00')
