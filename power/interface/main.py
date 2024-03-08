import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from power.params import *
from power.ml_ops.data import get_data_with_cache, clean_data, load_data_to_bq
from power.ml_ops.model import initialize_model, compile_model, train_model, evaluate_model
from power.ml_ops.registry import load_model, save_model, save_results
from power.ml_ops.registry import mlflow_run, mlflow_transition_model

def preprocess(min_date:str = '2009-01-01', max_date:str = '2015-01-01') -> None:
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
    data_clean = clean_data(data_query)


    load_data_to_bq(
        data_clean,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'processed_pv',
        truncate=True
    )

    print("✅ preprocess() done \n")

@mlflow_run
def train(
        min_date:str = '2009-01-01',
        max_date:str = '2015-01-01',
        split_ratio: float = 0.02, # 0.02 represents ~ 1 month of validation data on a 2009-2015 train set
        learning_rate=0.0005,
        batch_size = 256,
        patience = 2
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

    if data_processed.shape[0] < 10:
        print("❌ Not enough processed data retrieved to train on")
        return None

    # Create (X_train_processed, y_train, X_val_processed, y_val)
    # < MARIUS - ALI CODE HERE >

    # Train model using `model.py`
    model = load_model()

    if model is None:
        model = initialize_model(input_shape=X_train_processed.shape[1:])

    model = compile_model(model, learning_rate=learning_rate)
    model, history = train_model(
        model, X_train_processed, y_train,
        batch_size=batch_size,
        patience=patience,
        validation_data=(X_val_processed, y_val)
    )

    val_mae = np.min(history.history['val_mae'])

    params = dict(
        context="train",
        training_set_size=DATA_SIZE,
        row_count=len(X_train_processed),
    )

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(mae=val_mae))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    print("✅ train() done \n")

    return val_mae


@mlflow_run
def evaluate(
        min_date:str = '2014-01-01',
        max_date:str = '2015-01-01',
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

    data_processed = data_processed.to_numpy()

    X_new = data_processed[:, :-1]
    y_new = data_processed[:, -1]

    metrics_dict = evaluate_model(model=model, X=X_new, y=y_new)
    mae = metrics_dict["mae"]

    params = dict(
        context="evaluate", # Package behavior
        training_set_size=DATA_SIZE,
        row_count=len(X_new)
    )

    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")

    return mae


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    if X_pred is None:
        X_pred = pd.DataFrame(dict(
        pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz='UTC')],
        pickup_longitude=[-73.950655],
        pickup_latitude=[40.783282],
        dropoff_longitude=[-73.984365],
        dropoff_latitude=[40.769802],
        passenger_count=[1],
    ))

    model = load_model()
    assert model is not None

    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred


if __name__ == '__main__':
    preprocess(min_date='2009-01-01', max_date='2015-01-01')
    train(min_date='2009-01-01', max_date='2015-01-01')
    evaluate(min_date='2009-01-01', max_date='2015-01-01')
    pred()
