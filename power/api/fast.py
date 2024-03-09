import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from power.ml_ops.data import get_pv_data, clean_pv_data, get_data_with_cache
from power.ml_ops.model import model_yesterday
from power.ml_ops.registry import load_model
from datetime import datetime, timedelta

from pathlib import Path
from power.params import *

app = FastAPI()
# app.state.data_pv = get_pv_data()
# app.state.data_pv_clean = clean_pv_data(app.state.data_pv)
app.state.model = load_model()

data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_pv.csv")
query = f"""
    SELECT *
    FROM {GCP_PROJECT}.{BQ_DATASET}.processed_pv
    ORDER BY utc_time
"""
app.state.data_pv_clean = get_data_with_cache(
    gcp_project=GCP_PROJECT,
    query=query,
    cache_path=data_processed_cache_path,
    data_has_header=True
)

# app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# @app.get("/predict/huajie")
# def predict(
#     given_date: str,  # 2013-07-06
#     ):

#     df= pd.read_csv('raw_data/1980-2022_pv.csv')
#     given_date_converted = datetime.strptime(given_date, "%Y-%m-%d")
#     one_day_before = given_date_converted - timedelta(days=1)
#     given_date_before = one_day_before.strftime("%Y-%m-%d")
#     given_date_dt = pd.to_datetime(given_date_before)
#     df['local_time'] = pd.to_datetime(df['local_time'], utc=True)
#     selected_data = df[df['local_time'].dt.date == pd.to_datetime(given_date_dt).date()]
#     new_df = pd.DataFrame({
#         'local_time': selected_data['local_time'].dt.strftime('%Y-%m-%d %H:%M:%S'),
#         'electricity': selected_data['electricity']
#     })
#     df_dict = new_df.to_dict(orient='records')
#     converted_dict = {entry['local_time']: entry['electricity'] for entry in df_dict}

#     return converted_dict

@app.get("/predict/baseline_yesterday")
def predict_baseline_yesterday(input_date: str):
    data_pv_clean = app.state.data_pv_clean
    data = data_pv_clean[data_pv_clean['utc_time'] < input_date][-24:]
    values = data.electricity.to_list()
    return {input_date: values}

@app.get("/predict")
def predict(input_date: str, n_days=2):
    # pv_data_clean = app.state.data_pv_clean
    # input_timestamp = pd.Timestamp(input_date, tz='UTC')
    # idx = pv_data_clean[pv_data_clean.utc_time == input_timestamp].index[0]

    # n_rows = 24 * n_days
    # if idx <= n_rows:
    #     df = pv_data_clean.iloc[0:idx,:]
    # else:
    #     df = pv_data_clean.iloc[idx-n_rows:idx,:].reset_index()

    # model = app.state.model
    # assert model is not None

    # y_pred = model.predict(df)

    # predicted_data = {
    #     'utc_time':XXX.get('utc_time').tolist(),
    #     'local_time':XXX.get('local_time').tolist(),
    #     'electricity':XXX.get('electricity').tolist()
    # }

    return {'predicted_data': input_date}


# def predict(
#         pickup_datetime: str,  # 2013-07-06 17:18:00
#         pickup_longitude: float,    # -73.950655
#         pickup_latitude: float,     # 40.783282
#         dropoff_longitude: float,   # -73.984365
#         dropoff_latitude: float,    # 40.769802
#         passenger_count: int
#     ):      # 1
#     """
#     Make a single course prediction.
#     Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
#     Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
#     """

#     X_pred = pd.DataFrame(dict(
#         pickup_datetime=[pd.Timestamp(pickup_datetime, tz='US/Eastern')],
#         pickup_longitude=[pickup_longitude],
#         pickup_latitude=[pickup_latitude],
#         dropoff_longitude=[dropoff_longitude],
#         dropoff_latitude=[dropoff_latitude],
#         passenger_count=[passenger_count],
#         ))

#     model = app.state.model
#     assert model is not None

#     X_processed = preprocess_features(X_pred)
#     y_pred = float(model.predict(X_processed))

#     return  {'fare_amount': round(y_pred, 2)}


@app.get("/extract_data")
def extract_pv_data(input_date: str, n_days=10):
    pv_data_clean = app.state.data_pv_clean
    input_timestamp = pd.Timestamp(input_date, tz='UTC')
    idx = pv_data_clean[pv_data_clean.utc_time == input_timestamp].index[0]

    n_rows = 24 * n_days
    if idx <= n_rows:
        df = pv_data_clean.iloc[0:idx+24,:]
    else:
        df = pv_data_clean.iloc[idx-n_rows:idx+24,:].reset_index()

    extracted_data = {
        'utc_time':df.get('utc_time').tolist(),
        'local_time':df.get('local_time').tolist(),
        'electricity':df.get('electricity').tolist()
    }
    return {'predicted_data': input_date}

@app.get("/")
def root():

    return {'greeting': 'Hello'}
