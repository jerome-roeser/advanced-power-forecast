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

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/extract_data")
def extract_pv_data(input_date: str, n_days=10):
    pv_data_clean = app.state.data_pv_clean
    n_rows = 24 * n_days
    days_before = pv_data_clean[pv_data_clean['utc_time'] < input_date] \
                                        ['electricity'][-n_rows:].to_list()
    day_after = pv_data_clean[pv_data_clean['utc_time'] >= input_date] \
                                        ['electricity'][:24].to_list()

    extracted_data = {
        'days_before':days_before,
        'day_after':day_after
        }
    return {input_date: extracted_data}

@app.get("/predict_baseline_yesterday")
def predict_baseline_yesterday(input_date: str):
    data_pv_clean = app.state.data_pv_clean
    data = data_pv_clean[data_pv_clean['utc_time'] < input_date][-24:]
    values = data.electricity.to_list()
    return {input_date: values}

@app.get("/predict")
def predict(input_date: str, n_days=2):
    pv_data_clean = app.state.data_pv_clean
    X_pred = pv_data_clean[pv_data_clean['utc_time'] < input_date][-48:]

    # model = app.state.model
    # assert model is not None

    # y_pred = model.predict(df)

    predicted_data = {
        'days_before':X_pred.electricity.to_list(),
        'day_after':'y_pred'
        }

    return {f'dataframe to predict': X_pred.electricity.to_list()}


@app.get("/")
def root():

    return {'greeting': 'Hello'}

#################### OLD ENDPOINT   ########################

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
