import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from power.ml_ops.data import get_pv_data, clean_pv_data
from power.ml_ops.model import model_yesterday
from datetime import datetime, timedelta

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(
    given_date: str,  # 2013-07-06
    ):

    df= pd.read_csv('raw_data/1980-2022_pv.csv')
    given_date_converted = datetime.strptime(given_date, "%Y-%m-%d")
    one_day_before = given_date_converted - timedelta(days=1)
    given_date_before = one_day_before.strftime("%Y-%m-%d")
    given_date_dt = pd.to_datetime(given_date_before)
    df['local_time'] = pd.to_datetime(df['local_time'], utc=True)
    selected_data = df[df['local_time'].dt.date == pd.to_datetime(given_date_dt).date()]
    new_df = pd.DataFrame({
        'local_time': selected_data['local_time'].dt.strftime('%Y-%m-%d %H:%M:%S'),
        'electricity': selected_data['electricity']
    })
    df_dict = new_df.to_dict(orient='records')
    converted_dict = {entry['local_time']: entry['electricity'] for entry in df_dict}

    return converted_dict

@app.get("/predict/previous_value")
def predict_previous_value(input_date: str):
    pv_data = get_pv_data()
    pv_data_clean = clean_pv_data(pv_data)
    yesterday_baseline = model_yesterday(pv_data_clean, input_date)
    values = yesterday_baseline.get('electricity').to_list()
    return {input_date: values}


@app.get("/")
def root():

    return {'greeting': 'Hello'}
