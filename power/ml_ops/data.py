import numpy as np
import pandas as pd
import datetime as dt
import os


def get_pv_data() -> pd.DataFrame:
    """
    Load raw data from local directory
    """
    absolute_path = os.path.dirname(
                        os.path.dirname(
                            os.path.dirname( __file__ )))
    relative_path = 'raw_data/'
    csv_path = os.path.join(absolute_path, relative_path)

    df = pd.read_csv(csv_path + '1980-2022_pv.csv')

    print('# data loaded')
    return df


def clean_pv_data(pv_df: pd.DataFrame) ->pd.DataFrame:
    """
    Remove unnecessary columns and convert to right dtypes
    """
    # remove unnevessary columns
    df = pv_df.drop(columns=['irradiance_direct','irradiance_diffuse','temperature',
                    'source','Unnamed: 0.1'])

    # convert dtypes
    df.electricity = df.electricity.astype(float)

    df.local_time = df.local_time.apply(lambda x:
                                            dt.datetime.strptime(x,
                                            "%Y-%m-%d %H:%M:%S%z")) # pd.to_datetime gives warning

    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'],
                                         unit='ms').dt.tz_localize('UTC')
    # correct column names
    df.rename(columns={'Unnamed: 0': 'utc_time'}, inplace=True)

    print('# data cleaned')
    return df
