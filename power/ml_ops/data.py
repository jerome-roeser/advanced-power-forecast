import numpy as np
import pandas as pd
import datetime as dt


def get_pv_data() -> pd.DataFrame:
    """
    Load raw data from local directory
    """
    df = pd.read_csv('../raw_data/1980-2022_pv.csv')
    print('# data loaded')
    return df


def clean_pv_data(pv_df: pd.DataFrame) ->pd.DataFrame:
    """
    Remove unnecessary columns and convert to right dtypes
    """
    # remove unnevessary columns
    pv_df.drop(columns=['irradiance_direct','irradiance_diffuse','temperature',
                    'source','Unnamed: 0.1'], inplace=True)

    # convert dtypes
    pv_df.electricity = pv_df.electricity.astype(float)

    pv_df.local_time    = pv_df.local_time.apply(lambda x:
                                            dt.datetime.strptime(x,
                                            "%Y-%m-%d %H:%M:%S%z")) # pd.to_datetime gives warning

    pv_df['Unnamed: 0'] = pd.to_datetime(pv_df['Unnamed: 0'],
                                         unit='ms').dt.tz_localize('UTC')
    # correct column names
    pv_df.rename(columns={'Unnamed: 0': 'utc_time'}, inplace=True)

    print('# data cleaned')
    return pv_df
