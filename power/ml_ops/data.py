import os
import datetime as dt
import pandas as pd

from colorama import Fore, Style
from google.cloud import bigquery
from pathlib import Path

from power.params import *


def clean_pv_data(pv_df: pd.DataFrame) ->pd.DataFrame:
    """
    Remove unnecessary columns and convert to right dtypes
    """
    # remove unnecessary columns
    df = pv_df.drop(columns=[
                    'irradiance_direct',
                    'irradiance_diffuse',
                    'temperature',
                    'source',
                    '_0-1'])

    # convert dtypes
    df.electricity = df.electricity.astype(float)


    # pd.to_datetime gives warning
    df.local_time = df.local_time.apply(lambda x:
                                            dt.datetime.strptime(x,
                                            "%Y-%m-%d %H:%M:%S%z"))

    df['_0'] = pd.to_datetime(df['_0'], unit='ms').dt.tz_localize('UTC')

    # correct column names
    df.rename(columns={'_0': 'utc_time'}, inplace=True)

    print('# data cleaned')
    return df


def get_data_with_cache(
        gcp_project:str,
        query:str,
        cache_path:Path,
        data_has_header=True
    ) -> pd.DataFrame:
    """
    Retrieve `query` data from BigQuery, or from `cache_path` if the file exists
    Store at `cache_path` if retrieved from BigQuery for future use
    """
    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)
    else:
        print(Fore.BLUE + "\nLoad data from BigQuery server..." + Style.RESET_ALL)
        client = bigquery.Client(project=gcp_project)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        # Store as CSV if the BQ query returned at least one valid line
        print(f"cache_path: {cache_path}")
        print(f"header: {data_has_header}")
        print(f"index: {False}")
        if df.shape[0] > 1:
            df.to_csv(cache_path, header=data_has_header, index=False)
    print(f"cache_path: {cache_path}")
    print(f"header: {data_has_header}")
    print(f"index: {False}")
    print(f"✅ Data loaded, with shape {df.shape}")

    return df

def load_data_to_bq(
        data: pd.DataFrame,
        gcp_project:str,
        bq_dataset:str,
        table: str,
        truncate: bool
    ) -> None:
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """

    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(Fore.BLUE + f"\nSave data to BigQuery @ {full_table_name}...:" + Style.RESET_ALL)

    # Load data onto full_table_name
    client = bigquery.Client()

    # Define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    # Load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")

# probably obsolete now ....
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


def select_years(df: pd.DataFrame, start=1980, end=1980)-> pd.DataFrame:
    """
    Select a subset of the cleaned data to process it further. Use this function
    to split into test set and train+validation set.
    Input:
      - cleaned dataframe from the raw data
      - start year (inclusive)
      - end year (inclusive)
    Output:
      - df between start and end year
    """
    start_point = f"{start}-01-01 00:00:00"
    end_point   = f"{end}-12-31 23:00:00"
    years_df = df[df.utc_time.between(start_point, end_point)]

    n_years = years_df['utc_time'].dt.year.nunique()
    print(f"# selected {n_years} years from {start} to {end}")

    return years_df


def get_stats_table(
  years_df: pd.DataFrame,
  capacity: False) -> pd.DataFrame:
  """
  Creates a table with statistics for electricity and optional capacity factor
  for every hour of the year (8784).
  Input:
    - Cleaned df that contains at least electricity
    as column. The df should span several years, because every
    year is one sample for the statictics.
    - Optional flag for capacity factor
  Output:
    - df with 8784 hours of the years (including leap years) as rows. The df has
    multilevel index because statistics are returned for electricity and
    capacity factor.
  """
  years_df['hour_of_year'] = years_df.utc_time.\
                           apply(lambda x: x.strftime("%m%d%H"))
  if capacity:
    stats_df = years_df[['hour_of_year', 'cap_fac']]\
                    .groupby(['hour_of_year']).agg(['mean','median','std',
                                                    'skew','min','max','count'])
  else:
    stats_df = years_df[['hour_of_year', 'electricity']]\
                    .groupby(['hour_of_year']).agg(['mean','median','std',
                                                    'skew','min','max','count'])
  return stats_df


def postprocess(
  today: str,
  preprocessed_df: pd.DataFrame,
  stats_df: pd.DataFrame,
  pred_df: pd.DataFrame,
) -> pd.DataFrame:
  """
  Create a df that contains all information necessary for the plot in streamlit.
  Accumulate all data for a specific time window defined by today.
  Input:
    - today: User input from streamlit; e.g. '2000-05-15'
    - preprocessed_df: df with all years that can be selected in streamlit
      (2000-2022)
    - stats_df: provided by get_stats_table()
    - pred_df: df from pred(), should contain two columns 'utc_time' and 'pred'
  Output:
    - plot_df with columns: utc_time, local_time, electricity, hour_of_year,
      mean, median, std, skew, min, max, count, pred
    - plot_df contains NaN-values! You have to replace them for api
  """
  # define time period (3 days) for plotting
  today_timestamp = pd.Timestamp(today, tz='UTC')
  window_df= pd.date_range(
            start=today_timestamp - pd.Timedelta(days=1),
            end=  today_timestamp + pd.Timedelta(days=2) - pd.Timedelta(hours=1),
            freq=pd.Timedelta(hours=1)).to_frame(index=False, name='utc_time')

  # create df with the preprocessed data in the time window
  plot_df = pd.merge(window_df, preprocessed_df, on='utc_time', how='inner')

  # add statistics in the time window
  plot_df['hour_of_year'] = plot_df.utc_time.\
                           apply(lambda x: x.strftime("%m%d%H"))
  stats_df.columns = stats_df.columns.droplevel(level=0)
  plot_df = pd.merge(plot_df, stats_df, on='hour_of_year', how='inner')

  # add prediction for day-ahead in time window
  plot_df = pd.merge(plot_df, pred_df, on='utc_time', how='left')

  return plot_df
