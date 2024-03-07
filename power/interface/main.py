
from power.ml_ops.data import get_pv_data, clean_pv_data, select_years
from power.ml_ops.model import model_yesterday
from power.ml_ops.cross_val import train_test_split, get_X_y


###

def preprocess_pv() -> None:

  train_test_ratio = 0.8
  input_length =
  number_of_sequences =
  output_length =

  df = get_pv_data()
  df = clean_pv_data(df)

  test_df   = select_years(df, start=2020, end=2022)
  train_val_df  = select_years(df, start=2080, end=2019)



  train_df, val_df = train_test_split(train_val_df,
                                            train_test_ratio, input_length)

  X_train, y_train = get_X_y(train_df,
                             number_of_sequences, input_length, output_length):


  # load_data to google (Jerome)
  # X_train, y_train
  # test_df
  # load_data_to_bq(
  #       data_processed_with_timestamp,
  #       gcp_project=GCP_PROJECT,
  #       bq_dataset=BQ_DATASET,
  #       table=f'processed_{DATA_SIZE}',
  #       truncate=True
  #   )


def training():

  return val_mae



if __name__ == '__main__':



    #create_yesterday_baseline(date='1982-07-06 05:00:00', power_source='PV')
    #create_mean_baseline(date='1982-07-06 05:00:00', power_source='PV')






# def create_yesterday_baseline(date, power_source='PV'):
#     """
#     create a simple baseline model based on the data form d-1.
#     """

#     pv_data = get_pv_data()
#     pv_data_clean = clean_pv_data(pv_data)

#     yesterday_baseline = model_yesterday(pv_data_clean, date)
#     return yesterday_baseline

# def create_mean_baseline(date, power_source='PV'):
#     """
#     create a simple baseline model based on the data form d-1.
#     """

#     pv_data = get_pv_data()
#     pv_data_clean = clean_pv_data(pv_data)

#     # <YOUR CODE>

#     pass
