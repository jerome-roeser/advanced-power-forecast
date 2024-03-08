
from tensorflow.keras.callbacks import EarlyStopping

from power.ml_ops.data import get_pv_data, clean_pv_data, select_years
from power.ml_ops.model import model_yesterday, init_RNN, init_baseline_yesterday
from power.ml_ops.cross_val import get_X_y


###

def preprocess_pv() -> None:
  """
  Check availability of traing and test data at cache or big query. If data is
  not available from cache or big query, then preprocess raw data from local
  folder and upload the data to big query and save it in the cache.
  Input:
    - No function input
    - Try to load data from different sources in the following order:
      1. Load traing and test data from cache folder
      2. Load traing and test data from google big query
      3. Load raw data from local folder
    - Enviromental variables used:
      - N_SEQUENCES_TRAIN = 250     # number_of_sequences_train
      - OUTPUT_LENGTH = 24          # predict 24 hours
      - TRAIN_TEST_RATIO = 0.83     # 5 years train, 1 year test
      - INPUT_LENGTH = 24 * 2       # records every hour (24 hours) x 2 days = 48
  Output:
    - No function output
    - Upload data to google big query and save to cache
  """
  # ----------------------------------------------------------------------------
  ### Check availability of traing and test data from cache folder

  # TODO: check for test_df, train_val_df, X_train, y_train, X_test, y_test

  # ----------------------------------------------------------------------------
  ### Check availability of traing and test data from big query

  # TODO: check for test_df, train_df, X_train, y_train, X_test, y_test;
  #       save test_df, train_df, X_train, y_train, X_test, y_test to cache

  # ----------------------------------------------------------------------------
  ### Load raw data and preprocess it
  df = get_pv_data()
  df = clean_pv_data(df)

  # Test train split the clean data
  test_df       = select_years(df, start=2020, end=2022)
  train_df  = select_years(df, start=2080, end=2019)

  # Preprocess
  # to train the model
  X_train, y_train = get_X_y(train_df,
                            N_SEQUENCES_TRAIN, INPUT_LENGTH, OUTPUT_LENGTH)

  # to test the model
  X_test, y_test = get_X_y(test_df,
                            N_SEQUENCES_TRAIN, INPUT_LENGTH, OUTPUT_LENGTH)

  ## Save preprocessed data to cache

  # TODO: save test_df, train_df, X_train, y_train, X_test, y_test to cache

  ## Upload preprocessed data to google query

  # TODO: upload test_df, train_df, X_train, y_train, X_test, y_test to cache

  # load_data_to_bq(
  #       data_processed_with_timestamp,
  #       gcp_project=GCP_PROJECT,
  #       bq_dataset=BQ_DATASET,
  #       table=f'processed_{DATA_SIZE}',
  #       truncate=True
  #   )


def training(
  validation_split: float = 0.3,
  epochs=50,
  batch_size=32,
  patience=5
  ) -> float:
  """
  First get training data from google big query or cache. Then train the
  RNN model and store training results and the model.
  Input:
    - No mandaory function input; optional parameters to adjust the training
  Output:
    -
  """
  # ----------------------------------------------------------------------------
  ### Load training data

  # TODO: Get X_train, y_train from the cache

  # ----------------------------------------------------------------------------
  ### Initialize model
  model_RNN = init_RNN(X_train, y_train) # from module model inpackage ml_ops

  # ----------------------------------------------------------------------------
  ### train model

  es = EarlyStopping(monitor = "val_mae",
                     patience = patience,
                     restore_best_weights = True)

  history = model_RNN.fit(X_train, y_train,
                      validation_split = validation_split,
                      shuffle = False,
                      batch_size = batch_size,
                      epochs = epochs,
                      callbacks = [es],
                      verbose = 0)

  # ----------------------------------------------------------------------------
  ### Save result and model


  print('# model training done')

  return val_mae



if __name__ == '__main__':
  # execute function in the right order






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
