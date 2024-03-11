# GCP Setup

💻 Create a bucket in your GCP account

Make sure to create the bucket where you are located yourself (use `GCP_REGION` in the `.env`)
Fill also the `BUCKET_NAME` variable with the name of your choice (must be globally unique and lower case! If you have an uppercase letter in your GitHub username, you’ll need to make it lower case!)
e.g.

``` python
gsutil mb \
    -l $GCP_REGION \
    -p $GCP_PROJECT \
    gs://$BUCKET_NAME
```
``` python
direnv reload .
```

💻 Create an dataset where preprocessed data will be stored & queried

Using `bq` and the following env variables, create a new dataset called power on your own `GCP_PROJECT`

``` python
bq mk \
    --project_id $GCP_PROJECT \
    --data_location $BQ_REGION \
    $BQ_DATASET
```
``` python
bq mk --location=$GCP_REGION $BQ_DATASET.raw_pv
bq mk --location=$GCP_REGION $BQ_DATASET.raw_wind
bq mk --location=$GCP_REGION $BQ_DATASET.processed_pv
bq mk --location=$GCP_REGION $BQ_DATASET.processed_wind
```
```  python
bq show
bq show $BQ_DATASET
bq show $BQ_DATASET.raw_pv
bq show $BQ_DATASET.processed_pv
```
``` python
direnv reload .
```

👉 Run make `reset_all_files` directive –> It resets all local files (csvs, models, …) and data from bq tables and buckets, but preserve local folder structure, bq tables schema, and gsutil buckets.

👉 Run separately `reset_local_files`, `reset_bq_files` or `reset_gcs_files` to remove local, BigQuery or Cloud Storage independantly.

👉 Run make `show_sources_all` to see that you’re back from a blank state!

