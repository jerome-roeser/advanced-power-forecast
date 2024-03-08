# GCP Setup

💻 Create a bucket in your GCP account using gsutil

Make sure to create the bucket where you are located yourself (use GCP_REGION in the .env)
Fill also the BUCKET_NAME variable with the name of your choice (must be globally unique and lower case! If you have an uppercase letter in your GitHub username, you’ll need to make it lower case!)
e.g.
""" python
BUCKET_NAME = power_<github-username>
"""
""" python
direnv reload .
"""

💻 Create our own dataset where we’ll store & query preprocessed data !

Using bq and the following env variables, create a new dataset called power on your own GCP_PROJECT

""" python
bq mk \
    --project_id $GCP_PROJECT \
    --data_location $BQ_REGION \
    $BQ_DATASET
"""
""" pyhton
bq mk --location=$GCP_REGION $BQ_DATASET.processed_pv
bq mk --location=$GCP_REGION $BQ_DATASET.processed_wind
"""
"""  python
bq show
bq show $BQ_DATASET
bq show $BQ_DATASET.processed_pv
"""

🎁 Look at make reset_all_files directive –> It resets all local files (csvs, models, …) and data from bq tables and buckets, but preserve local folder structure, bq tables schema, and gsutil buckets.

Very useful to reset state of your challenge if you are uncertain and you want to debug yourself!

👉 Run make reset_all_files safely now, it will remove files from unit 01 and make it clearer

👉 Run make show_sources_all to see that you’re back from a blank state!

✅ When you are all set, track your results on Kitt with make test_kitt (don’t wait, this takes > 1min)
