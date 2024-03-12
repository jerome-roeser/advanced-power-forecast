
run_main:
	python -m power.interface.main

streamlit:
	@streamlit run ui/app.py

reinstall_package:
	@pip uninstall -y power || :
	@pip install -e .

run_preprocess:
	python -c 'from power.interface.main import preprocess; preprocess()'

run_train:
	python -c 'from power.interface.main import train; train()'

run_pred:
	python -c 'from power.interface.main import pred; pred()'

run_evaluate:
	python -c 'from power.interface.main import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate

run_api:
	uvicorn power.api.fast:app --reload


################### DATA SOURCES ACTIONS ################

# Data sources: targets for monthly data imports
ML_DIR= power/.lewagon/mlops

show_sources_all:
	-ls -laR power/.lewagon/mlops/data
	-bq ls ${BQ_DATASET}
	-bq show ${BQ_DATASET}.processed_pv
	-bq show ${BQ_DATASET}.processed_wind
	-gsutil ls gs://${BUCKET_NAME}

reset_local_files:
	rm -rf ${ML_DIR}
	mkdir -p power/.lewagon/mlops/data/
	mkdir power/.lewagon/mlops/data/raw
	mkdir power/.lewagon/mlops/data/processed
	mkdir power/.lewagon/mlops/training_outputs
	mkdir power/.lewagon/mlops/training_outputs/metrics
	mkdir power/.lewagon/mlops/training_outputs/models
	mkdir power/.lewagon/mlops/training_outputs/params

reset_bq_files:
	-bq rm --project_id ${GCP_PROJECT} ${BQ_DATASET}.processed_pv
	-bq rm --project_id ${GCP_PROJECT} ${BQ_DATASET}.processed_wind

	-bq mk --sync --project_id ${GCP_PROJECT} --location=${BQ_REGION} ${BQ_DATASET}.processed_pv
	-bq mk --sync --project_id ${GCP_PROJECT} --location=${BQ_REGION} ${BQ_DATASET}.processed_wind


reset_gcs_files:
	-gsutil rm -r gs://${BUCKET_NAME}
	-gsutil mb -p ${GCP_PROJECT} -l ${GCP_REGION} gs://${BUCKET_NAME}

reset_all_files: reset_local_files reset_bq_files reset_gcs_files
