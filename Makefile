
run_main:
	python -m power.interface.main

streamlit:
	@streamlit run ui/app.py

reinstall_package:
	@pip uninstall -y power || :
	@pip install -e .

load_raw_pv:
	-bq rm --project_id ${GCP_PROJECT} ${BQ_DATASET}.raw_pv
	-bq mk --sync --project_id ${GCP_PROJECT} --location=${BQ_REGION} ${BQ_DATASET}.raw_pv
	python -c 'from power.ml_ops.data import load_raw_pv; load_raw_pv()'

load_raw_forecast:
	-bq rm --project_id ${GCP_PROJECT} ${BQ_DATASET}.raw_weather_forecast
	-bq mk --sync --project_id ${GCP_PROJECT} --location=${BQ_REGION} ${BQ_DATASET}.raw_weather_forecast
	python -c 'from power.ml_ops.data import load_raw_forecast; load_raw_forecast()'

load_raw_all: load_raw_pv, load_raw_forecast

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
	-bq rm --project_id ${GCP_PROJECT} ${BQ_DATASET}.processed_weather_forecast

	-bq mk --sync --project_id ${GCP_PROJECT} --location=${BQ_REGION} ${BQ_DATASET}.processed_pv
	-bq mk --sync --project_id ${GCP_PROJECT} --location=${BQ_REGION} ${BQ_DATASET}.processed_wind
	-bq mk --sync --project_id ${GCP_PROJECT} --location=${BQ_REGION} ${BQ_DATASET}.processed_weather_forecast


reset_gcs_files:
	-gsutil rm -r gs://${BUCKET_NAME}
	-gsutil mb -p ${GCP_PROJECT} -l ${GCP_REGION} gs://${BUCKET_NAME}

reset_all_files: reset_local_files reset_bq_files reset_gcs_files



################### DOCKER COMMANDS ##########################

# Edit tag (prod, 0.1, light, dev, ...)
TAG=1.2

PROJECT=${GCP_PROJECT}
IMAGE=${GAR_IMAGE}
REGION=${GCP_REGION}
DOCKER_REPO_NAME=${GAR_IMAGE}
IMAGE_URI=${REGION}-docker.pkg.dev/${PROJECT}/${DOCKER_REPO_NAME}/${IMAGE}:${TAG}

# authorize docker to work with GCP
authorize:
	gcloud auth configure-docker ${REGION}-docker.pkg.dev

# create GAR repo
create_repo:
	gcloud artifacts repositories create ${DOCKER_REPO_NAME} --repository-format=docker \
--location=${REGION} --description="Repository for storing advanced power forecast images in GAR"

# build and run locally - for apple silicon
build_silicon_local:
	docker build -t ${IMAGE} . -f Dockerfile_m1

run_silicon_local:
	docker run -e PORT=8000 -p 8080:8000 ${IMAGE}

build_silicon_prod:
	docker build --platform linux/amd64 -t ${IMAGE_URI} .

# build and run locally - for other machines
build_image:
	docker build -t ${IMAGE_URI} .

run_image:
	docker run -e PORT=8000 -p 8080:8000 --env-file .env ${IMAGE_URI}

# put in production - for all machines
push_image:
	docker push ${IMAGE_URI}

deploy_image:
	gcloud run deploy --image ${IMAGE_URI} --memory ${GAR_MEMORY} \
	--region ${REGION} --env-vars-file .env.yaml
