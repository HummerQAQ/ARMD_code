set shell := ["bash", "-c"] 

VERSION               := "0.1.0"
ML_JOB_NAME           := "exodus-lstm"

# Registry Image
IMAGE_NAME            := ML_JOB_NAME
IMAGE_TAG             := "mobagel.com:31320/" + IMAGE_NAME + ":latest"

HARBOR                := "harbor.mobagel.com"
CI_REGISTRY           := "core-tech-ci"
RELEASE_REGISTRY      := "core-tech-release"
REGISTRY              := if env_var_or_default("ACTION", "") == "release" { HARBOR / RELEASE_REGISTRY } else { HARBOR / CI_REGISTRY }
HARBOR_IMAGE_TAG      := REGISTRY / IMAGE_NAME + ":" + VERSION

# Prefect
HOST_IP               := `hostname -I | awk '{print $1}'`
PREFECT_API_URL       := "http://" + HOST_IP + ":4200/api"
ML_RUNNER_FOLDER      := "./prefect-infra"

# The protobuf definition
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION := 'python'  # tensorflow and grpc package conflict
FILES_HOST            := "files.mobagel.com"
ML_PROTOBUF_DIR       := "/ct_artifact/ml-protobuf"
ML_PROTOBUF_VERSION   := "0.3.14"
ML_PROTOBUF           := "https://" + FILES_HOST + ML_PROTOBUF_DIR / ML_PROTOBUF_VERSION + ".tar.gz"

default:
  just --list

# Update submodule and protobuf
pull: get-protobuf sync-submodule build-image

# Update submodule to latest version
sync-submodule:
  git submodule init
  git submodule update --remote --recursive

# Builds the base image for our machine learning runner
build:
  poetry export -o requirements.txt
  docker build -t {{IMAGE_TAG}} .
  rm -f requirements.txt

# Build and push to local Registry
build-image: build
  docker push {{IMAGE_TAG}}

# Builds the k8s job infra block
build-k8s-job: build-image
  docker run --rm --name {{ML_JOB_NAME}} \
              -e PREFECT_API_URL={{PREFECT_API_URL}} \
              -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
              -e PYTHONUNBUFFERED=1 \
              {{IMAGE_TAG}} \
              python {{ML_RUNNER_FOLDER}}/scripts/setup_kubernetes_job.py {{ML_JOB_NAME}} {{IMAGE_TAG}}


# Builds a specified deployment
build-deployment DEPLOYMENT: build-image
  poetry run python -m `echo {{DEPLOYMENT}} | cut -f 1 -d '.' | sed 's/\//\./g'` {{ML_JOB_NAME}}

# Push image to Harbor registry server
push: build
  docker tag {{IMAGE_TAG}} {{HARBOR_IMAGE_TAG}}
  docker push {{HARBOR_IMAGE_TAG}}

# Removes all completed jobs from the K8S cluster.
clean-completed-jobs:
  @for i in $(kubectl get pods | grep Completed | awk '{print $1;}'); do kubectl delete pod $i; done

# Build GRPC template files
get-protobuf:
  mkdir -p src/external/grpc
  curl -u $FILES_USER:$FILES_PASSWORD -s {{ML_PROTOBUF}} | tar xzf - --strip-components 1 -C src/external/grpc
  docker run --rm -v ${PWD}:/algo-builder \
                  -w /algo-builder harbor.mobagel.com/exodus-v2/builder \
                  poetry run python -m grpc_tools.protoc -I . \
                                                         --python_out=. \
                                                         --pyi_out=. \
                                                         --grpc_python_out=. \
                                                         src/external/grpc/*.proto

# Link data to exist deployment block, you can save file into '/data' path
link-pvc: build
  mkdir -p ./data
  docker run --rm --name {{ML_JOB_NAME}} \
              -e PREFECT_API_URL={{PREFECT_API_URL}} \
              -e PYTHONUNBUFFERED=1 \
              {{IMAGE_TAG}} \
              python {{ML_RUNNER_FOLDER}}/scripts/link_pvc.py {{ML_JOB_NAME}} {{IMAGE_TAG}} {{justfile_directory()}}/data

# Formats the entire repo with black and isort
format:
  poetry run black .
  poetry run isort .

# Installs dependencies
install-dependencies:
  poetry install --no-root