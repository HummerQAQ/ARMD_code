FROM harbor.mobagel.com/exodus-v2/builder AS builder
# Please append your name and update version code if you change this file.
LABEL maintainer="Pichu Chen <pichu@mobagel.com>"
LABEL version="1.0.1"
LABEL description="Exodus Dockerfile"

# Pichu: ENV will be set in exodus-builder, you can override it here
#        Please left comment to explain why you need to override it.
#        Comment is default value on exodus-builder 1.0.0 please check 
#        https://source.mobagel.com/exodus/builder/-/blob/main/Dockerfile
#        for latest version
# ENV PYTHONFAULTHANDLER=1 \
#   PYTHONUNBUFFERED=1 \
#   PYTHONHASHSEED=random \
#   PIP_NO_CACHE_DIR=off \
#   PIP_DISABLE_PIP_VERSION_CHECK=on \
#   PIP_DEFAULT_TIMEOUT=100 \
#   POETRY_VERSION=1.4.0 \
#   PYTHONDONTWRITEBYTECODE=duh \
#   CONTAINERIZED=1 

# System deps: (Will be done in exodus-builder and python image)
# RUN pip install "poetry==$POETRY_VERSION"
# RUN apt update; apt install build-essential -y

WORKDIR /code

# We will generate poetry.lock later
COPY pyproject.toml /code/

# Project initialization, installation and remove cache
RUN poetry config virtualenvs.create false \
  && poetry lock --no-update \
  && poetry install --no-dev --no-interaction --no-ansi \ 
  && rm -rf /root/.cache/pypoetry

# Pichu: Please check .dockerignore to make sure what will and will not be copied
COPY . /code

