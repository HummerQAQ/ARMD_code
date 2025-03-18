FROM python:3.8-slim

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.1.12 \
  PYTHONDONTWRITEBYTECODE=duh \
  CONTAINERIZED=1

# System deps:
RUN pip install "poetry==$POETRY_VERSION"
RUN apt update; apt install gcc -y

# Copy only requirements to cache them in docker layer
WORKDIR /code
COPY pyproject.toml build.py /code/

# This speeds up build time, but messes up poetry such that `poetry build` is not usable
# This is fine in debug build!
COPY poetry.lock /code/

# Copy the files we need when the image is exported
COPY lstm /code/lstm
COPY exodus_common /code/exodus_common
COPY README.md LICENSE /code/

# Project initialzation and installation
RUN poetry config virtualenvs.create false \
  && poetry install --no-dev --no-interaction --no-ansi
