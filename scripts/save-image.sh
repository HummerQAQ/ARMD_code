#!/usr/bin/env bash

set -e

version="1.17.0"
build_type=$1
harbor=$2
name="lstm"

save () {
  version=$1
  dockerfile=$2
  tag=$harbor/$name:$version
  echo; echo ---------; echo Building $name version $version; echo ---------;
  just get-protobuf
  docker build -t $tag -f $dockerfile .
  docker push $tag
}

case $build_type in
  debug)
    version=$version"-debug"
    save $version ./scripts/debug/Dockerfile
    ;;

  nightly)
    # Generates a file called `cythonize-build-pyproject.toml`
    poetry run python ./scripts/gen_cython_pyproject.py
    version=$version"-nightly"
    save $version ./Dockerfile
    rm -f ./cythonize-build-pyproject.toml
    ;;

  release)
    # Generates a file called `cythonize-build-pyproject.toml`
    poetry run python ./scripts/gen_cython_pyproject.py
    save $version ./Dockerfile
    rm -f ./cythonize-build-pyproject.toml
    ;;

  *)
    echo "Unknown build type: "$build_type
    exit 1
    ;;
esac


