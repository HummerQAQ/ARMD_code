import json
import pathlib
from io import BytesIO

import pytest
from bson import ObjectId
from testcontainers.minio import MinioContainer
from testcontainers.mongodb import MongoDbContainer

from exodus_common.exodus_common import Column, DataType, PredictRespBody, TimeUnit
from exodus_common.exodus_common.schemas import TrainTSReqBody
from src.model_algorithm import ModelAlgorithm

minio_container = MinioContainer().start()
mongodb_container = MongoDbContainer().start()
model_algorithm = ModelAlgorithm(test_containers=(minio_container, mongodb_container))

for bucket in ["test", "models", "prediction"]:
    if not model_algorithm.minio_client.bucket_exists(bucket):
        model_algorithm.minio_client.make_bucket(bucket)


@pytest.fixture(scope="session", autouse=True)
def teardown() -> None:
    # Just yield!
    yield
    print("Teardown: destroying db containers")
    minio_container.stop()
    mongodb_container.stop()


async def train(name: str) -> str:
    """
    Trains a dataset and returns the response.

    The train request is generated from "`name`/train.csv" and "`name`/meta.json".

    Make sure your `meta.json` contains the following fields:
    - `target_column_name`
    - `features`: a dict from the feature names to the feature types.

    Parameters
    ----------
    name : str
        The directory containing the dataset and configuration.

    Returns
    -------
    str
        The model ID returned after the training process.
    """
    path = pathlib.Path("tests/datasets")
    with open(path / name / "train.csv", "rb") as f:
        buf = f.read()
        model_algorithm.minio_client.put_object(
            "test", f"{name}_train.csv", BytesIO(buf), len(buf)
        )

    with open(path / name / "meta.json") as f:
        meta = json.loads(f.read())
        columns = [
            Column(name=name, data_type=DataType(data_type))
            for name, data_type in meta["features"].items()
        ]
        request = TrainTSReqBody(
            experiment_id=str(ObjectId()),  # This is bogus
            training_data=f"s3a://test/{name}_train.csv",
            feature_types=columns,
            target=next(x for x in columns if x.name == meta["target_column_name"]),
            folds=meta["folds"],
            date_column_name=meta["date_column_name"],
            derivation_window=meta["derivation_window"],
            forecast_horizon=meta["forecast_horizon"],
            gap=meta["gap"],
            time_unit=TimeUnit(meta["time_unit"]),
            endo_features=meta.get("endo_features", []),
            exo_features=meta.get("exo_features", []),
            time_groups=meta.get("time_groups", []),
            group_by_method=meta.get("groupby", "standard"),
        )
        resp = await model_algorithm.train_ts(request)
        return resp.model_id


async def predict(name: str, model_id: str) -> str:
    """
    Predicts over a dataset with a specified model, and returns the response.

    Parameters
    ----------
    name : str
        The directory containing the dataset.

    Returns
    -------
    str
        The prediction file uri.

    Raises
    ------
    AssertionError
        If the response is not an instance of `PredictRespBody`.
    """
    path = pathlib.Path("tests/datasets")
    with open(path / name / "prediction.csv", "rb") as f:
        buf = f.read()
        model_algorithm.minio_client.put_object(
            "test", f"{name}_prediction.csv", BytesIO(buf), len(buf)
        )
        f.seek(0)
        header = f.readline().decode().split(",")

    with open(path / name / "meta.json") as f:
        meta = json.loads(f.read())
        columns = [
            Column(name=name, data_type=DataType(data_type))
            for name, data_type in meta["features"].items()
        ]
        resp = await model_algorithm.predict(
            model_id=model_id,
            target=next(x for x in columns if x.name == meta["target_column_name"]),
            feature_types=[x for x in columns if x.name in header],
            training_feature_types=columns,
            prediction_input=f"s3a://test/{name}_prediction.csv",
            threshold=None,
            keep_columns=[],
            time_groups=meta.get("time_groups", []),
            time_unit=TimeUnit(meta["time_unit"]),
            derivation_window=meta["derivation_window"],
            forecast_horizon=meta["forecast_horizon"],
            gap=meta["gap"],
            datetime_column=meta["date_column_name"],
            endogenous_features=meta.get("endo_features", []),
            exogenous_features=meta.get("exo_features", []),
        )
        assert isinstance(resp, PredictRespBody)
        return resp.prediction


async def train_predict_delete(name: str) -> None:
    """
    Trains a model, checks it's been trained successfully, predicts with the models, checks if it succeeded, deletes the model, and finally checks if it is deleted.
    """
    model_id = await train(name)
    assert model_id is not None
    prediction_uri = await predict(name, model_id)
    assert prediction_uri is not None
