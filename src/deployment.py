import asyncio
import sys
from typing import List

from prefect.deployments import Deployment
from prefect.filesystems import LocalFileSystem
from prefect.infrastructure import KubernetesJob
from pydantic import BaseModel

from src.external.grpc.prefect_routing_pb2 import GroupByResult

from .flow import predict, train


class Spec(BaseModel):
    """
    Contains information such as whether the algorithm supports additional
    aggregated columns, max. number of time group values, and what model types
    are supported.
    """

    supported_group_by_methods: List[str] = [
        GroupByResult.GroupByMethod.Name(GroupByResult.STANDARD)
    ]
    maximum_time_group_value_count: int = 10_000
    supported_model_types: List[str] = ["regression"]
    supported_modes: List[str] = ["accuracy", "balance", "speed"]


async def build_deployments(
    storage: LocalFileSystem,
    infrastructure: KubernetesJob,
) -> None:
    await Deployment.build_from_flow(
        flow=train,
        name="autotsf-lstm",
        tags=["exodus", "train", "autotsf", "ts"],
        work_queue_name="default",  # TODO change to something like `setup`
        skip_upload=True,  # ignore upload to container folder
        parameters={"spec": Spec().dict()},
        storage=storage,
        infrastructure=infrastructure,
        apply=True,
    )
    # Predict deployment
    await Deployment.build_from_flow(
        flow=predict,
        name="autotsf-lstm",
        tags=["exodus", "predict", "ts"],
        work_queue_name="default",  # TODO change to something like `setup`
        skip_upload=True,  # ignore upload to container folder
        parameters={
            "spec": {}
        },  # Spec is like max_time_group which corex server wanna know
        storage=storage,
        infrastructure=infrastructure,
        apply=True,
    )


async def build(job_name: str) -> None:
    storage = await LocalFileSystem.load("local")
    infrastructure = await KubernetesJob.load(job_name)
    await build_deployments(storage, infrastructure)


if __name__ == "__main__":
    job_name = sys.argv[1]
    asyncio.run(build(job_name))
