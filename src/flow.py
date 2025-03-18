import os
from typing import Any, Dict, List, Optional, Union

import prefect.context
from prefect.context import FlowRunContext, TaskRunContext
from prefect.flows import flow

from exodus_common.exodus_common import (
    AlgoType,
    Column,
    ExodusForbidden,
    PredictRespBody,
    TimeUnit,
    TrainRespBody,
    TrainTSReqBody,
)
from exodus_common.exodus_common.infra.grpc_client import grpc_client
from exodus_common.exodus_common.infra.prefect import error_handler, get_flow_info
from src.external.grpc.prefect_routing_pb2 import (
    ExodusPredictResult,
    TrainAutoTSFResult,
)
from src.model_algorithm import ModelAlgorithm


@flow(name="train", log_prints=True)
@error_handler
async def train(req: TrainTSReqBody, spec: Dict[str, Any] = {}) -> None:
    """
    Trains a IID model based on the given configurations.
    - **spec**: Algorithm spec. like how many time_group could be used in this algorithm:
    - **req**: The train request. Several things to note:
      - Make sure your **target** field is included in the **feature_types**.
      - The **url** field represents the server that is expecting a callback message from this API. If this is specified, make sure there is really a server listening on that url (i.e. by running `poe start-simple-server` in another terminal).
      - If there's no **simple_request** given, this API will respond immediately with an OK message. The actual training will be done in the event loop, and a response will print to the logs once training is complete.

    """
    # Initital database connection
    model_algorithm = ModelAlgorithm()
    if model_algorithm.grpc_channel is not None:
        grpc_client.init_app(model_algorithm.grpc_channel)
    if model_algorithm.algo_type != AlgoType.TS:
        raise ExodusForbidden(
            f"`train_ts` is invalid for model with `algo_type` = {model_algorithm.algo_type.value}",
        )

    # Get the current number of runs have been executed
    run_context: Union[
        FlowRunContext, TaskRunContext
    ] = prefect.context.get_run_context()
    if isinstance(run_context, FlowRunContext):
        run_count: int = run_context.flow_run.run_count

        if run_count == 2:  # The flow run retries once
            raise MemoryError(
                "The algorithm failed to train. Currently, this happened due to memory issues."
            )

    result: TrainRespBody = await model_algorithm.train_ts(req)

    grpc_result = TrainAutoTSFResult(
        flow_info=get_flow_info(),
        model_id=result.model_id,
    )

    await grpc_client.send_train_tsf_result(grpc_result)


@flow(name="predict", log_prints=True)
@error_handler
async def predict(
    model_id: str,
    target: Column,
    feature_types: List[Column],
    training_feature_types: List[Column],
    prediction_input: str,
    time_groups: List[str],
    time_unit: TimeUnit,
    derivation_window: int,
    forecast_horizon: int,
    gap: int,
    datetime_column: str,
    endogenous_features: List[str],
    exogenous_features: List[str],
    threshold: Optional[float] = None,
    keep_columns: List[str] = [],
    non_negative: bool = False,
    prediction_output_uri: Optional[str] = None,
    override_grpc_host: Optional[str] = None,
    spec: Dict[str, Any] = {},  # TODO fix this
) -> None:
    """
    Predict a IID model based on the given configurations.
    - **spec**: Algorithm spec. like how many time_group could be used in this algorithm:
    - **req**: The train request. Several things to note:
      - Make sure your **target** field is included in the **feature_types**.
      - The **url** field represents the server that is expecting a callback message from this API. If this is specified, make sure there is really a server listening on that url (i.e. by running `poe start-simple-server` in another terminal).
      - If there's no **simple_request** given, this API will respond immediately with an OK message. The actual training will be done in the event loop, and a response will print to the logs once training is complete.

    """
    # Initital database connection
    if override_grpc_host:
        os.environ["EXODUS_GRPC_HOST"] = override_grpc_host
    model_algorithm = ModelAlgorithm()
    if model_algorithm.grpc_channel is not None:
        grpc_client.init_app(model_algorithm.grpc_channel)

    # Get the current number of runs have been executed
    run_context: Union[
        FlowRunContext, TaskRunContext
    ] = prefect.context.get_run_context()
    if isinstance(run_context, FlowRunContext):
        run_count: int = run_context.flow_run.run_count

        if run_count == 2:  # The flow run retries once
            raise MemoryError(
                "The algorithm failed to predict. Currently, this happened due to memory issues."
            )

    predict_result = await model_algorithm.predict(
        model_id=model_id,
        target=target,
        feature_types=feature_types,
        training_feature_types=training_feature_types,
        prediction_input=prediction_input,
        time_groups=time_groups,
        time_unit=time_unit,
        derivation_window=derivation_window,
        forecast_horizon=forecast_horizon,
        gap=gap,
        datetime_column=datetime_column,
        endogenous_features=endogenous_features,
        exogenous_features=exogenous_features,
        threshold=threshold,
        keep_columns=keep_columns,
        non_negative=non_negative,
        prediction_output_uri=prediction_output_uri,
    )
    assert isinstance(predict_result, PredictRespBody)

    grpc_result = ExodusPredictResult(
        flow_info=get_flow_info(),
        uri=predict_result.prediction,
    )
    print(f"Sending ExodusPredictResult via gRPC = {grpc_result}")
    await grpc_client.send_predict_result(grpc_result)
