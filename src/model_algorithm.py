import pickle
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from sklearn.preprocessing import StandardScaler
from testcontainers.minio import MinioContainer
from testcontainers.mongodb import MongoDbContainer

from exodus_common.exodus_common import (
    PREDICTION_COLNAME,
    TGVAL_NO_TGVAL_KEY,
    AlgoType,
    Column,
    Configs,
    CVScores,
    DataType,
    ExodusForbidden,
    MinioURI,
    ModelAttributeSummary,
    PredictRespBody,
    RegressionScores,
    Scores,
    SingleModel,
    SplitFrames,
    TimeUnit,
    TrainRespBody,
    TrainTSReqBody,
)
from exodus_common.exodus_common.infra import ModelStore, get_prediction_header
from exodus_common.exodus_common.internal import cast_df_types, get_df
from exodus_common.exodus_common.model_algorithm.models.ts import (
    TimeGroupValue,
    TSCVFrames,
    get_valid_cv_time_series_splits,
    is_valid_fold,
    non_negative_prediction_handler,
)
from exodus_common.exodus_common.model_algorithm.predict.ts import (
    calculate_all_time_groups_prediction_normalization_statistics,
    filter_score_data,
    normalize_prediction_result,
)
from src.extra_attributes import ExtraAttributes
from src.training import train_lstm_model
from src.training.helpers import early_stop_conditions
from src.training.utils import (
    from_y_shape,
    get_last_period_data,
    get_params,
    get_prediction_result,
)

MIN_DERIVATION_WINDOW = 2
ALGORITHM_NAME = "LSTM"


def run_cv(
    request: TrainTSReqBody,
    splits: List[SplitFrames],
    all_time_group_values: List[str],
) -> Tuple[CVScores, Dict[str, List[List[float]]]]:
    split_scores: List[Scores] = []  # List to store regression scores for each fold
    fold_prediction: pd.DataFrame

    # Get CV scores for each fold
    print(f"There are {len(splits)} CV Fold(s) to run")
    fold_predictions: List[pd.DataFrame] = []
    for fold, split_frames in enumerate(splits):
        print(f"CV Fold: {fold + 1} of {len(splits)}")
        actual = (
            split_frames.test[request.target.name].tolist()
            if split_frames.test is not None
            else []
        )
        # The CV fold is valid when any of the actual target value is not NA
        if is_valid_fold(actual):
            # initialize dataframe
            if not request.time_groups:
                training_df_pivot = split_frames.train
                target_pivot = [request.target.name]
            else:
                # create a map from each time group to its possible values
                training_df_pivot = split_frames.train.pivot_table(
                    index=request.date_column_name,
                    columns=request.time_groups,
                    values=request.target.name,
                )
                # format: [('CA_1', 'HOBBIES_1_001'), ('CA_1', 'HOBBIES_1_002'), ... ('CA_1', 'HOBBIES_1_003')]
                target_pivot = training_df_pivot.columns.tolist()

            early_stop = early_stop_conditions(training_df_pivot, request.time_groups)

            train = training_df_pivot.loc[:, target_pivot].values
            # To avoid erros when modeling on small amount of data,
            # so try use the lowest limit of the derivation window.
            # It depends on the length of training data in each cv folds.
            if len(train) - request.folds * request.forecast_horizon < (
                request.forecast_horizon + request.derivation_window + request.gap
            ):
                derivation_window = MIN_DERIVATION_WINDOW
            else:
                derivation_window = request.derivation_window

            # not enough data to train LSTM model, so the metric of the fold is NaN.
            if (
                len(train)
                - (request.forecast_horizon + request.gap + derivation_window)
                < 0
            ):
                split_scores.append(
                    RegressionScores(
                        mse=np.nan,
                        rmse=np.nan,
                        rmsle=np.nan,
                        mae=np.nan,
                        mape=np.nan,
                        wmape=np.nan,
                        r2=np.nan,
                        deviance=np.nan,
                    )
                )
                print(
                    f"The length of training data is {len(train)}, which is not enough to train."
                )
                continue

            cv_model, last_period_train_data, scaler = train_last_period_data(
                train,
                early_stop,
                derivation_window,
                request.forecast_horizon,
                target_pivot,
                request.gap,
                request.time_groups,
            )
            predict_result = cv_model.predict(
                np.nan_to_num(last_period_train_data, nan=0), verbose=0
            )
            # take notes:
            # - scikit-learn==1.0.2(now version) is different data manipulate on 0.24.1
            if request.time_groups:
                predict_result = from_y_shape(
                    predict_result, request.forecast_horizon, target_pivot
                )
            if not request.time_groups:
                time_group_values = TGVAL_NO_TGVAL_KEY
                fold_prediction = pd.DataFrame(
                    {
                        "predictions": scaler.inverse_transform(predict_result)[0],
                        "time_group_value": time_group_values,
                        request.target.name: split_frames.test[request.target.name]
                        if split_frames.test is not None
                        else None,
                        "fold": fold,
                    }
                ).dropna()
            else:
                if split_frames.test is not None:
                    actual_df = split_frames.test[
                        [request.date_column_name, request.target.name]
                        + request.time_groups
                    ]
                else:
                    actual_df = pd.DataFrame()
                # Due to the impact of valid and invalid cv on the output of predictions,
                # it's necessary to list the time groups with predictions and forecasting date.
                predict_vals = pd.DataFrame(
                    scaler.inverse_transform(predict_result[0]),
                    columns=pd.MultiIndex.from_tuples(
                        target_pivot, names=request.time_groups
                    )
                    if len(request.time_groups) > 1
                    else target_pivot,
                )
                predict_vals.index = (
                    split_frames.test[request.date_column_name].unique().tolist()
                )
                predict_df = predict_vals.unstack()
                predict_df.index.names = request.time_groups + [
                    request.date_column_name
                ]
                predict_df = predict_df.reset_index(name="predictions")
                predict_df[request.date_column_name] = pd.to_datetime(
                    predict_df[request.date_column_name]
                )

                # Do the intersection of actual and predict
                # based on time groups and date column
                fold_prediction = pd.merge(
                    actual_df,
                    predict_df,
                    on=[request.date_column_name] + request.time_groups,
                ).dropna()
                fold_prediction["fold"] = fold

                def decode_to_key(xs: pd.Series) -> str:
                    return TimeGroupValue(val=xs.values.tolist()).to_mongodb_key()

                fold_prediction["time_group_value"] = fold_prediction[
                    request.time_groups
                ].apply(
                    decode_to_key,
                    axis=1,
                )
            fold_predictions.append(
                fold_prediction.loc[:, ["predictions", "time_group_value", "fold"]]
            )
            invalid_time_group_values = [
                x
                for x in all_time_group_values
                if x not in fold_prediction["time_group_value"]
            ]
            invalid_time_group_value_predictions = pd.DataFrame(
                {
                    "predictions": [
                        np.nan for _ in range(len(invalid_time_group_values))
                    ],
                    "time_group_value": invalid_time_group_values,
                    "fold": fold,
                }
            )
            fold_predictions.append(invalid_time_group_value_predictions)

            pred, actual = filter_score_data(
                fold_prediction["predictions"].tolist(),
                fold_prediction[request.target.name].tolist(),
            )
            split_scores.append(
                RegressionScores.get_scores(np.array(pred), np.array(actual))
            )
        # For fold without any valid time series, make all the metrics NAs
        else:
            print(f"There is no valid time series in Fold {fold+1}.")
            fold_predictions.append(
                pd.DataFrame(
                    {
                        "predictions": [
                            np.nan for _ in range(len(all_time_group_values))
                        ],
                        "time_group_value": all_time_group_values,
                        "fold": fold,
                    }
                )
            )
            split_scores.append(
                RegressionScores(
                    mse=np.nan,
                    rmse=np.nan,
                    rmsle=np.nan,
                    mae=np.nan,
                    mape=np.nan,
                    wmape=np.nan,
                    r2=np.nan,
                    deviance=np.nan,
                )
            )
    fold_predictions_df = pd.concat(fold_predictions, axis="index")

    def extract_fold_predictions(df: pd.DataFrame) -> List[List[float]]:
        return (
            df.groupby("fold")
            .apply(lambda x: x["predictions"].dropna().values.tolist())
            .tolist()
        )

    print(fold_predictions_df)
    predictions: Dict[str, List[List[float]]] = (
        fold_predictions_df.groupby("time_group_value")
        .apply(extract_fold_predictions)
        .to_dict()
    )

    # Collect scores of each fold
    cv_scores = CVScores(split_scores=split_scores)

    return cv_scores, predictions


def train_last_period_data(
    train,
    early_stop,
    derivation_window,
    forecast_horizon,
    target_pivot,
    gap,
    time_groups,
):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    last_period_train_data = get_last_period_data(
        train_scaled, derivation_window, target_pivot
    )

    model = train_lstm_model(
        np.nan_to_num(train_scaled, nan=0),
        early_stop,
        derivation_window,
        forecast_horizon,
        gap,
        time_groups,
    )
    return model, last_period_train_data, scaler


class ModelAlgorithm:
    def __init__(
        self, test_containers: Optional[Tuple[MinioContainer, MongoDbContainer]] = None
    ) -> None:
        """
        Initializes the model algorithm instance. Will also instantiate the underlying `MongoInstance`.
        """
        self.name: str = "lstm"
        self.description: str = """
        it is implementing the Keras.Sequential algorithm.
        """

        if test_containers:
            minio_container, mongodb_container = test_containers

            self.mongo_client: MongoClient = mongodb_container.get_connection_client()
            self.collection: Collection = self.mongo_client.get_database(
                "exodus"
            ).get_collection("SingleModel")

            self.minio_client = minio_container.get_client()
            self.storage_options: Dict[str, Any] = {
                "key": "minioadmin",
                "secret": "minioadmin",
                "client_kwargs": {
                    "endpoint_url": f"http://{minio_container.get_container_host_ip()}:9000"
                },
            }
            self.n_jobs = 8

            self.grpc_channel = None  # Unused
        else:
            configs = Configs.get(self.name)

            self.mongo_client: MongoClient = configs.mongo_instance.client  # type: ignore[no-redef]
            self.collection: Collection = self.mongo_client.get_database(configs.mongodb_configs.database).get_collection("SingleModel")  # type: ignore[no-redef]

            self.minio_client = configs.minio_configs.get_client()
            self.storage_options: Dict[str, Any] = {  # type: ignore[no-redef]
                "key": configs.minio_configs.username,
                "secret": configs.minio_configs.password,
                "client_kwargs": {
                    "endpoint_url": f"http://{configs.minio_configs.host}:{configs.minio_configs.port}"
                },
            }
            self.n_jobs = configs.cores

            self.grpc_channel = configs.grpc_channel

        self.algo_type: AlgoType = AlgoType.TS

    async def train_ts(self, request: TrainTSReqBody) -> TrainRespBody:
        """
        Trains a TS model.

        Parameters
        ----------
        request : TrainTSReqBody
            The training request containing necessary information for this task.

        Returns
        -------
        TrainRespBody
            The results of this training task.

        """

        start_time = datetime.utcnow()

        # LSTM only supports regression problems, fail if we see something else
        if request.target.data_type != DataType.double:
            raise ExodusForbidden(
                f"{self.name} only supports regression problems",
            )

        print(f"Get training data.")
        time_group_dfs = request.get_time_grouped_df(
            self.minio_client, request.time_groups, request.target.name
        )  # The data types have been cast
        training_df: pd.DataFrame = pd.concat(
            time_group_dfs.values(), axis=0, ignore_index=True
        )

        last_date = training_df[request.date_column_name].astype(str).iloc[-1]

        request.derivation_window = min(
            request.forecast_horizon * 2, request.derivation_window
        )

        # Get Cross-Validation data
        print("Split the training data into Cross-Validation splits")
        splits = TSCVFrames.get_frames(
            training_df,
            request.date_column_name,
            request.folds,
            request.time_unit,
            request.forecast_horizon,
            request.gap,
            False,
        ).splits

        # Get valid time series for scoring in cross-validation stage
        for fold, split_frames in enumerate(splits):
            print(f"Fold: {fold + 1} of {len(splits)}")

            # Show original number of time series
            if request.time_groups:
                n_original_time_series = split_frames.test.groupby(
                    request.time_groups
                ).ngroups
            else:
                n_original_time_series = 1
            print(f"Original number of time series: {n_original_time_series}")

            # Process the split_frames to only keep the valid time series
            split_frames = get_valid_cv_time_series_splits(
                split_frames=split_frames,
                target_column_name=request.target.name,
                time_groups=request.time_groups,
            )

            # Show number of valid time series after being processed
            if split_frames.test.shape[0] > 0:
                if request.time_groups:
                    n_valid_time_series = (
                        split_frames.test.loc[
                            split_frames.test[request.target.name].notnull(), :
                        ]
                        .groupby(request.time_groups)
                        .ngroups
                    )
                else:
                    if split_frames.test[request.target.name].notnull().any():
                        n_valid_time_series = 1
                    else:
                        n_valid_time_series = 0
            else:
                n_valid_time_series = 0
            print(f"Number of valid time series: {n_valid_time_series}")

        if request.time_groups:
            time_group_values = (
                training_df[request.time_groups]
                .apply(
                    lambda xs: TimeGroupValue(val=xs.values.tolist()).to_mongodb_key(),
                    axis=1,
                )
                .unique()
                .tolist()
            )
        else:
            time_group_values = [TGVAL_NO_TGVAL_KEY]

        # Gather the folds and calculate the cross validation scores
        cv_scores, predictions = run_cv(request, splits, time_group_values)
        cv_scores_report = cv_scores.to_report()
        cv_averages = {
            metric: np.nanmean(cv_scores_report[metric]) for metric in cv_scores_report
        }
        cv_deviations = {
            metric: np.nanstd(cv_scores_report[metric]) for metric in cv_scores_report
        }

        # initialize dataframe
        if not request.time_groups:
            df_pivot = training_df
            target_pivot = [request.target.name]
        else:
            # create a map from each time group to its possible values
            df_pivot = training_df.pivot_table(
                index=request.date_column_name,
                columns=request.time_groups,
                values=request.target.name,
            )
            # format: [('CA_1', 'HOBBIES_1_001'), ('CA_1', 'HOBBIES_1_002'), ... ('CA_1', 'HOBBIES_1_003')]
            target_pivot = df_pivot.columns.tolist()

        early_stop = early_stop_conditions(df_pivot, request.time_groups)
        # Build final model
        final_model_train: np.ndarray = df_pivot.loc[:, target_pivot].values

        model, final_model_data, scaler = train_last_period_data(
            np.nan_to_num(final_model_train, nan=0),
            early_stop,
            request.derivation_window,
            request.forecast_horizon,
            target_pivot,
            request.gap,
            request.time_groups,
        )

        end_time = datetime.utcnow()
        training_time = (end_time - start_time).total_seconds()

        attributes_summary = ModelAttributeSummary(
            name=self.name,
            training_time=training_time,
            cv_scores=cv_scores_report,
            cv_deviations=cv_deviations,
            cv_averages=cv_averages,
            variable_importance=[],
            validation_scores={},  # TODO: DS recommand don't need to setting validation_scores
            holdout_scores=None,
        )
        fe_display_name = "LSTM"
        if request.n is not None:
            model_name = f"{fe_display_name} {request.n}"
        else:
            model_name = fe_display_name

        # Calculate statistics for normalizing the prediction results
        print("Calculate statistics for normalizing the prediction results.")
        prediction_normalization_statistics: Dict[
            str, Dict[str, float]
        ] = calculate_all_time_groups_prediction_normalization_statistics(
            time_group_dfs,
            request.target.name,
        )

        extra_attributes = ExtraAttributes(
            final_model_data=final_model_data,
            scaler=scaler,
            meta={"target_pivot": target_pivot, "last_date": last_date},
            predictions=predictions,
            prediction_normalization_statistics=prediction_normalization_statistics,
        )
        model_id = str(ObjectId())
        model_bytes = pickle.dumps(model)
        hyperparameters = get_params(early_stop)
        hyperparameters["group_by_method"] = request.group_by_method
        model_store = ModelStore(
            name=f"{self.name}_{model_id}",  # There can be more than 1 LGBM speed model, gotta make sure the model names are distinct.
            minio_client=self.minio_client,
            collection=self.collection,
        )
        model_store.save(
            model_id=model_id,
            single_model=SingleModel(
                algorithm_key=self.name,  # Algo. key
                key=model_store.name,  # Minio file location
                algorithm_name=fe_display_name,  # FE Display Algo. Name
                name=model_name,  # FE Display Model Name
                experiment=request.experiment_id,
                importances=[],
                attributes=attributes_summary,
                hyperparameters=hyperparameters,
                created_at=start_time,
                updated_at=end_time,
                completed_at=end_time,
                options=ExtraAttributes.keys(),
            ),
            model_bytes=model_bytes,
            extra_attributes=extra_attributes.to_dict(),
        )

        # Create the response, then return it
        return TrainRespBody(
            model_id=model_id,
        )

    async def predict(
        self,
        model_id: str,
        target: Column,
        feature_types: List[Column],
        training_feature_types: List[Column],
        prediction_input: str,  # This is an URI pointing to a file in Minio
        threshold: Optional[float],
        keep_columns: List[str],
        time_groups: List[str],
        time_unit: TimeUnit,
        derivation_window: int,
        forecast_horizon: int,
        gap: int,
        datetime_column: str,
        endogenous_features: List[str],
        exogenous_features: List[str],
        is_batch_predict: bool = False,
        non_negative: bool = False,
        prediction_output_uri: Optional[str] = None,
    ) -> Union[PredictRespBody, str]:
        """
        Predicts based on a request containing the model id and prediction dataframe.

        Parameters
        ----------
        request : PredictReqBody
            The prediction request.

        Returns
        -------
        PredictRespBody
            The prediction response.
        """
        # Load back the model information
        keys = list(ExtraAttributes.__fields__)
        _, model_bytes, extra_atts = ModelStore(
            name=f"{self.name}_{model_id}",  # There can be more than 1 LGBM speed model, gotta make sure the model names are distinct.
            minio_client=self.minio_client,
            collection=self.collection,
        ).load(model_id, keys)
        if model_bytes is not None:
            model = pickle.loads(model_bytes)
        else:
            raise ValueError("Failed to load model: model_bytes is None")

        extra_attributes = ExtraAttributes.from_dict(**extra_atts)
        final_model_data = extra_attributes.final_model_data
        scaler = extra_attributes.scaler
        target_pivot = extra_attributes.meta.get("target_pivot")
        last_date = extra_attributes.meta.get("last_date")
        prediction_normalization_statistics: Dict[
            str, Dict[str, float]
        ] = extra_attributes.prediction_normalization_statistics
        print(extra_attributes.meta)

        prediction_input_uri = MinioURI.parse(prediction_input)
        if is_batch_predict:
            df = pd.read_json(
                BytesIO(
                    self.minio_client.get_object(
                        prediction_input_uri.bucket, prediction_input_uri.key
                    ).data
                ),
                orient="records",
            ).pipe(cast_df_types, training_feature_types)
        else:
            df = get_df(prediction_input_uri, self.minio_client, feature_types)

        yhat_array = model.predict(np.nan_to_num(final_model_data, nan=0))
        if len(time_groups) > 0:
            yhat_array = from_y_shape(yhat_array, forecast_horizon, target_pivot)[0]
        yhat = scaler.inverse_transform(yhat_array)
        # create a date periods for this experiments
        dates_range = pd.date_range(
            start=last_date,
            periods=(forecast_horizon + gap + 1),
            freq=TimeUnit(time_unit).to_resample_rule(),
            inclusive="right",
        )
        focus_dates = dates_range[-forecast_horizon:]
        if not time_groups:
            pred_df = pd.DataFrame(
                {
                    datetime_column: focus_dates,
                    PREDICTION_COLNAME: yhat.ravel(),
                }
            )
            pred_df_merge = pred_df.merge(df, on=datetime_column, how="right")
            pred_df_merge.sort_values(by=[datetime_column], inplace=True)
            # impute
            # ffill: impute last valid observation to next one
            # bfill: impute next vaild observation to last
            pred_df_merge[PREDICTION_COLNAME] = (
                pred_df_merge[PREDICTION_COLNAME].ffill().bfill()
            )
        else:
            # unpivot(stack) predictions, in order to combine prediction_json
            pred_df = pd.DataFrame(yhat)
            if len(time_groups) == 1:
                pred_df.columns = pd.Index(target_pivot, name=time_groups[0])
            else:
                pred_df.columns = pd.MultiIndex.from_tuples(
                    target_pivot, names=time_groups
                )
            pred_df.index = focus_dates
            pred_df.index.names = [datetime_column]
            pred_df = pred_df.stack(level=time_groups).reset_index(
                name=PREDICTION_COLNAME
            )
            pred_df = cast_df_types(pred_df, training_feature_types)
            pred_df_merge = pred_df.merge(
                df,
                on=time_groups + [datetime_column],
                how="right",
            )
            # impute
            # Test dataset's time groups don't appear in train datasets.
            # impute mean by datetime column
            pred_df_merge[PREDICTION_COLNAME] = pred_df_merge.groupby(datetime_column)[
                PREDICTION_COLNAME
            ].transform(lambda x: x.fillna(x.mean()))
            # ffill: impute last valid observation to next one by time groups
            # bfill: impute next vaild observation to last one by time groups
            pred_df_merge[PREDICTION_COLNAME] = pred_df_merge.groupby(time_groups)[
                PREDICTION_COLNAME
            ].transform(lambda x: x.ffill().bfill())

        # fill mean of model prediction to unexpected time interval
        pred_df_merge[PREDICTION_COLNAME].fillna(np.mean(yhat), inplace=True)

        # Get list of columns to keep for displaying
        keep_columns = get_prediction_header(
            target.name in df.columns,
            keep_columns,
            datetime_column,
            time_groups,
            target.name,
        )

        # generate predict result
        prediction_result_df = get_prediction_result(
            pred_df_merge,
            target.name,
            keep_columns,
            time_groups,
            datetime_column,
        )

        # Make the prediction results non-negative if needed
        if non_negative:
            print("Make prediction results non-negative.")
            prediction_result_df = non_negative_prediction_handler(
                prediction_result_df, time_groups
            )

        # Normalize the prediction results
        print("Normalize prediction results.")
        prediction_result_df = normalize_prediction_result(
            prediction_result_df,
            prediction_normalization_statistics,
            time_groups,
            self.n_jobs,
        )

        # Sort the prediction result data by time group columns and datetime column
        print("Sort prediction results.")
        prediction_result_df = prediction_result_df.sort_values(
            by=time_groups + [datetime_column],
        ).reset_index(drop=True)

        print(f"Successfully run the {self.name} prediction process.")

        if is_batch_predict:
            return prediction_result_df.to_json(orient="records")
        else:
            if prediction_output_uri:
                prediction_result_df.to_csv(
                    prediction_output_uri,
                    index=False,
                    storage_options=self.storage_options,
                )
                return PredictRespBody(prediction=prediction_output_uri)
            # Generate the prediction response, then return it!
            return PredictRespBody.create(
                self.name, prediction_result_df, self.minio_client
            )
