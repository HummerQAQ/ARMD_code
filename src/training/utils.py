from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi.logger import logger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

EPOCHS = 200
BATCH_SIZE = 32
OPTIMIZER = "adam"
LOSS_FUNCTION = "mse"
ACTIVATION = "relu"
RECURRENT_ACTIVATION = "sigmoid"
EARLY_STOP_MONITOR = "val_loss"


def df_to_train_test_sets(
    df: pd.DataFrame,
    fold: int,
    forecast_horizon: int,
    gap: int,
    target_pivot: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    #             [ ----------- TRAIN DATA ----------- | ----- IGNORED ----- | ------- TEST ------- ]
    # lengths:       (total - forecast_horizon - gap)           gap              forecast_horizon
    # preconditions:
    #   1. no duplicate dates
    #   2. dates are already sorted
    horizon_by_fold = forecast_horizon * (fold + 1)
    rows = df.shape[0]
    values = df.loc[:, target_pivot].values

    train_start = 0
    train_end = rows - gap - horizon_by_fold
    train = values[train_start:train_end]

    test_start = train_end + gap
    test_end = test_start + forecast_horizon
    test = values[test_start:test_end]
    return train, test


def split_train_to_xy(
    training_array: np.ndarray,
    early_stop: bool,
    derivation_window: int,
    forecast_horizon: int,
    gap: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], bool]:
    """
    - Output:
        - if early stopping is True, will return 4 array: Train_X, Train_y, Validation_X and Validaiton_y.
        - if early stopping is False, will return 2 array: Train_X and Train_y.
    """
    # check if each row has correct number of cells
    rows = len(training_array)
    x_list = list()
    y_list = list()
    for x_start in range(rows):
        x_end = x_start + derivation_window
        y_start = x_end + gap
        y_end = y_start + forecast_horizon
        if y_end > len(training_array):
            break
        x_list.append(training_array[x_start:x_end])
        y_list.append(training_array[y_start:y_end])
    x: np.ndarray = np.array(x_list)
    y: np.ndarray = np.array(y_list)
    val_range = forecast_horizon
    if early_stop:
        if val_range < x.shape[0]:
            return (
                x[:-val_range, :],
                y[:-val_range, :],
                x[-val_range:, :],
                y[-val_range:, :],
                early_stop,
            )
        else:
            logger.warn("Cannot take validation set, will not stop early")
            early_stop = False
            return x, y, None, None, early_stop
    else:
        return x, y, None, None, early_stop


def to_y_shape(arr: np.ndarray) -> np.ndarray:
    return np.reshape(arr, (arr.shape[0], -1))


def from_y_shape(
    arr: np.ndarray, forecast_horizon: int, target_pivot: List[str]
) -> np.ndarray:
    return np.reshape(arr, (arr.shape[0], forecast_horizon, len(target_pivot)))


def generate_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validate: Optional[np.ndarray],
    y_validate: Optional[np.ndarray],
    early_stop: bool,
    time_groups: List[str],
) -> Sequential:
    model = Sequential()
    shuffle = time_groups is not None
    return_sequences = time_groups is None
    if time_groups is not None:
        y_train = to_y_shape(y_train)
        if early_stop and y_validate is not None:
            y_validate = to_y_shape(y_validate)

    model.add(
        LSTM(
            EPOCHS,
            activation=ACTIVATION,
            recurrent_activation=RECURRENT_ACTIVATION,
            return_sequences=return_sequences,
            input_shape=(x_train.shape[1], x_train.shape[2]),
        )
    )
    if time_groups is None:
        model.add(
            LSTM(
                EPOCHS, activation=ACTIVATION, recurrent_activation=RECURRENT_ACTIVATION
            )
        )
    else:
        model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1]))
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION)
    if early_stop:
        early_stop_cb = EarlyStopping(
            monitor=EARLY_STOP_MONITOR, min_delta=0, patience=10, verbose=0
        )
        model.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            validation_data=(x_validate, y_validate),
            callbacks=[early_stop_cb],
            shuffle=shuffle,
        )
    else:
        model.fit(x_train, y_train, epochs=EPOCHS, shuffle=shuffle)
    return model


def get_last_period_data(
    train_scaled: np.ndarray, derivation_window: int, target_pivot: List[str]
) -> np.ndarray:
    last_period_train_data = train_scaled[-derivation_window:].reshape(
        (1, derivation_window, len(target_pivot))
    )
    return last_period_train_data


def get_params(early_stop) -> Dict[str, Any]:
    return {
        "Epochs": EPOCHS,
        "Batch_size": BATCH_SIZE,
        "Optimizer": OPTIMIZER,
        "Loss_function": LOSS_FUNCTION,
        "Activation": ACTIVATION,
        "Recurrent_Activation": RECURRENT_ACTIVATION,
        "Standardization": True,
        "Early_Stopping": early_stop,
    }


def get_keep_columns(
    has_answer: bool,
    col_target: str,
    col_datetime: str,
    time_groups: list,
    col_prediction: str,
) -> List[str]:
    """
    Construct columns to present in prediction result
    - Input:
        - df_test_template (pandas.DataFrame): Prediction json file template
        - col_target (str): Target column
        - col_datetime (str): Datetime column
        - time_groups (List[str]): Time group columns
        - col_prediction (str): Column name for prediction result
    - Output:
        - keep_columns (List[str]): Columns to present in prediction result
    """
    target_column = [col_target] if has_answer else []
    return [col_datetime] + time_groups + target_column + [col_prediction]


def get_prediction_result(
    df: pd.DataFrame,
    col_target: str,
    keep_columns: list,
    time_groups: list,
    date_column: str,
) -> pd.DataFrame:
    """
    Construct pandas.DataFrame as output when making predictions
    - Input:
        - df (pandas.DataFrame): Testing set
        - col_target (str): Target column
        - keep_columns (List[str]): Columns to present in prediction result
        - time_groups (List[str]): Time group columns
        - date_column (str): Datetime column
    - Output:
        - result (pandas.DataFrame): Output presented when making predictions
    """
    result = df[[col for col in keep_columns if col in df.columns]].copy()

    result.sort_values(time_groups + [date_column], inplace=True)

    if col_target in keep_columns:
        result[col_target].replace(np.nan, "", inplace=True)
    # fix the date column's type to `str` so that it is json compatible
    result[date_column] = result[date_column].astype(str)
    return result
