import warnings
from typing import List

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", FutureWarning)

from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential

from src.training.utils import generate_model, split_train_to_xy

UPPER_ROW_LIMIT = 500
LOWER_ROW_LIMIT = 30


def train_lstm_model(
    train_scaled: np.ndarray,
    early_stop: bool,
    derivation_window: int,
    forecast_horizon: int,
    gap: int,
    time_groups: List[str],
) -> Sequential:
    """
    Trains a Keras.Sequential.
    """
    # both x_train and y_train are dimension 3 arrays derived from training set
    x_train, y_train, x_val, y_val, early_stop = split_train_to_xy(
        train_scaled, early_stop, derivation_window, forecast_horizon, gap
    )
    clear_session()

    with warnings.catch_warnings():
        model = generate_model(
            x_train=x_train,
            y_train=y_train,
            x_validate=x_val,
            y_validate=y_val,
            early_stop=early_stop,
            time_groups=time_groups,
        )
    return model


def early_stop_conditions(
    df: pd.DataFrame,
    time_groups: List[str],
) -> bool:
    """
    According to different conditions to judge early_stop
    """
    if df.shape[0] <= LOWER_ROW_LIMIT:
        early_stop = False
    elif (len(time_groups) > 0) or (df.shape[0] > UPPER_ROW_LIMIT):
        early_stop = True
    else:
        early_stop = False
    return early_stop
