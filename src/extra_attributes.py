import json
import pickle
from io import StringIO
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder, StandardScaler


class ExtraAttributes(BaseModel):
    final_model_data: np.ndarray
    scaler: StandardScaler
    meta: Dict[str, Any]
    predictions: Dict[str, List[List[float]]]
    prediction_normalization_statistics: Dict[str, Dict[str, float]]

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def keys() -> Dict[str, str]:
        """
        FIXME fix exodusutils so that we don't need this
        """
        _keys = ExtraAttributes.__fields__.keys()
        return dict(zip(_keys, _keys))

    def to_dict(self) -> Dict[str, bytes]:
        return {
            "final_model_data": pickle.dumps(self.final_model_data),
            "scaler": pickle.dumps(self.scaler),
            "meta": json.dumps(self.meta).encode("utf-8"),
            "predictions": json.dumps(self.predictions).encode(),
            "prediction_normalization_statistics": json.dumps(
                self.prediction_normalization_statistics
            ).encode(),
        }

    @classmethod
    def from_dict(
        cls,
        final_model_data: bytes,
        scaler: bytes,
        meta: bytes,
        predictions: bytes,
        prediction_normalization_statistics: bytes = b"{}",  # The default value is used to handle the condition that the older version of the extra attributes doesn't have `prediction_normalization_statistics` (before Exodus 1.22.0)
    ):
        return cls(
            final_model_data=pickle.loads(final_model_data),
            scaler=pickle.loads(scaler),
            meta=json.loads(meta.decode("utf-8")),
            predictions=json.loads(predictions.decode()),
            prediction_normalization_statistics=json.loads(
                prediction_normalization_statistics.decode()
            ),
        )
