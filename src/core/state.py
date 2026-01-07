from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from collections.abc import MutableMapping

import pandas as pd


DATA_KEY = "dsai_df"
CLEAN_KEY = "dsai_df_clean"
DATA_SOURCE_KEY = "dsai_data_source"
MODEL_ACTIVE_KEY = "dsai_active_model"
MODEL_ACTIVE_META_KEY = "dsai_active_model_meta"


@dataclass
class ActiveModel:
    model_type: str  # tabular|forecast
    path: str


def get_df(session_state: MutableMapping[Any, Any]) -> Optional[pd.DataFrame]:
    df = session_state.get(DATA_KEY)
    return df if isinstance(df, pd.DataFrame) else None


def set_df(session_state: MutableMapping[Any, Any], df: pd.DataFrame, source: str) -> None:
    session_state[DATA_KEY] = df
    session_state[DATA_SOURCE_KEY] = source


def get_clean_df(session_state: MutableMapping[Any, Any]) -> Optional[pd.DataFrame]:
    df = session_state.get(CLEAN_KEY)
    return df if isinstance(df, pd.DataFrame) else None


def set_clean_df(session_state: MutableMapping[Any, Any], df: pd.DataFrame) -> None:
    session_state[CLEAN_KEY] = df
