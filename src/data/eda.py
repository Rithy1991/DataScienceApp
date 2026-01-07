from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st


@dataclass
class EdaSummary:
    n_rows: int
    n_cols: int
    missing_total: int
    missing_by_col: Dict[str, int]
    numeric_cols: List[str]
    categorical_cols: List[str]


@st.cache_data
def summarize(df: pd.DataFrame) -> EdaSummary:
    missing_by_col = df.isna().sum().to_dict()
    missing_total = int(sum(missing_by_col.values()))
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return EdaSummary(
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        missing_total=missing_total,
        missing_by_col={str(k): int(v) for k, v in missing_by_col.items()},
        numeric_cols=[str(c) for c in numeric_cols],
        categorical_cols=[str(c) for c in categorical_cols],
    )


@st.cache_data
def correlation_matrix(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    numeric: pd.DataFrame = df.select_dtypes(include=["number"]).copy()
    if cols:
        selected_cols = [c for c in cols if c in numeric.columns]
        numeric = numeric.reindex(columns=selected_cols)
    if numeric.shape[1] == 0:
        return pd.DataFrame()
    # Keep compatible across pandas versions and stubs.
    corr = numeric.corr(numeric_only=True)
    return corr


@st.cache_data
def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number"]).copy()
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe().T


@st.cache_data
def value_counts(df: pd.DataFrame, col: str, top_n: int = 20) -> pd.DataFrame:
    vc = df[col].astype(str).value_counts(dropna=False).head(top_n)
    return pd.DataFrame({"value": vc.index.astype(str), "count": vc.values})


@st.cache_data
def detect_anomaly_zscore(df: pd.DataFrame, cols: List[str], z: float = 3.0) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(columns=["row", "column", "value", "zscore"])

    records = []
    for c in cols:
        s: pd.Series = pd.Series(pd.to_numeric(df[c], errors="coerce"), index=df.index)
        if bool(pd.isna(s).all()):
            continue
        mean = float(np.nanmean(s))
        std = float(np.nanstd(s))
        if std == 0 or not np.isfinite(std):
            continue
        zscores: pd.Series = (s - mean) / std
        idx: pd.Series = (zscores.abs() > z).fillna(False)
        for i in df.index[idx]:
            records.append({"row": int(i), "column": c, "value": df.loc[i, c], "zscore": float(zscores.loc[i])})

    return pd.DataFrame.from_records(records)
