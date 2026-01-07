from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


@dataclass
class CleaningReport:
    missing_before: int
    missing_after: int
    outliers_capped: int
    columns_dropped: List[str]
    notes: List[str]


@st.cache_data
def infer_datetime_columns(df: pd.DataFrame, sample_size: int = 200) -> List[str]:
    candidates: List[str] = []
    for col in df.columns:
        if df[col].dtype == "object":
            s = df[col].dropna().astype(str).head(sample_size)
            if s.empty:
                continue
            parsed: pd.Series = pd.to_datetime(s, errors="coerce", utc=False)  # type: ignore[assignment]
            if float(parsed.notna().mean()) > 0.8:
                candidates.append(col)
    return candidates


@st.cache_data
def apply_datetime_parsing(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


@st.cache_data
def fill_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",  # mean|median|zero
    categorical_strategy: str = "mode",  # mode|unknown
) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in out.columns if c not in num_cols]

    for col in num_cols:
        if numeric_strategy == "mean":
            value = out[col].mean()
        elif numeric_strategy == "zero":
            value = 0
        else:
            value = out[col].median()
        out[col] = out[col].fillna(value)

    for col in cat_cols:
        if categorical_strategy == "unknown":
            value = "Unknown"
        else:
            mode = out[col].mode(dropna=True)
            value = mode.iloc[0] if len(mode) else "Unknown"
        out[col] = out[col].fillna(value)

    return out


def cap_outliers_iqr(df: pd.DataFrame, columns: Optional[List[str]] = None, factor: float = 1.5) -> Tuple[pd.DataFrame, int]:
    out = df.copy()
    cols = columns or out.select_dtypes(include=["number"]).columns.tolist()
    capped = 0
    for col in cols:
        s = out[col].astype(float)
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            continue
        lo = q1 - factor * iqr
        hi = q3 + factor * iqr
        before = s.copy()
        out[col] = s.clip(lower=lo, upper=hi)
        capped += int((before != out[col]).sum())
    return out, capped


def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 0.6) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    to_drop: List[str] = []
    for col in out.columns:
        ratio = float(out[col].isna().mean())
        if ratio >= threshold:
            to_drop.append(col)
    if to_drop:
        out = out.drop(columns=to_drop)
    return out, to_drop


def basic_feature_engineering(df: pd.DataFrame, datetime_cols: Optional[List[str]] = None) -> pd.DataFrame:
    out = df.copy()
    dt_cols = datetime_cols or []
    for col in dt_cols:
        if col not in out.columns:
            continue
        if not pd.api.types.is_datetime64_any_dtype(out[col]):
            continue
        dt_idx = pd.DatetimeIndex(pd.to_datetime(out[col], errors="coerce"))
        out[f"{col}__year"] = dt_idx.year
        out[f"{col}__month"] = dt_idx.month
        out[f"{col}__day"] = dt_idx.day
        out[f"{col}__dayofweek"] = dt_idx.dayofweek
        out[f"{col}__hour"] = dt_idx.hour
    return out


def validate_dataframe(df: pd.DataFrame) -> Dict[str, str]:
    issues: Dict[str, str] = {}
    if df.empty:
        issues["empty"] = "DataFrame is empty"
    if df.columns.duplicated().any():
        issues["duplicate_columns"] = "Duplicate column names detected"
    if any(str(c).strip() == "" for c in df.columns):
        issues["blank_columns"] = "Blank column name detected"
    return issues


def clean_pipeline(
    df: pd.DataFrame,
    drop_missing_threshold: float,
    numeric_strategy: str,
    categorical_strategy: str,
    outlier_cap: bool,
    outlier_factor: float,
    parse_datetimes: bool,
    datetime_cols: Optional[List[str]] = None,
    create_datetime_features: bool = True,
) -> Tuple[pd.DataFrame, CleaningReport]:
    notes: List[str] = []
    missing_before = int(df.isna().sum().sum())

    out, dropped = drop_high_missing_columns(df, threshold=drop_missing_threshold)
    if dropped:
        notes.append(f"Dropped {len(dropped)} columns with >= {drop_missing_threshold:.0%} missing")

    dt_cols_raw = datetime_cols or infer_datetime_columns(out)
    dt_cols = [c for c in dt_cols_raw if c in out.columns]
    dropped_dt_cols = [c for c in dt_cols_raw if c not in out.columns]
    if parse_datetimes and dt_cols:
        out = apply_datetime_parsing(out, dt_cols)
        notes.append(f"Parsed datetime columns: {', '.join(dt_cols[:8])}{'...' if len(dt_cols) > 8 else ''}")
    elif dropped_dt_cols:
        notes.append(f"Skipped datetime parsing for dropped columns: {', '.join(dropped_dt_cols[:8])}{'...' if len(dropped_dt_cols) > 8 else ''}")

    out = fill_missing_values(out, numeric_strategy=numeric_strategy, categorical_strategy=categorical_strategy)

    outliers_capped = 0
    if outlier_cap:
        out, outliers_capped = cap_outliers_iqr(out, factor=outlier_factor)
        if outliers_capped:
            notes.append(f"Capped {outliers_capped} outlier values using IQR")

    if create_datetime_features and dt_cols:
        out = basic_feature_engineering(out, datetime_cols=dt_cols)
        notes.append("Added datetime-derived features")

    missing_after = int(out.isna().sum().sum())

    report = CleaningReport(
        missing_before=missing_before,
        missing_after=missing_after,
        outliers_capped=outliers_capped,
        columns_dropped=dropped,
        notes=notes,
    )
    return out, report
