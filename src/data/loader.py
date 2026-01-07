from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests
import streamlit as st


@dataclass
class LoadResult:
    df: pd.DataFrame
    source: str
    meta: Dict[str, Any]


@st.cache_data
def load_from_upload(uploaded_file) -> Optional[LoadResult]:
    if uploaded_file is None:
        return None

    name = getattr(uploaded_file, "name", "uploaded")
    lower = str(name).lower()

    content = uploaded_file.getvalue()
    bio = io.BytesIO(content)

    if lower.endswith(".csv"):
        df = pd.read_csv(bio)
        return LoadResult(df=df, source=f"file:{name}", meta={"format": "csv"})

    if lower.endswith(".parquet"):
        df = pd.read_parquet(bio)
        return LoadResult(df=df, source=f"file:{name}", meta={"format": "parquet"})

    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        df = pd.read_excel(bio)
        return LoadResult(df=df, source=f"file:{name}", meta={"format": "excel"})

    return None


@st.cache_data
def load_from_api(url: str, headers: Optional[Dict[str, str]] = None, timeout_seconds: int = 20) -> LoadResult:
    resp = requests.get(url, headers=headers or {}, timeout=timeout_seconds)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "application/json" in content_type or resp.text.strip().startswith("{") or resp.text.strip().startswith("["):
        data = resp.json()
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            df = pd.DataFrame(data["data"])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.json_normalize(data)
        else:
            df = pd.DataFrame()
        return LoadResult(df=df, source=f"api:{url}", meta={"format": "json"})

    # fallback: treat as CSV
    df = pd.read_csv(io.StringIO(resp.text))
    return LoadResult(df=df, source=f"api:{url}", meta={"format": "csv"})


def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
