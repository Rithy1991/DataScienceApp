"""
Sample data generators for demonstration and testing.
Provides realistic datasets for quick onboarding and feature showcasing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple
import streamlit as st


@st.cache_data
def generate_sales_data(n_records: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate sample e-commerce sales dataset."""
    np.random.seed(seed)

    dates = pd.date_range("2023-01-01", periods=n_records, freq="h")
    regions = ["North", "South", "East", "West"]
    categories = ["Electronics", "Clothing", "Home", "Sports", "Books"]

    df = pd.DataFrame({
        "date": dates,
        "region": np.random.choice(regions, n_records),
        "category": np.random.choice(categories, n_records),
        "customer_id": np.random.randint(1000, 9999, n_records),
        "quantity": np.random.randint(1, 20, n_records),
        "price_per_unit": np.round(np.random.exponential(50, n_records) + 10, 2),
        "discount": np.round(np.random.uniform(0, 0.3, n_records), 2),
        "is_weekend": [d.dayofweek >= 5 for d in dates],
    })

    df["total_sale"] = df["quantity"] * df["price_per_unit"] * (1 - df["discount"])
    return df.round(2)


@st.cache_data
def generate_timeseries_data(n_points: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate sample time series (e.g., stock price, sensor readings)."""
    np.random.seed(seed)

    dates = pd.date_range("2023-01-01", periods=n_points, freq="D")
    trend = np.linspace(100, 150, n_points)
    seasonality = 20 * np.sin(np.arange(n_points) * 2 * np.pi / 52)
    noise = np.random.normal(0, 5, n_points)
    value = trend + seasonality + noise

    df = pd.DataFrame({
        "date": dates,
        "value": np.maximum(value, 10),  # Ensure positive
        "volume": np.random.randint(100000, 1000000, n_points),
        "ma_7": pd.Series(value).rolling(7, min_periods=1).mean(),
    })

    return df.round(2)


@st.cache_data
def generate_iris_like_data(n_records: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate sample classification dataset (iris-like)."""
    np.random.seed(seed)

    n_per_class = n_records // 3
    classes = ["class_a", "class_b", "class_c"]

    dfs = []
    for i, cls in enumerate(classes):
        # For last class, use remaining records to ensure exact total
        n_samples = n_records - (n_per_class * (len(classes) - 1)) if i == len(classes) - 1 else n_per_class
        mean = [5 + i, 3 + i, 1.5 + i, 0.3 + i]
        cov = np.eye(4) * 0.5
        features = np.random.multivariate_normal(mean, cov, n_samples)
        df_cls = pd.DataFrame(
            features,
            columns=["feature_1", "feature_2", "feature_3", "feature_4"],
        )
        df_cls["target"] = cls
        dfs.append(df_cls)

    df = pd.concat(dfs, ignore_index=True)
    return df.round(3)


@st.cache_data
def generate_housing_data(n_records: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate sample regression dataset (housing prices)."""
    np.random.seed(seed)

    df = pd.DataFrame({
        "square_feet": np.random.uniform(800, 5000, n_records),
        "bedrooms": np.random.choice([1, 2, 3, 4, 5], n_records),
        "bathrooms": np.random.choice([1, 1.5, 2, 2.5, 3], n_records),
        "year_built": np.random.randint(1980, 2023, n_records),
        "garage_spaces": np.random.choice([0, 1, 2, 3], n_records),
        "lot_size": np.random.uniform(0.1, 2.0, n_records),
    })

    # Synthetic price generation (for demo)
    df["price"] = (
        50000
        + df["square_feet"] * 150
        + df["bedrooms"] * 50000
        + df["bathrooms"] * 30000
        + (2023 - df["year_built"]) * -500
        + df["garage_spaces"] * 25000
        + np.random.normal(0, 100000, n_records)
    )

    df["price"] = np.maximum(df["price"], 100000)
    return df.round(2)


@st.cache_data
def generate_customer_churn_data(n_records: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate sample binary classification dataset (customer churn)."""
    np.random.seed(seed)

    df = pd.DataFrame({
        "customer_id": np.arange(1, n_records + 1),
        "tenure_months": np.random.randint(1, 72, n_records),
        "monthly_charges": np.random.uniform(20, 150, n_records),
        "total_charges": np.random.uniform(100, 10000, n_records),
        "contract_type": np.random.choice(["month_to_month", "one_year", "two_year"], n_records),
        "tech_support": np.random.choice(["Yes", "No"], n_records),
        "online_backup": np.random.choice(["Yes", "No"], n_records),
    })

    # Generate churn based on features (synthetic)
    churn_prob = (
        0.5
        - (df["tenure_months"] / 100)
        + (df["monthly_charges"] / 200)
        + 0.1 * (df["contract_type"] == "month_to_month")
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    df["churn"] = (np.random.random(n_records) < churn_prob).astype(int)

    return df.round(2)


@st.cache_data
def get_sample_datasets() -> dict[str, pd.DataFrame]:
    """Return a dictionary of all available sample datasets."""
    return {
        "Sales Data": generate_sales_data(),
        "Time Series": generate_timeseries_data(),
        "Classification": generate_iris_like_data(),
        "Housing Prices": generate_housing_data(),
        "Customer Churn": generate_customer_churn_data(),
    }


def describe_sample_dataset(name: str) -> str:
    """Return a description of a sample dataset."""
    descriptions = {
        "Sales Data": "E-commerce transaction data with regional, temporal, and categorical features. Useful for regression/forecasting.",
        "Time Series": "Daily price/value data with volume and moving averages. Good for forecasting and trend analysis.",
        "Classification": "Iris-like multi-feature classification dataset. Ideal for testing classifiers.",
        "Housing Prices": "Real estate data with physical attributes and prices. Perfect for regression modeling.",
        "Customer Churn": "Telecom customer data with churn labels. Great for binary classification demos.",
    }
    return descriptions.get(name, "")
