"""
Real-world end-to-end lab scenarios for Data Science Academy.
Demonstrates full workflows: load → clean → explore → model → evaluate → deploy.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_ecommerce_sales_data(n=1000, seed=42):
    """
    Real-world e-commerce dataset: daily sales, customers, products, revenue.
    Common issues: missing values, outliers, date parsing, categorical encoding.
    """
    np.random.seed(seed)
    
    dates = [datetime.now() - timedelta(days=int(x)) for x in np.random.exponential(100, n)]
    dates.sort()
    
    data = {
        'date': dates,
        'customer_id': np.random.randint(1000, 5000, n),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports', None], n, p=[0.3, 0.3, 0.2, 0.15, 0.05]),
        'order_value': np.random.exponential(100, n) + 20,
        'units_sold': np.random.randint(1, 20, n),
        'discount_pct': np.random.choice([0, 5, 10, 15, 20, 25, None], n, p=[0.4, 0.2, 0.15, 0.1, 0.08, 0.05, 0.02]),
        'customer_age': np.random.randint(18, 75, n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n),
        'is_returning': np.random.choice([True, False], n, p=[0.4, 0.6]),
    }
    
    # Add some realistic outliers
    outlier_idx = np.random.choice(n, size=int(0.05 * n), replace=False)
    for idx in outlier_idx:
        data['order_value'][idx] *= np.random.uniform(5, 10)
    
    # Add some missing values
    missing_idx = np.random.choice(n, size=int(0.03 * n), replace=False)
    data['discount_pct'] = [data['discount_pct'][i] if i not in missing_idx else np.nan for i in range(n)]
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['order_value'] = df['order_value'].round(2)
    
    return df


def generate_customer_churn_data(n=500, seed=42):
    """
    Real-world SaaS customer churn dataset: subscription, usage, support.
    Common issues: class imbalance, feature scaling, categorical encoding.
    """
    np.random.seed(seed)
    
    tenure_months = np.random.exponential(12, n)
    monthly_spend = np.abs(np.random.normal(50, 30, n))
    
    data = {
        'customer_id': range(1, n + 1),
        'tenure_months': tenure_months,
        'monthly_spend': monthly_spend,
        'total_support_tickets': np.random.poisson(5, n),
        'response_time_hours': np.random.exponential(24, n),
        'product_upgrades': np.random.poisson(2, n),
        'login_count_30d': np.random.poisson(10, n),
        'contract_type': np.random.choice(['Monthly', 'Annual'], n, p=[0.6, 0.4]),
        'industry': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Retail'], n),
    }
    
    df = pd.DataFrame(data)
    
    # Create churn target (imbalanced, ~15% churn)
    churn_prob = (
        0.8 * np.exp(-df['tenure_months'] / 12) +  # Longer tenure = less likely to churn
        0.5 * (1 / (1 + np.exp(df['monthly_spend'] / 100))) +  # Higher spend = less likely
        0.1 * (df['response_time_hours'] / 48) +  # Slow support → more churn
        np.random.normal(0, 0.05, n)  # Random noise
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    df['churned'] = (np.random.random(n) < churn_prob).astype(int)
    
    return df


def generate_time_series_website_traffic(n=365, seed=42):
    """
    Real-world website traffic data: daily visitors, pages, bounce rate, device.
    Common issues: seasonality, trends, missing dates, categorical patterns.
    """
    np.random.seed(seed)
    
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
    
    # Trend: growing traffic
    trend = np.linspace(1000, 3000, n)
    
    # Seasonality: weekly pattern (weekends lower)
    seasonal = np.where(dates.dayofweek < 5, 500, 200)  # Weekday vs weekend
    
    # Random noise
    noise = np.random.normal(0, 200, n)
    
    visitors = np.maximum(trend + seasonal + noise, 100).astype(int)
    
    data = {
        'date': dates,
        'visitors': visitors,
        'page_views': visitors * np.random.uniform(2, 5, n),
        'bounce_rate': np.clip(np.random.beta(2, 5, n) * 100, 10, 90),
        'avg_session_duration': np.random.exponential(3, n),  # minutes
        'device_mobile': np.random.binomial(visitors, 0.6),
        'device_desktop': np.random.binomial(visitors, 0.3),
        'device_tablet': np.random.binomial(visitors, 0.1),
        'traffic_source': np.random.choice(['Organic', 'Paid', 'Direct', 'Referral'], n, p=[0.5, 0.2, 0.2, 0.1]),
    }
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Add missing date (simulate data collection gap)
    gap_idx = np.random.choice(n, 5, replace=False)
    df = df.drop(gap_idx).reset_index(drop=True)
    
    return df


def generate_housing_price_data(n=300, seed=42):
    """
    Real-world housing dataset: features predict price.
    Common issues: multicollinearity, outliers, non-linear relationships.
    """
    np.random.seed(seed)
    
    sqft = np.random.uniform(1000, 5000, n)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3], n, p=[0.15, 0.2, 0.4, 0.15, 0.1])
    age_years = np.random.uniform(0, 50, n)
    garage_spaces = np.random.choice([0, 1, 2, 3], n, p=[0.1, 0.3, 0.4, 0.2])
    
    # Price based on features + noise + outliers
    price = (
        200 * sqft +  # Price per sqft
        50000 * bedrooms +
        30000 * bathrooms -
        1000 * age_years +
        20000 * garage_spaces +
        np.random.normal(0, 50000, n)
    )
    
    # Add some outliers (luxury homes)
    outlier_idx = np.random.choice(n, 10, replace=False)
    price[outlier_idx] *= np.random.uniform(1.5, 3, 10)
    
    data = {
        'square_feet': sqft,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age_years': age_years,
        'garage_spaces': garage_spaces,
        'lot_size': np.random.uniform(0.1, 2, n),  # acres
        'has_pool': np.random.choice([0, 1], n, p=[0.8, 0.2]),
        'neighborhood': np.random.choice(['Downtown', 'Suburbs', 'Rural', 'Waterfront'], n),
        'price': np.maximum(price, 50000),  # Minimum price
    }
    
    return pd.DataFrame(data)


# Export all generators
LABS = {
    'E-Commerce Sales': {
        'generator': generate_ecommerce_sales_data,
        'description': 'Predict daily sales & identify product performance.',
        'key_issues': ['Missing values (discount)', 'Outliers (high-value orders)', 'Date parsing', 'Categorical encoding'],
        'learning_goals': ['Clean messy real-world data', 'Handle time series', 'Feature engineering', 'Exploratory analysis'],
    },
    'Customer Churn': {
        'generator': generate_customer_churn_data,
        'description': 'Predict which customers will cancel subscriptions.',
        'key_issues': ['Class imbalance (~15% churn)', 'Feature scaling', 'Categorical encoding', 'Interpretation'],
        'learning_goals': ['Handle imbalanced data', 'Feature importance', 'Classification metrics', 'Business insights'],
    },
    'Website Traffic': {
        'generator': generate_time_series_website_traffic,
        'description': 'Forecast daily website visitors & understand traffic patterns.',
        'key_issues': ['Seasonality', 'Trend', 'Missing dates', 'Multicollinearity (device columns)'],
        'learning_goals': ['Time series analysis', 'Seasonality detection', 'Forecasting', 'Data gaps handling'],
    },
    'Housing Prices': {
        'generator': generate_housing_price_data,
        'description': 'Predict house prices from property features.',
        'key_issues': ['Outliers (luxury homes)', 'Multicollinearity', 'Non-linear relationships'],
        'learning_goals': ['Regression modeling', 'Outlier detection', 'Feature relationships', 'Scaling'],
    },
}


def get_lab(lab_name: str) -> pd.DataFrame:
    """Load a lab dataset by name."""
    if lab_name not in LABS:
        raise ValueError(f"Lab '{lab_name}' not found. Available: {list(LABS.keys())}")
    return LABS[lab_name]['generator']()
