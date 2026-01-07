
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

# Mock Streamlit for imports that rely on it
import streamlit
class MockSessionState(dict):
    pass
streamlit.session_state = MockSessionState()

def generate_housing_data(n_records=500, seed=42):
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

def run_test():
    print("🚀 Starting User Workflow Simulation: Housing Prices")
    print("-" * 50)

    # 1. Ingestion
    print("\n1️⃣ Ingestion Phase")
    df = generate_housing_data()
    print(f"Loaded Housing Data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head(3))

    # 2. Cleaning
    print("\n2️⃣ Cleaning Phase")
    from src.data.cleaning import clean_pipeline
    # Simulate user selecting parameters
    df_clean, report = clean_pipeline(
        df,
        drop_missing_threshold=0.6,
        numeric_strategy="median",
        categorical_strategy="mode",  # Added
        outlier_cap=True,
        outlier_factor=3.0,
        parse_datetimes=True,         # Added
        datetime_cols=[],             # Added
        create_datetime_features=True # Added
    )
    print(f"Cleaning complete. Missing before: {report.missing_before}, after: {report.missing_after}")

    # 3. Feature Engineering
    print("\n3️⃣ Feature Engineering Phase (Simulation)")
    # Logic from Page 4: Create new feature
    df_clean["age"] = 2025 - df_clean["year_built"]
    df_clean["price_per_sqft"] = df_clean["price"] / df_clean["square_feet"]
    print("Added features: 'age', 'price_per_sqft'")
    
    # We must drop target from engineering if we want to simulate properly, but for ML training we need target
    # Usually we don't put target-derived features (leakage) into X, but price_per_sqft definitely leaks price.
    # Let's drop price_per_sqft before training or just not use it.
    # Actually, a user might mistakenly add it. The ML module should handle it or it will overfit.
    # We will remove the leak feature to be a "good user".
    df_clean = df_clean.drop(columns=["price_per_sqft"]) 
    print("Dropped leakage feature 'price_per_sqft' prior to modeling.")

    # 4. Modeling
    print("\n4️⃣ Modeling Phase (Tabular ML)")
    from src.ml.tabular import train_tabular
    from src.storage.model_registry import register_model
    
    target = "price"
    print(f"Training RandomForest on target: {target}")
    
    # Ensure directories exist
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Train
    result = train_tabular(
        df_clean,
        target_col=target,
        model_name="RandomForest",
        test_size=0.2,
        artifact_path="artifacts/test_housing_model.joblib"
    )
    
    print(f"Training completed in {result.meta['train_seconds']:.2f}s")
    print("Metrics:", result.metrics)
    
    # Register
    print("Registering model...")
    config_mock = type('Config', (), {'registry_dir': 'models', 'artifacts_dir': 'artifacts'})
    
    rec = register_model(
        registry_dir="models",
        artifacts_dir="artifacts",
        kind="tabular",
        name="HousingPriceRF",
        artifact_path=result.artifact_path,
        meta=result.meta
    )
    print(f"Model Registered: ID={rec.model_id}")

    # 5. Prediction
    print("\n5️⃣ Prediction Phase")
    from src.ml.tabular import predict_tabular
    import joblib
    
    # Load model
    model = joblib.load(rec.artifact_path)
    
    # Create sample new data (without target)
    new_data = df_clean.drop(columns=[target]).sample(5)
    print("Predicting for 5 random samples...")
    
    preds, proba = predict_tabular(model, new_data)
    
    result_df = new_data.copy()
    result_df["predicted_price"] = preds
    print(result_df[["square_feet", "bedrooms", "predicted_price"]])
    
    print("\n" + "="*50)
    print("✅ START-TO-FINISH WORKFLOW TEST PASSED")
    print("="*50)

if __name__ == "__main__":
    run_test()
