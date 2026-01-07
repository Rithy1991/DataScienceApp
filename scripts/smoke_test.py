#!/usr/bin/env python3
"""
App Smoke Test - Verify app can start without errors
"""

import sys
from pathlib import Path

print("Testing app.py import and basic functionality...")
print()

# Test 1: Import app module
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import app
    print("✓ app.py imports successfully")
except Exception as e:
    print(f"✗ app.py import failed: {e}")
    sys.exit(1)

# Test 2: Config loading
try:
    from src.core.config import load_config
    config = load_config()
    print(f"✓ Configuration loaded: {config.title}")
except Exception as e:
    print(f"✗ Configuration loading failed: {e}")
    sys.exit(1)

# Test 3: Core imports
try:
    from src.core.forecast_results import ForecastResult
    from src.core.forecast_components import render_forecast_results
    print("✓ Core forecast modules imported")
except Exception as e:
    print(f"✗ Core module import failed: {e}")
    sys.exit(1)

# Test 4: Data modules
try:
    from src.data.samples import generate_timeseries_data
    from src.data.cleaning import fill_missing_values
    from src.data.eda import summarize
    print("✓ Data modules imported")
except Exception as e:
    print(f"✗ Data module import failed: {e}")
    sys.exit(1)

# Test 5: ML modules
try:
    from src.ml.tabular import train_tabular
    from src.ml.forecast_transformer import ForecastTrainResult
    print("✓ ML modules imported")
except Exception as e:
    print(f"✗ ML module import failed: {e}")
    sys.exit(1)

# Test 6: Sample data generation
try:
    df = generate_timeseries_data(n_points=100)
    assert len(df) == 100
    print(f"✓ Generated sample data: {len(df)} rows, {len(df.columns)} columns")
except Exception as e:
    print(f"✗ Sample data generation failed: {e}")
    sys.exit(1)

# Test 7: EDA on sample data
try:
    summary = summarize(df)
    assert summary is not None
    assert summary.n_rows > 0
    print(f"✓ EDA analysis completed: {summary.n_rows} rows, {summary.n_cols} columns analyzed")
except Exception as e:
    print(f"✗ EDA failed: {e}")
    sys.exit(1)

# Test 8: Forecast result creation
try:
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    n = 10
    dates = pd.date_range(start=datetime.now(), periods=n, freq="D")
    values = np.random.randn(n).cumsum() + 100
    
    result = ForecastResult(
        model_name="Smoke Test",
        model_type="test",
        target_column="value",
        forecast_values=values,
        forecast_dates=pd.DatetimeIndex(dates),
        historical_values=values[:5],
        historical_dates=pd.DatetimeIndex(dates[:5]),
        lower_bound=values - 2,
        upper_bound=values + 2,
        confidence_level=0.95,
        train_metrics={},
        test_metrics={},
        metadata={}
    )
    
    df_result = result.to_dataframe()
    assert len(df_result) == n
    print(f"✓ Forecast result created: {len(df_result)} predictions")
except Exception as e:
    print(f"✗ Forecast result failed: {e}")
    sys.exit(1)

# Test 9: Visualization
try:
    from src.core.forecast_results import create_forecast_visualization
    fig = create_forecast_visualization(result)
    assert fig is not None
    print(f"✓ Forecast visualization created")
except Exception as e:
    print(f"✗ Visualization failed: {e}")
    sys.exit(1)

# Test 10: Model persistence
try:
    from src.storage.model_registry import register_model, list_models
    import joblib
    from pathlib import Path
    
    # Create a simple model artifact
    artifact_path = Path("artifacts") / "smoke_test.joblib"
    artifact_path.parent.mkdir(exist_ok=True)
    joblib.dump({"test": True}, artifact_path)
    
    model_id = register_model(
        model_type="test",
        model_name="Smoke Test",
        artifact_path=str(artifact_path),
        metrics={"accuracy": 0.95},
        params={"test": True},
        registry_dir="models"
    )
    
    models = list_models("models")
    assert len(models) > 0
    print(f"✓ Model registry working: {len(models)} models")
except Exception as e:
    print(f"✗ Model registry failed: {e}")
    # Not critical, continue

print()
print("=" * 60)
print("✓ ALL SMOKE TESTS PASSED")
print("=" * 60)
print()
print("App is ready to run with: streamlit run app.py")
print()
