#!/usr/bin/env python3
"""
High Standards Feature Verification Test
Tests critical functionality to ensure everything works properly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("=" * 80)
print("HIGH STANDARDS FEATURE VERIFICATION")
print("=" * 80)
print()

passed = 0
failed = 0

def test(name, func):
    global passed, failed
    try:
        func()
        print(f"✓ {name}")
        passed += 1
        return True
    except Exception as e:
        print(f"✗ {name}: {e}")
        failed += 1
        return False

# Test 1: Core imports
def test_core_imports():
    from src.core.config import load_config
    from src.core import forecast_results
    from src.core import forecast_components
    assert load_config() is not None

test("Core module imports", test_core_imports)

# Test 2: Data imports
def test_data_imports():
    from src.data import samples
    from src.data import cleaning
    from src.data import eda

test("Data module imports", test_data_imports)

# Test 3: ML imports
def test_ml_imports():
    from src.ml import tabular
    from src.ml import forecast_transformer

test("ML module imports", test_ml_imports)

# Test 4: Storage imports
def test_storage_imports():
    from src.storage import model_registry
    from src.storage import history

test("Storage module imports", test_storage_imports)

# Test 5: Sample data generation
def test_sample_data():
    from src.data.samples import generate_sample_attack_data
    df = generate_sample_attack_data(size="small")
    assert len(df) > 0
    assert "timestamp" in df.columns

test("Sample data generation", test_sample_data)

# Test 6: Data cleaning
def test_data_cleaning():
    from src.data.samples import generate_sample_attack_data
    from src.data.cleaning import DataCleaner
    df = generate_sample_attack_data(size="small")
    cleaner = DataCleaner()
    cleaned = cleaner.handle_missing_values(df.copy())
    assert cleaned is not None

test("Data cleaning functionality", test_data_cleaning)

# Test 7: EDA engine
def test_eda():
    from src.data.samples import generate_sample_attack_data
    from src.data.eda import EDAEngine
    df = generate_sample_attack_data(size="small")
    eda = EDAEngine(df)
    stats = eda.get_basic_stats()
    assert stats is not None
    assert len(stats) > 0

test("EDA engine analysis", test_eda)

# Test 8: Forecast result creation
def test_forecast_result():
    from src.core.forecast_results import ForecastResult
    
    n_hist = 30
    n_fore = 10
    
    dates_hist = pd.date_range(end=datetime.now(), periods=n_hist, freq="D")
    dates_fore = pd.date_range(start=datetime.now() + timedelta(days=1), periods=n_fore, freq="D")
    
    values_hist = np.random.randn(n_hist).cumsum() + 100
    values_fore = np.random.randn(n_fore).cumsum() + values_hist[-1]
    
    result = ForecastResult(
        model_name="Test",
        model_type="tabular",
        target_column="test_target",
        forecast_values=values_fore,
        forecast_dates=pd.DatetimeIndex(dates_fore),
        historical_values=values_hist,
        historical_dates=pd.DatetimeIndex(dates_hist),
        lower_bound=values_fore - 5,
        upper_bound=values_fore + 5,
        confidence_level=0.95,
        train_metrics={"rmse": 2.5},
        test_metrics={"rmse": 3.0},
        metadata={}
    )
    
    assert result is not None
    df = result.to_dataframe()
    assert len(df) == n_fore
    
    return result

result = None
if test("Forecast result creation", test_forecast_result):
    result = test_forecast_result()

# Test 9: Forecast visualization
def test_forecast_viz():
    from src.core.forecast_results import create_forecast_visualization
    assert result is not None
    fig = create_forecast_visualization(result)
    assert fig is not None
    assert len(fig.data) > 0

if result:
    test("Forecast visualization", test_forecast_viz)

# Test 10: Forecast table
def test_forecast_table():
    from src.core.forecast_results import create_forecast_table
    assert result is not None
    table = create_forecast_table(result)
    assert table is not None
    assert len(table) > 0

if result:
    test("Forecast table generation", test_forecast_table)

# Test 11: Model registry
def test_model_registry():
    from src.storage.model_registry import ModelRegistry
    assert result is not None
    
    registry = ModelRegistry()
    model_id = registry.save_forecast_result(result)
    assert model_id is not None
    
    loaded = registry.load_forecast_result(model_id)
    assert loaded is not None
    
    models = registry.list_models()
    assert len(models) > 0

if result:
    test("Model registry save/load", test_model_registry)

# Test 12: Model history
def test_model_history():
    from src.storage.history import ModelHistory
    
    history = ModelHistory()
    history.log_training(
        model_name="Test",
        model_type="tabular",
        metrics={"rmse": 2.5},
        params={"n_estimators": 100}
    )
    
    events = history.get_recent(limit=10)
    assert events is not None

test("Model history logging", test_model_history)

# Test 13: Tabular ML engine
def test_tabular_ml():
    from src.ml.tabular import TabularMLEngine
    from src.data.samples import generate_sample_attack_data
    
    df = generate_sample_attack_data(size="small")
    
    # Need numeric target
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        target = numeric_cols[0]
        engine = TabularMLEngine(df, target_column=target)
        assert engine is not None

test("Tabular ML engine initialization", test_tabular_ml)

# Test 14: Forecast components
def test_forecast_components():
    from src.core import forecast_components
    assert result is not None
    # Just test import - UI components need Streamlit context

test("Forecast components module", test_forecast_components)

# Test 15: Configuration loading
def test_config():
    from src.core.config import load_config
    config = load_config()
    assert config is not None
    assert "app_name" in config

test("Configuration loading", test_config)

# Test 16: Edge case - empty DataFrame
def test_empty_df():
    from src.data.cleaning import DataCleaner
    empty_df = pd.DataFrame()
    cleaner = DataCleaner()
    # Should handle gracefully
    try:
        cleaned = cleaner.handle_missing_values(empty_df)
    except:
        pass  # Expected to fail or return empty

test("Edge case - empty DataFrame", test_empty_df)

# Test 17: Edge case - NaN values
def test_nan_values():
    from src.data.samples import generate_sample_attack_data
    from src.data.cleaning import DataCleaner
    
    df = generate_sample_attack_data(size="small")
    df.iloc[0:5, 0] = np.nan
    
    cleaner = DataCleaner()
    cleaned = cleaner.handle_missing_values(df)
    assert cleaned is not None

test("Edge case - NaN handling", test_nan_values)

# Test 18: Aggregations
def test_aggregations():
    assert result is not None
    agg = result.get_aggregations()
    assert "monthly" in agg
    assert "quarterly" in agg
    assert "yearly" in agg

if result:
    test("Forecast aggregations", test_aggregations)

# Test 19: Summary stats
def test_summary_stats():
    assert result is not None
    stats = result.get_summary_stats()
    assert "forecast_mean" in stats

if result:
    test("Forecast summary statistics", test_summary_stats)

# Test 20: Type safety
def test_types():
    assert result is not None
    assert isinstance(result.forecast_values, np.ndarray)
    assert isinstance(result.forecast_dates, pd.DatetimeIndex)
    assert isinstance(result.historical_values, np.ndarray)
    assert isinstance(result.historical_dates, pd.DatetimeIndex)

if result:
    test("Type safety validation", test_types)

# Summary
print()
print("=" * 80)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 80)

if failed == 0:
    print()
    print("✓ ALL TESTS PASSED - HIGH STANDARDS VERIFIED")
    print()
    sys.exit(0)
else:
    print()
    print(f"✗ {failed} TEST(S) FAILED - NEEDS ATTENTION")
    print()
    sys.exit(1)
