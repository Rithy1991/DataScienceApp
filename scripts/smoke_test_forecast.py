from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.forecast_results import (
    ForecastResult,
    estimate_confidence_intervals,
    generate_ai_summary,
    create_forecast_visualization,
)


def main() -> None:
    # Synthetic historical series (60 daily points)
    dates_hist = pd.date_range(start="2025-10-01", periods=60, freq="D")
    values_hist = 100 + 10 * np.sin(np.linspace(0, 6, 60)) + np.random.normal(0, 2, 60)

    # Synthetic forecast (12 future steps)
    horizon = 12
    base = float(values_hist[-1])
    values_fore = base + 10 * np.sin(np.linspace(0.2, 1.5, horizon)) + np.random.normal(0, 2, horizon)
    dates_fore = pd.date_range(start=dates_hist[-1] + pd.Timedelta(days=1), periods=horizon, freq="D").tolist()

    # Confidence intervals
    lower, upper = estimate_confidence_intervals(np.asarray(values_fore), model_type="transformer")

    # Build ForecastResult
    fr = ForecastResult(
        model_name="SmokeTest Transformer",
        model_type="transformer",
        target_column="synthetic_target",
        forecast_values=np.asarray(values_fore),
        forecast_dates=pd.DatetimeIndex(dates_fore),
        historical_values=np.asarray(values_hist),
        historical_dates=pd.DatetimeIndex(dates_hist),
        lower_bound=lower,
        upper_bound=upper,
        confidence_level=0.95,
        train_metrics=None,
        test_metrics={"rmse": float(np.sqrt(np.mean((values_fore - np.mean(values_fore)) ** 2)))},
        metadata={"note": "synthetic"},
    )

    # AI summary
    summary = generate_ai_summary(fr)
    print("\n===== AI Summary =====\n")
    print(summary)

    # Visualization
    fig = create_forecast_visualization(fr, title="Smoke Test Forecast")

    # Save to artifacts
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    html_path = artifacts_dir / "smoke_forecast.html"
    try:
        fig.write_html(str(html_path))
        print(f"\nSaved forecast visualization to: {html_path}")
    except Exception as e:
        print(f"Failed to save HTML plot: {e}")


if __name__ == "__main__":
    main()
