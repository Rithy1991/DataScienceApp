"""
Universal Forecast Results Module
Standardized post-training results experience for all forecasting models
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class ForecastResult:
    """Universal forecast result container for all model types."""
    
    # Core forecast data
    forecast_values: np.ndarray  # Predicted values
    forecast_dates: pd.DatetimeIndex  # Future dates
    historical_values: np.ndarray  # Past actual values
    historical_dates: pd.DatetimeIndex  # Past dates
    
    # Confidence/uncertainty
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None
    confidence_level: float = 0.95
    
    # Model metadata
    model_type: str = "Unknown"  # e.g., "Transformer", "TFT", "RandomForest"
    model_name: str = "Forecast Model"
    target_column: str = "Value"
    
    # Performance metrics
    train_metrics: Optional[Dict[str, float]] = None
    test_metrics: Optional[Dict[str, float]] = None
    
    # AI summary
    ai_summary: Optional[str] = None
    
    # Optional metadata
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecast to clean DataFrame."""
        df = pd.DataFrame({
            "Date": self.forecast_dates,
            self.target_column: self.forecast_values,
        })
        
        if self.lower_bound is not None:
            df["Lower Bound"] = self.lower_bound
        if self.upper_bound is not None:
            df["Upper Bound"] = self.upper_bound
            
        return df
    
    def get_time_aggregations(self) -> Dict[str, pd.DataFrame]:
        """Get forecast aggregated by month, quarter, year."""
        df = self.to_dataframe()
        df = df.set_index("Date")
        
        aggregations = {}
        
        # Monthly
        agg_dict = {self.target_column: "mean"}
        if "Lower Bound" in df.columns:
            agg_dict["Lower Bound"] = "mean"
        if "Upper Bound" in df.columns:
            agg_dict["Upper Bound"] = "mean"
        monthly = df.resample("M").agg(agg_dict).reset_index()  # type: ignore
        monthly["Period"] = pd.to_datetime(monthly["Date"]).dt.strftime("%b %Y")
        aggregations["monthly"] = monthly
        
        # Quarterly
        agg_dict_q = {self.target_column: "mean"}
        if "Lower Bound" in df.columns:
            agg_dict_q["Lower Bound"] = "mean"
        if "Upper Bound" in df.columns:
            agg_dict_q["Upper Bound"] = "mean"
        quarterly = df.resample("Q").agg(agg_dict_q).reset_index()  # type: ignore
        quarterly["Period"] = pd.to_datetime(quarterly["Date"]).dt.to_period("Q").astype(str)
        aggregations["quarterly"] = quarterly
        
        # Yearly
        agg_dict_y = {self.target_column: "mean"}
        if "Lower Bound" in df.columns:
            agg_dict_y["Lower Bound"] = "mean"
        if "Upper Bound" in df.columns:
            agg_dict_y["Upper Bound"] = "mean"
        yearly = df.resample("Y").agg(agg_dict_y).reset_index()  # type: ignore
        yearly["Period"] = pd.to_datetime(yearly["Date"]).dt.year.astype(str)
        aggregations["yearly"] = yearly
        
        return aggregations
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get human-readable summary statistics."""
        stats = {
            "forecast_horizon": len(self.forecast_values),
            "start_date": self.forecast_dates[0].strftime("%Y-%m-%d"),
            "end_date": self.forecast_dates[-1].strftime("%Y-%m-%d"),
            "mean_forecast": float(np.mean(self.forecast_values)),
            "min_forecast": float(np.min(self.forecast_values)),
            "max_forecast": float(np.max(self.forecast_values)),
            "trend": self._detect_trend(),
        }
        
        if self.lower_bound is not None and self.upper_bound is not None:
            stats["avg_uncertainty"] = float(np.mean(self.upper_bound - self.lower_bound))
            stats["uncertainty_pct"] = float(
                100 * np.mean((self.upper_bound - self.lower_bound) / self.forecast_values)
            )
        
        return stats
    
    def _detect_trend(self) -> str:
        """Detect overall forecast trend."""
        if len(self.forecast_values) < 2:
            return "Stable"
        
        # Linear regression on forecast
        x = np.arange(len(self.forecast_values))
        slope = np.polyfit(x, self.forecast_values, 1)[0]
        
        # Compare to historical volatility
        if len(self.historical_values) > 0:
            hist_std = np.std(self.historical_values)
            threshold = 0.1 * hist_std
        else:
            threshold = 0.01 * np.mean(self.forecast_values)
        
        if abs(slope) < threshold:
            return "Stable"
        elif slope > 0:
            if slope > 2 * threshold:
                return "Strong Uptrend"
            return "Uptrend"
        else:
            if slope < -2 * threshold:
                return "Strong Downtrend"
            return "Downtrend"


def generate_ai_summary(result: ForecastResult) -> str:
    """Generate human-readable AI summary of forecast."""
    stats = result.get_summary_stats()
    
    # Build summary
    summary_parts = []
    
    # Opening: What and when
    horizon = stats["forecast_horizon"]
    if horizon <= 30:
        time_desc = f"next {horizon} days"
    elif horizon <= 90:
        time_desc = f"next {horizon // 7} weeks"
    elif horizon <= 365:
        time_desc = f"next {horizon // 30} months"
    else:
        time_desc = f"next {horizon // 365} years"
    
    summary_parts.append(
        f"📊 **Forecast Overview:** The {result.model_name} predicts {result.target_column} "
        f"for the {time_desc} ({stats['start_date']} to {stats['end_date']})."
    )
    
    # Trend and direction
    trend = stats["trend"]
    mean_val = stats["mean_forecast"]
    
    if "Uptrend" in trend:
        direction = "increase" if trend == "Uptrend" else "significantly increase"
        summary_parts.append(
            f"📈 **Trend:** Values are expected to {direction}, "
            f"with an average of {mean_val:.2f}."
        )
    elif "Downtrend" in trend:
        direction = "decrease" if trend == "Downtrend" else "significantly decrease"
        summary_parts.append(
            f"📉 **Trend:** Values are expected to {direction}, "
            f"with an average of {mean_val:.2f}."
        )
    else:
        summary_parts.append(
            f"➡️ **Trend:** Values are expected to remain relatively stable "
            f"around {mean_val:.2f}."
        )
    
    # Range
    min_val = stats["min_forecast"]
    max_val = stats["max_forecast"]
    summary_parts.append(
        f"📏 **Range:** Predicted values will likely range from {min_val:.2f} "
        f"to {max_val:.2f}."
    )
    
    # Confidence/Uncertainty
    if "uncertainty_pct" in stats:
        unc_pct = stats["uncertainty_pct"]
        if unc_pct < 10:
            conf_desc = "high confidence"
        elif unc_pct < 25:
            conf_desc = "moderate confidence"
        else:
            conf_desc = "lower confidence with wider uncertainty"
        
        summary_parts.append(
            f"🎯 **Confidence:** This forecast shows {conf_desc} "
            f"(±{unc_pct:.1f}% average uncertainty)."
        )
    
    # Model performance context
    if result.test_metrics and "rmse" in result.test_metrics:
        rmse = result.test_metrics["rmse"]
        summary_parts.append(
            f"✅ **Model Performance:** The {result.model_type} model achieved "
            f"an RMSE of {rmse:.2f} on validation data."
        )
    
    return "\n\n".join(summary_parts)


def create_forecast_visualization(
    result: ForecastResult,
    title: Optional[str] = None,
    height: int = 500,
) -> go.Figure:
    """Create standardized interactive forecast visualization."""
    
    if title is None:
        title = f"{result.model_name}: {result.target_column} Forecast"
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=result.historical_dates,
        y=result.historical_values,
        mode="lines+markers",
        name="Historical Data",
        line=dict(color="#0ea5e9", width=2),
        marker=dict(size=4),
        hovertemplate="<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>",
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=result.forecast_dates,
        y=result.forecast_values,
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#f97316", width=2, dash="dash"),
        marker=dict(size=5, symbol="diamond"),
        hovertemplate="<b>%{x}</b><br>Forecast: %{y:.2f}<extra></extra>",
    ))
    
    # Confidence bounds
    if result.lower_bound is not None and result.upper_bound is not None:
        # Upper bound
        fig.add_trace(go.Scatter(
            x=result.forecast_dates,
            y=result.upper_bound,
            mode="lines",
            name=f"{int(result.confidence_level * 100)}% Upper Bound",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        
        # Lower bound with fill
        fig.add_trace(go.Scatter(
            x=result.forecast_dates,
            y=result.lower_bound,
            mode="lines",
            name=f"{int(result.confidence_level * 100)}% Confidence",
            line=dict(width=0),
            fillcolor="rgba(249, 115, 22, 0.2)",
            fill="tonexty",
            hovertemplate="<b>%{x}</b><br>Range: %{y:.2f}<extra></extra>",
        ))
    
    # Forecast start line (best-effort; skip if plotting library errors)
    if len(result.historical_dates) > 0:
        try:
            forecast_start = result.forecast_dates[0]
            fig.add_vline(
                x=forecast_start,
                line_dash="dot",
                line_color="gray",
                annotation_text="Forecast Start",
                annotation_position="top",
            )
        except Exception:
            pass
    
    # Layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=result.target_column,
        hovermode="x unified",
        height=height,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    
    return fig


def create_forecast_table(result: ForecastResult, max_rows: int = 50) -> pd.DataFrame:
    """Create clean forecast table for display."""
    df = result.to_dataframe()
    
    # Format dates nicely
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    
    # Round numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(2)
    
    # Limit rows
    if len(df) > max_rows:
        df = df.head(max_rows)
    
    return df


def create_model_comparison_chart(results: List[ForecastResult]) -> go.Figure:
    """Create comparison chart for multiple forecast models."""
    
    fig = go.Figure()
    
    colors = ["#0ea5e9", "#f97316", "#8b5cf6", "#10b981", "#ef4444"]
    
    for i, result in enumerate(results):
        color = colors[i % len(colors)]
        
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=result.forecast_dates,
            y=result.forecast_values,
            mode="lines+markers",
            name=f"{result.model_name} ({result.model_type})",
            line=dict(color=color, width=2),
            marker=dict(size=4),
            hovertemplate=f"<b>{result.model_name}</b><br>%{{x}}<br>Value: %{{y:.2f}}<extra></extra>",
        ))
    
    # Add historical data from first result (should be same for all)
    if results and len(results[0].historical_dates) > 0:
        fig.add_trace(go.Scatter(
            x=results[0].historical_dates,
            y=results[0].historical_values,
            mode="lines",
            name="Historical Data",
            line=dict(color="gray", width=2, dash="dot"),
            hovertemplate="<b>Historical</b><br>%{x}<br>Value: %{y:.2f}<extra></extra>",
        ))
        
        # Forecast start line
        fig.add_vline(
            x=results[0].forecast_dates[0],
            line_dash="dot",
            line_color="gray",
            annotation_text="Forecast Start",
        )
    
    fig.update_layout(
        title="Model Comparison: Forecast Predictions",
        xaxis_title="Date",
        yaxis_title=results[0].target_column if results else "Value",
        hovermode="x unified",
        height=500,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig


def estimate_confidence_intervals(
    predictions: np.ndarray,
    model_type: str,
    historical_errors: Optional[np.ndarray] = None,
    confidence_level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate confidence intervals based on model type and historical errors.
    
    Returns: (lower_bound, upper_bound)
    """
    
    # Z-score for confidence level
    from scipy import stats as scipy_stats
    z_score = scipy_stats.norm.ppf((1 + confidence_level) / 2)
    
    if historical_errors is not None and len(historical_errors) > 0:
        # Use actual historical errors
        std_error = np.std(historical_errors)
    else:
        # Estimate based on prediction variance and model type
        if model_type.lower() in ["transformer", "tft", "lstm"]:
            # Neural models: ~5-15% uncertainty
            std_error = 0.10 * np.std(predictions)
        elif model_type.lower() in ["randomforest", "xgboost", "lightgbm"]:
            # Tree models: ~10-20% uncertainty
            std_error = 0.15 * np.std(predictions)
        else:
            # Default: ~15% uncertainty
            std_error = 0.15 * np.std(predictions)
    
    # Increasing uncertainty over time (forecasts get less certain further out)
    time_decay = np.linspace(1.0, 1.5, len(predictions))
    std_error_array = std_error * time_decay
    
    lower = predictions - z_score * std_error_array
    upper = predictions + z_score * std_error_array
    
    return lower, upper
