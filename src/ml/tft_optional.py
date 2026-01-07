from __future__ import annotations

import shutil
from dataclasses import dataclass
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TFTTrainResult:
    artifact_path: str
    metrics: Dict[str, float]
    meta: Dict[str, Any]


def tft_available() -> bool:
    """Return True if TFT optional dependencies are importable.

    Supports both Lightning 2.x (``lightning.pytorch``) and legacy ``pytorch_lightning``.
    """
    try:
        import torch  # noqa: F401
    except Exception:
        return False

    try:
        import pytorch_forecasting  # noqa: F401
    except Exception:
        return False

    # Accept either Lightning import path
    try:
        import lightning.pytorch  # noqa: F401
        return True
    except Exception:
        try:
            import pytorch_lightning  # noqa: F401
            return True
        except Exception:
            return False


def explain_tft_requirements() -> str:
    return (
        "To enable Temporal Fusion Transformer (TFT), install the following packages in the same Python environment running Streamlit:\n\n"
        "# Required\n"
        "pip install --upgrade torch pytorch-forecasting\n\n"
        "# Lightning (either one works)\n"
        "pip install --upgrade lightning\n"
        "# or\n"
        "pip install --upgrade pytorch-lightning\n\n"
        "After installing, fully restart Streamlit."
    )


def install_tft_dependencies(prefer: str = "lightning") -> Dict[str, Any]:
    """Attempt to install TFT optional dependencies using pip.

    Parameters
    ----------
    prefer: str
        "lightning" (default) or "pytorch-lightning". If the preferred
        installation fails, we try the alternate automatically.

    Returns
    -------
    Dict[str, Any]
        {"success": bool, "logs": str, "tried": list[str]}
    """
    tried = []
    base = ["torch", "pytorch-forecasting"]
    flavors = [prefer] + (["pytorch-lightning"] if prefer != "pytorch-lightning" else ["lightning"])

    logs_all: list[str] = []
    for flavor in flavors:
        tried.append(flavor)
        pkgs = base + [flavor]
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *pkgs]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            logs_all.append(proc.stdout or "")
            logs_all.append(proc.stderr or "")
            if proc.returncode == 0:
                return {"success": True, "logs": "\n".join(logs_all), "tried": tried}
        except Exception as e:  # pragma: no cover
            logs_all.append(f"Exception while installing {pkgs}: {e}")

    return {"success": False, "logs": "\n".join(logs_all), "tried": tried}


def _prepare_series_frame(df: pd.DataFrame, time_col: str, target_col: str) -> pd.DataFrame:
    d = df[[time_col, target_col]].copy()
    d = d.dropna()
    # Parse time column when possible
    if not pd.api.types.is_datetime64_any_dtype(d[time_col]):
        d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d[d[time_col].notna()]
    d = d.sort_values(time_col)
    d[target_col] = pd.to_numeric(d[target_col], errors="coerce")
    d = d[d[target_col].notna()]
    d = d.reset_index(drop=True)
    d["series_id"] = "series_0"
    d["time_idx"] = np.arange(len(d), dtype=np.int64)
    return d


def _infer_time_delta(d: pd.DataFrame, time_col: str) -> Optional[pd.Timedelta]:
    if not pd.api.types.is_datetime64_any_dtype(d[time_col]):
        return None
    diffs = d[time_col].diff().dropna()
    if diffs.empty:
        return None
    delta = diffs.median()
    if not isinstance(delta, pd.Timedelta):
        return None
    if pd.isna(delta) or delta <= pd.Timedelta(0):
        return None
    return delta


def train_tft_forecaster(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    horizon: int = 12,
    lookback: int = 48,
    epochs: int = 20,
    lr: float = 1e-3,
    artifact_path: str = "artifacts/forecast_tft.ckpt",
    seed: int = 42,
) -> TFTTrainResult:
    if not tft_available():
        raise RuntimeError(
            "Temporal Fusion Transformer training requires optional dependencies: "
            "torch, pytorch-lightning, pytorch-forecasting."
        )

    import torch
    # Import Lightning with fallback to legacy pytorch_lightning
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import ModelCheckpoint
    except Exception:  # pragma: no cover - legacy fallback
        import pytorch_lightning as pl  # type: ignore
        from pytorch_lightning.callbacks import ModelCheckpoint  # type: ignore
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss

    pl.seed_everything(seed, workers=True)

    d = _prepare_series_frame(df, time_col=time_col, target_col=target_col)
    if len(d) < (lookback + horizon + 10):
        raise ValueError("Not enough data for selected lookback/horizon.")

    max_time_idx = int(d["time_idx"].max())
    training_cutoff = max_time_idx - int(horizon)
    if training_cutoff <= int(lookback):
        raise ValueError("Time series too short for the selected parameters.")

    training = TimeSeriesDataSet(
        d[d.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=target_col,
        group_ids=["series_id"],
        max_encoder_length=int(lookback),
        max_prediction_length=int(horizon),
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[target_col],
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, d, predict=True, stop_randomization=True)
    train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=float(lr),
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=-1,
        reduce_on_plateau_patience=3,
    )

    ckpt = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    trainer = pl.Trainer(
        max_epochs=int(epochs),
        accelerator="auto",
        devices="auto",
        logger=False,
        enable_model_summary=False,
        callbacks=[ckpt],
        enable_progress_bar=False,
    )

    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = ckpt.best_model_path
    if not best_path:
        # fallback to last checkpoint
        best_path = artifact_path
        trainer.save_checkpoint(best_path)

    # Copy best checkpoint to stable artifact path
    shutil.copy2(best_path, artifact_path)

    # Evaluate: median quantile as point prediction
    preds = tft.predict(val_loader).detach().cpu().numpy()
    # Collect actuals
    actuals = []
    for _, y in val_loader:
        # y can be tuple (target, weight)
        y_t = y[0] if isinstance(y, (tuple, list)) else y
        actuals.append(y_t.detach().cpu().numpy())
    y_true = np.concatenate(actuals, axis=0)

    # Align shapes (N, horizon)
    n = min(len(preds), len(y_true))
    preds = preds[:n]
    y_true = y_true[:n]

    mae = float(np.mean(np.abs(y_true - preds)))
    rmse = float(np.sqrt(np.mean((y_true - preds) ** 2)))

    delta = _infer_time_delta(d, time_col=time_col)
    meta: Dict[str, Any] = {
        "type": "tft",
        "time_col": time_col,
        "target_col": target_col,
        "lookback": int(lookback),
        "horizon": int(horizon),
        "epochs": int(epochs),
        "lr": float(lr),
        "series_id": "series_0",
        "time_delta_seconds": int(delta.total_seconds()) if delta is not None else None,
        "metrics": {"mae": mae, "rmse": rmse},
    }

    return TFTTrainResult(artifact_path=artifact_path, metrics={"mae": mae, "rmse": rmse}, meta=meta)


def load_tft_forecaster(path: str):
    if not tft_available():
        raise RuntimeError("TFT requires optional dependencies.")

    from pytorch_forecasting import TemporalFusionTransformer

    model = TemporalFusionTransformer.load_from_checkpoint(path)
    model.eval()
    return model


def forecast_tft(
    model,
    history_df: pd.DataFrame,
    time_col: str,
    target_col: str,
    horizon: int,
    lookback: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    if not tft_available():
        raise RuntimeError("TFT requires optional dependencies.")

    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import GroupNormalizer

    d = _prepare_series_frame(history_df, time_col=time_col, target_col=target_col)
    if len(d) < (lookback + 2):
        raise ValueError("Not enough history for TFT forecast.")

    last_time = d[time_col].iloc[-1]
    delta = _infer_time_delta(d, time_col=time_col)

    max_time_idx = int(d["time_idx"].max())
    future_idx = np.arange(max_time_idx + 1, max_time_idx + 1 + int(horizon), dtype=np.int64)
    future = pd.DataFrame({"series_id": "series_0", "time_idx": future_idx})
    if delta is not None:
        future[time_col] = [last_time + (i + 1) * delta for i in range(int(horizon))]
    else:
        future[time_col] = pd.NaT
    future[target_col] = np.nan

    full = pd.concat([d, future], ignore_index=True)
    # Fill NaN values in future rows (forecasting targets) to avoid TFT validation errors
    full[target_col] = full[target_col].fillna(full[target_col].median())

    ds = TimeSeriesDataSet(
        full,
        time_idx="time_idx",
        target=target_col,
        group_ids=["series_id"],
        max_encoder_length=int(lookback),
        max_prediction_length=int(horizon),
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[target_col],
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        predict_mode=True,
    )

    loader = ds.to_dataloader(train=False, batch_size=1, num_workers=0)
    pred = model.predict(loader).detach().cpu().numpy().reshape(-1)
    forecast_df = pd.DataFrame({"step": list(range(1, len(pred) + 1)), "forecast": pred})
    if delta is not None:
        forecast_df["forecast_time"] = [last_time + (i + 1) * delta for i in range(len(pred))]

    return forecast_df, pred
