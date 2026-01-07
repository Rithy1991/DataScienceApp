from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ForecastTrainResult:
    model: Any
    artifact_path: str
    metrics: Dict[str, float]
    meta: Dict[str, Any]


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _make_supervised(series: np.ndarray, lookback: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lookback, len(series) - horizon + 1):
        X.append(series[i - lookback : i])
        y.append(series[i : i + horizon])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def train_simple_transformer_forecaster(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    horizon: int = 12,
    lookback: int = 48,
    epochs: int = 15,
    lr: float = 1e-3,
    artifact_path: str = "artifacts/forecast_transformer.pt",
) -> ForecastTrainResult:
    if not _torch_available():
        raise RuntimeError("PyTorch is not installed. Install 'torch' to train Transformer forecaster.")

    import torch
    import torch.nn as nn

    d = df[[time_col, target_col]].dropna().sort_values(time_col)
    series = pd.to_numeric(d[target_col], errors="coerce").dropna().to_numpy(dtype=np.float32)
    if len(series) < (lookback + horizon + 5):
        raise ValueError("Not enough data for the selected lookback/horizon.")

    X, y = _make_supervised(series, lookback=lookback, horizon=horizon)
    n = X.shape[0]
    split = max(1, int(n * 0.8))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # normalize
    mu = float(np.mean(X_train))
    sigma = float(np.std(X_train) + 1e-8)
    X_train_n = (X_train - mu) / sigma
    X_val_n = (X_val - mu) / sigma

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 4096):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class TransformerForecaster(nn.Module):
        def __init__(self, lookback: int, horizon: int, d_model: int = 64, nhead: int = 4, layers: int = 2):
            super().__init__()
            self.lookback = lookback
            self.horizon = horizon
            self.input_proj = nn.Linear(1, d_model)
            self.pos = PositionalEncoding(d_model)
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, horizon))

        def forward(self, x):
            # x: (B, L)
            x = x.unsqueeze(-1)  # (B, L, 1)
            x = self.input_proj(x)
            x = self.pos(x)
            h = self.encoder(x)
            h_last = h[:, -1, :]
            return self.head(h_last)

    model = TransformerForecaster(lookback=lookback, horizon=horizon).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    def batch_iter(Xb, yb, batch_size: int = 64):
        idx = np.arange(len(Xb))
        np.random.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            j = idx[i : i + batch_size]
            yield Xb[j], yb[j]

    t0 = time.time()
    best_val = float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        for xb, yb in batch_iter(X_train_n, y_train):
            xb_t = torch.from_numpy(xb).to(device)
            yb_t = torch.from_numpy(yb).to(device)
            pred = model(xb_t)
            loss = loss_fn(pred, yb_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(X_val_n).to(device)
            yv = torch.from_numpy(y_val).to(device)
            pv = model(xv)
            val_loss = float(loss_fn(pv, yv).item())
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    train_seconds = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    # compute metrics on val
    model.eval()
    with torch.no_grad():
        xv = torch.from_numpy(X_val_n).to(device)
        pv = model(xv).cpu().numpy()

    # de-normalize not needed because y is in original scale
    y_true = y_val
    y_pred = pv

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    import torch

    torch.save(
        {
            "state_dict": model.state_dict(),
            "lookback": lookback,
            "horizon": horizon,
            "mu": mu,
            "sigma": sigma,
        },
        artifact_path,
    )

    meta: Dict[str, Any] = {
        "type": "transformer_forecaster",
        "time_col": time_col,
        "target_col": target_col,
        "lookback": lookback,
        "horizon": horizon,
        "epochs": epochs,
        "lr": lr,
        "train_seconds": float(train_seconds),
        "metrics": {"mae": mae, "rmse": rmse},
    }

    return ForecastTrainResult(model=model, artifact_path=artifact_path, metrics={"mae": mae, "rmse": rmse}, meta=meta)


def load_transformer_forecaster(path: str):
    if not _torch_available():
        raise RuntimeError("PyTorch is not installed. Install 'torch' to load Transformer forecaster.")

    import torch
    import torch.nn as nn

    ckpt = torch.load(path, map_location="cpu")
    lookback = int(ckpt["lookback"])
    horizon = int(ckpt["horizon"])

    # minimal re-definition matching train
    import math

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 4096):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class TransformerForecaster(nn.Module):
        def __init__(self, lookback: int, horizon: int, d_model: int = 64, nhead: int = 4, layers: int = 2):
            super().__init__()
            self.input_proj = nn.Linear(1, d_model)
            self.pos = PositionalEncoding(d_model)
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, horizon))

        def forward(self, x):
            x = x.unsqueeze(-1)
            x = self.input_proj(x)
            x = self.pos(x)
            h = self.encoder(x)
            h_last = h[:, -1, :]
            return self.head(h_last)

    model = TransformerForecaster(lookback=lookback, horizon=horizon)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def forecast_transformer(
    model,
    ckpt: Dict[str, Any],
    history: np.ndarray,
) -> np.ndarray:
    if not _torch_available():
        raise RuntimeError("PyTorch is not installed.")

    import torch

    lookback = int(ckpt["lookback"])
    mu = float(ckpt["mu"])
    sigma = float(ckpt["sigma"])

    if len(history) < lookback:
        raise ValueError("History shorter than lookback")

    x = history[-lookback:].astype(np.float32)
    x = (x - mu) / sigma

    with torch.no_grad():
        pred = model(torch.from_numpy(x).unsqueeze(0)).cpu().numpy().reshape(-1)
    return pred
