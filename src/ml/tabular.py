from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


TaskType = Literal["regression", "classification"]


@dataclass
class TrainResult:
    model: Any
    task: TaskType
    metrics: Dict[str, float]
    artifact_path: str
    meta: Dict[str, Any]


def _datetime_to_features(values) -> np.ndarray:
    """Convert datetime columns into numeric feature arrays.

    Output features per datetime column:
    - year, month, day, dayofweek, hour, unix_s
    """
    if isinstance(values, pd.DataFrame):
        frame = values
    else:
        frame = pd.DataFrame(values)

    feature_cols = []
    for col in frame.columns:
        s = pd.to_datetime(frame[col], errors="coerce")

        year = s.dt.year.astype("float64")
        month = s.dt.month.astype("float64")
        day = s.dt.day.astype("float64")
        dayofweek = s.dt.dayofweek.astype("float64")
        hour = s.dt.hour.astype("float64")

        ns = s.astype("int64").astype("float64")
        ns[s.isna()] = np.nan
        unix_s = ns / 1_000_000_000

        feature_cols.extend(
            [
                year.to_numpy(),
                month.to_numpy(),
                day.to_numpy(),
                dayofweek.to_numpy(),
                hour.to_numpy(),
                unix_s.to_numpy(),
            ]
        )

    if not feature_cols:
        return np.empty((len(frame), 0), dtype="float64")
    return np.column_stack(feature_cols).astype("float64", copy=False)


def _detect_task(y: pd.Series) -> TaskType:
    if pd.api.types.is_numeric_dtype(y):
        unique = y.nunique(dropna=True)
        if unique <= 15 and y.dtype != float:
            return "classification"
        return "regression"
    return "classification"


def _build_model(name: str, task: TaskType):
    name = name.lower().strip()

    if name == "xgboost":
        try:
            import xgboost as xgb  # type: ignore[import-not-found]

            if task == "regression":
                return xgb.XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                )
            return xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                eval_metric="logloss",
            )
        except Exception:
            pass

    if name == "lightgbm":
        try:
            import lightgbm as lgb  # type: ignore[import-not-found]

            if task == "regression":
                return lgb.LGBMRegressor(
                    n_estimators=800,
                    learning_rate=0.05,
                    num_leaves=63,
                    random_state=42,
                )
            return lgb.LGBMClassifier(
                n_estimators=800,
                learning_rate=0.05,
                num_leaves=63,
                random_state=42,
            )
        except Exception:
            pass

    if name == "randomforest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        if task == "regression":
            return RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1)
        return RandomForestClassifier(n_estimators=600, random_state=42, n_jobs=-1)

    if name == "gradientboosting":
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        if task == "regression":
            return GradientBoostingRegressor(random_state=42)
        return GradientBoostingClassifier(random_state=42)

    raise ValueError(f"Unknown/unsupported model: {name}")


def train_tabular(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
    explicit_task: Optional[TaskType] = None,
    artifact_path: str = "artifacts/tabular_model.joblib",
) -> TrainResult:
    if target_col not in df.columns:
        raise ValueError("Target column not in dataframe")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    task: TaskType = explicit_task or _detect_task(y)

    datetime_features = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]
    numeric_features = [c for c in X.select_dtypes(include=["number"]).columns.tolist() if c not in datetime_features]
    categorical_features = [c for c in X.columns if c not in numeric_features and c not in datetime_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    datetime_transformer = Pipeline(
        steps=[
            ("featurize", FunctionTransformer(_datetime_to_features)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if datetime_features:
        transformers.append(("dt", datetime_transformer, datetime_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    model = _build_model(model_name, task)

    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # Determine if we can use stratified splitting
    stratify_arg = None
    if task == "classification":
        # Check if all classes have at least 2 samples for stratification
        class_counts = y.value_counts()
        min_samples = class_counts.min()
        if min_samples >= 2:
            stratify_arg = y
        # If stratification would fail, we'll use random split and show a warning later

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_seconds = time.time() - t0

    y_pred = clf.predict(X_test)

    # Extract Feature Importance
    feature_importance = {}
    try:
        # Get feature names from preprocessor
        if hasattr(preprocessor, "get_feature_names_out"):
             feature_names = preprocessor.get_feature_names_out()
        else:
             feature_names = [f"feat_{i}" for i in range(preprocessor.transform(X[:1]).shape[1])]
        
        # Get importances from model (if supported)
        est = clf.named_steps['model']
        if hasattr(est, "feature_importances_"):
            importances = est.feature_importances_
        elif hasattr(est, "coef_"):
            importances = np.abs(est.coef_[0]) if est.coef_.ndim > 1 else np.abs(est.coef_)
        else:
            importances = []
            
        if len(importances) == len(feature_names):
            feature_importance = dict(zip(feature_names, [float(x) for x in importances]))
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
            # Limit to top 20 to avoid bloating metadata
            feature_importance = dict(list(feature_importance.items())[:20])
            
    except Exception as e:
        print(f"Feature importance extraction failed: {e}")

    metrics: Dict[str, float] = {}
    if task == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics = {
            "rmse": rmse,
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }
    else:
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        }

    dump(clf, artifact_path)

    meta: Dict[str, Any] = {
        "model_name": model_name,
        "task": task,
        "target_col": target_col,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "train_seconds": float(train_seconds),
        "metrics": metrics,
        "stratified": stratify_arg is not None,
        "feature_importance": feature_importance
    }

    return TrainResult(model=clf, task=task, metrics=metrics, artifact_path=artifact_path, meta=meta)


def predict_tabular(model, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Make predictions on a dataframe using a trained model.
    
    Args:
        model: Trained sklearn model or pipeline
        df: Input dataframe for prediction
        
    Returns:
        Tuple of (predictions, probabilities or None)
    """
    try:
        # Handle empty dataframe
        if df.empty:
            raise ValueError("Input dataframe is empty")
        
        # Convert all columns to numeric where possible for safety
        df_pred = df.copy()
        for col in df_pred.columns:
            if pd.api.types.is_object_dtype(df_pred[col]):
                # Try to convert string inputs to appropriate types
                try:
                    df_pred[col] = pd.to_numeric(df_pred[col], errors='coerce')
                except:
                    pass  # Keep as object if conversion fails
        
        preds = model.predict(df_pred)
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(df_pred)
            except Exception:
                proba = None
        return preds, proba
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")
