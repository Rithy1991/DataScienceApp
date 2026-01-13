"""Production Readiness and Deployment Simulators."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ProductionCheck:
    """Single production readiness check result."""
    check_name: str
    status: str  # 'pass', 'warning', 'fail'
    score: Optional[float]
    message: str
    recommendation: str


@dataclass
class ProductionReport:
    """Complete production readiness report."""
    overall_status: str
    readiness_score: float
    checks: List[ProductionCheck]
    critical_issues: List[str]
    warnings: List[str]
    passed_checks: int
    total_checks: int


class ProductionReadinessSimulator:
    """Simulate production readiness checks for ML models."""
    
    @staticmethod
    def check_model_performance(
        model: any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        min_accuracy: float = 0.7,
        min_precision: float = 0.7,
        min_recall: float = 0.7
    ) -> ProductionCheck:
        """Check if model meets minimum performance thresholds."""
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        passed = accuracy >= min_accuracy and precision >= min_precision and recall >= min_recall
        
        if passed:
            status = 'pass'
            message = f"✅ Model performance meets thresholds (Acc: {accuracy:.3f}, Prec: {precision:.3f}, Rec: {recall:.3f})"
            recommendation = "Model performance is acceptable for production."
        else:
            status = 'fail'
            message = f"❌ Model below thresholds (Acc: {accuracy:.3f}, Prec: {precision:.3f}, Rec: {recall:.3f})"
            recommendation = "Improve model before deployment. Consider: more data, feature engineering, hyperparameter tuning."
        
        return ProductionCheck(
            check_name="Performance Thresholds",
            status=status,
            score=accuracy,
            message=message,
            recommendation=recommendation
        )
    
    @staticmethod
    def check_model_stability(
        model: any,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
        max_std: float = 0.05
    ) -> ProductionCheck:
        """Check model stability across cross-validation folds."""
        
        try:
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
            mean_score = scores.mean()
            std_score = scores.std()
            
            passed = std_score <= max_std
            
            if passed:
                status = 'pass'
                message = f"✅ Model is stable across folds (mean: {mean_score:.3f}, std: {std_score:.3f})"
                recommendation = "Model shows consistent performance."
            else:
                status = 'warning'
                message = f"⚠️ Model shows variability (mean: {mean_score:.3f}, std: {std_score:.3f})"
                recommendation = "Consider ensemble methods or more robust training."
            
            return ProductionCheck(
                check_name="Model Stability",
                status=status,
                score=std_score,
                message=message,
                recommendation=recommendation
            )
        except Exception as e:
            return ProductionCheck(
                check_name="Model Stability",
                status='fail',
                score=None,
                message=f"❌ Failed to assess stability: {e}",
                recommendation="Check model compatibility with cross-validation."
            )
    
    @staticmethod
    def check_data_quality(
        X: np.ndarray,
        max_missing_rate: float = 0.1,
        max_outlier_rate: float = 0.05
    ) -> ProductionCheck:
        """Check production data quality."""
        
        # Missing values
        if hasattr(X, 'isnull'):
            missing_rate = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
        else:
            missing_rate = np.isnan(X).sum() / X.size if X.size > 0 else 0
        
        # Outliers (using IQR method)
        outlier_count = 0
        if X.ndim == 2:
            for col_idx in range(X.shape[1]):
                col_data = X[:, col_idx]
                q1 = np.percentile(col_data[~np.isnan(col_data)], 25)
                q3 = np.percentile(col_data[~np.isnan(col_data)], 75)
                iqr = q3 - q1
                outlier_count += ((col_data < (q1 - 3 * iqr)) | (col_data > (q3 + 3 * iqr))).sum()
        
        outlier_rate = outlier_count / X.size if X.size > 0 else 0
        
        passed = missing_rate <= max_missing_rate and outlier_rate <= max_outlier_rate
        
        if passed:
            status = 'pass'
            message = f"✅ Data quality acceptable (Missing: {missing_rate:.2%}, Outliers: {outlier_rate:.2%})"
            recommendation = "Data quality meets production standards."
        elif missing_rate > max_missing_rate:
            status = 'fail'
            message = f"❌ Too many missing values ({missing_rate:.2%})"
            recommendation = "Implement robust missing value handling in production pipeline."
        else:
            status = 'warning'
            message = f"⚠️ High outlier rate ({outlier_rate:.2%})"
            recommendation = "Consider outlier detection and handling in production."
        
        return ProductionCheck(
            check_name="Data Quality",
            status=status,
            score=1 - max(missing_rate, outlier_rate),
            message=message,
            recommendation=recommendation
        )
    
    @staticmethod
    def check_prediction_latency(
        model: any,
        X_sample: np.ndarray,
        max_latency_ms: float = 100,
        n_iterations: int = 100
    ) -> ProductionCheck:
        """Check prediction latency."""
        
        import time
        
        latencies = []
        for _ in range(n_iterations):
            start = time.time()
            _ = model.predict(X_sample[:100] if len(X_sample) > 100 else X_sample)
            latencies.append((time.time() - start) * 1000)  # Convert to ms
        
        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        passed = p95_latency <= max_latency_ms
        
        if passed:
            status = 'pass'
            message = f"✅ Latency acceptable (mean: {mean_latency:.1f}ms, p95: {p95_latency:.1f}ms)"
            recommendation = "Prediction speed meets requirements."
        else:
            status = 'warning'
            message = f"⚠️ High latency (mean: {mean_latency:.1f}ms, p95: {p95_latency:.1f}ms)"
            recommendation = "Consider model optimization, quantization, or caching strategies."
        
        return ProductionCheck(
            check_name="Prediction Latency",
            status=status,
            score=max_latency_ms / max(p95_latency, 1),
            message=message,
            recommendation=recommendation
        )
    
    @staticmethod
    def check_model_size(
        model: any,
        max_size_mb: float = 100
    ) -> ProductionCheck:
        """Check model size for deployment."""
        
        import pickle
        import sys
        
        try:
            # Serialize model to check size
            model_bytes = pickle.dumps(model)
            size_mb = sys.getsizeof(model_bytes) / (1024 * 1024)
            
            passed = size_mb <= max_size_mb
            
            if passed:
                status = 'pass'
                message = f"✅ Model size acceptable ({size_mb:.2f} MB)"
                recommendation = "Model size is suitable for deployment."
            else:
                status = 'warning'
                message = f"⚠️ Large model size ({size_mb:.2f} MB)"
                recommendation = "Consider model compression, pruning, or knowledge distillation."
            
            return ProductionCheck(
                check_name="Model Size",
                status=status,
                score=max_size_mb / max(size_mb, 1),
                message=message,
                recommendation=recommendation
            )
        except Exception as e:
            return ProductionCheck(
                check_name="Model Size",
                status='warning',
                score=None,
                message=f"⚠️ Could not determine model size: {e}",
                recommendation="Ensure model is serializable for deployment."
            )
    
    @staticmethod
    def check_feature_availability(
        required_features: List[str],
        available_features: List[str]
    ) -> ProductionCheck:
        """Check if all required features are available in production."""
        
        missing_features = set(required_features) - set(available_features)
        
        if not missing_features:
            status = 'pass'
            message = f"✅ All {len(required_features)} features available"
            recommendation = "Feature pipeline is complete."
        else:
            status = 'fail'
            message = f"❌ Missing {len(missing_features)} features: {list(missing_features)[:5]}"
            recommendation = "Ensure all features can be computed in production environment."
        
        return ProductionCheck(
            check_name="Feature Availability",
            status=status,
            score=1 - len(missing_features) / len(required_features) if required_features else 1,
            message=message,
            recommendation=recommendation
        )
    
    @staticmethod
    def check_fairness_bias(
        model: any,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_feature_idx: Optional[int] = None,
        max_disparity: float = 0.1
    ) -> ProductionCheck:
        """Check model fairness across sensitive groups."""
        
        if sensitive_feature_idx is None:
            return ProductionCheck(
                check_name="Fairness & Bias",
                status='warning',
                score=None,
                message="⚠️ No sensitive feature specified",
                recommendation="Specify sensitive features (e.g., gender, race) for fairness testing."
            )
        
        try:
            sensitive_feature = X[:, sensitive_feature_idx]
            unique_groups = np.unique(sensitive_feature)
            
            if len(unique_groups) < 2:
                return ProductionCheck(
                    check_name="Fairness & Bias",
                    status='warning',
                    score=None,
                    message="⚠️ Insufficient groups for fairness analysis",
                    recommendation="Ensure sensitive feature has multiple groups."
                )
            
            # Calculate accuracy for each group
            group_accuracies = []
            for group in unique_groups:
                mask = sensitive_feature == group
                if mask.sum() > 0:
                    y_pred = model.predict(X[mask])
                    acc = accuracy_score(y[mask], y_pred)
                    group_accuracies.append(acc)
            
            if len(group_accuracies) >= 2:
                max_acc = max(group_accuracies)
                min_acc = min(group_accuracies)
                disparity = max_acc - min_acc
                
                passed = disparity <= max_disparity
                
                if passed:
                    status = 'pass'
                    message = f"✅ Fairness acceptable (disparity: {disparity:.3f})"
                    recommendation = "Model shows fairness across groups."
                else:
                    status = 'fail'
                    message = f"❌ Fairness concern (disparity: {disparity:.3f})"
                    recommendation = "Investigate bias. Consider fairness constraints or reweighting."
                
                return ProductionCheck(
                    check_name="Fairness & Bias",
                    status=status,
                    score=1 - disparity,
                    message=message,
                    recommendation=recommendation
                )
        except Exception as e:
            pass
        
        return ProductionCheck(
            check_name="Fairness & Bias",
            status='warning',
            score=None,
            message="⚠️ Could not assess fairness",
            recommendation="Manual fairness review recommended."
        )
    
    @staticmethod
    def check_monitoring_readiness() -> ProductionCheck:
        """Check if monitoring infrastructure is ready."""
        
        # In a real system, this would check actual monitoring setup
        # For simulation, we'll provide guidance
        
        return ProductionCheck(
            check_name="Monitoring Readiness",
            status='warning',
            score=None,
            message="⚠️ Monitoring setup required",
            recommendation="Implement: prediction logging, performance tracking, data drift detection, alert systems."
        )
    
    @staticmethod
    def generate_production_report(
        model: any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        required_features: Optional[List[str]] = None,
        available_features: Optional[List[str]] = None
    ) -> ProductionReport:
        """Generate comprehensive production readiness report."""
        
        checks = []
        
        # Run all checks
        checks.append(ProductionReadinessSimulator.check_model_performance(model, X_test, y_test))
        checks.append(ProductionReadinessSimulator.check_model_stability(model, X_train, y_train))
        checks.append(ProductionReadinessSimulator.check_data_quality(X_test))
        checks.append(ProductionReadinessSimulator.check_prediction_latency(model, X_test))
        checks.append(ProductionReadinessSimulator.check_model_size(model))
        
        if required_features and available_features:
            checks.append(ProductionReadinessSimulator.check_feature_availability(required_features, available_features))
        
        checks.append(ProductionReadinessSimulator.check_fairness_bias(model, X_test, y_test))
        checks.append(ProductionReadinessSimulator.check_monitoring_readiness())
        
        # Calculate overall status
        passed_checks = sum(1 for c in checks if c.status == 'pass')
        warning_checks = sum(1 for c in checks if c.status == 'warning')
        failed_checks = sum(1 for c in checks if c.status == 'fail')
        
        total_checks = len(checks)
        readiness_score = (passed_checks + 0.5 * warning_checks) / total_checks
        
        if failed_checks > 0:
            overall_status = 'not_ready'
        elif warning_checks > total_checks / 2:
            overall_status = 'review_needed'
        else:
            overall_status = 'ready'
        
        # Collect issues
        critical_issues = [c.message for c in checks if c.status == 'fail']
        warnings_list = [c.message for c in checks if c.status == 'warning']
        
        return ProductionReport(
            overall_status=overall_status,
            readiness_score=readiness_score,
            checks=checks,
            critical_issues=critical_issues,
            warnings=warnings_list,
            passed_checks=passed_checks,
            total_checks=total_checks
        )


class DataDriftSimulator:
    """Simulate data drift scenarios."""
    
    @staticmethod
    def simulate_covariate_shift(
        X_train: np.ndarray,
        shift_magnitude: float = 0.5
    ) -> np.ndarray:
        """Simulate covariate shift (input distribution changes)."""
        
        # Shift mean of features
        X_shifted = X_train + shift_magnitude * np.std(X_train, axis=0)
        return X_shifted
    
    @staticmethod
    def simulate_concept_drift(
        X: np.ndarray,
        y: np.ndarray,
        flip_rate: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate concept drift (relationship between X and y changes)."""
        
        # Randomly flip labels
        n_flips = int(len(y) * flip_rate)
        flip_indices = np.random.choice(len(y), n_flips, replace=False)
        
        y_drifted = y.copy()
        unique_classes = np.unique(y)
        
        for idx in flip_indices:
            current_class = y_drifted[idx]
            other_classes = [c for c in unique_classes if c != current_class]
            y_drifted[idx] = np.random.choice(other_classes)
        
        return X, y_drifted
    
    @staticmethod
    def detect_drift(
        X_reference: np.ndarray,
        X_current: np.ndarray,
        method: str = 'ks_test'
    ) -> Dict:
        """Detect drift using statistical tests."""
        
        from scipy.stats import ks_2samp
        
        results = {
            'drift_detected': False,
            'drift_score': 0.0,
            'drifted_features': []
        }
        
        if method == 'ks_test':
            p_values = []
            for col_idx in range(X_reference.shape[1]):
                statistic, p_value = ks_2samp(X_reference[:, col_idx], X_current[:, col_idx])
                p_values.append(p_value)
                
                if p_value < 0.05:
                    results['drifted_features'].append(col_idx)
            
            results['drift_score'] = 1 - np.mean(p_values)
            results['drift_detected'] = len(results['drifted_features']) > 0
        
        return results
