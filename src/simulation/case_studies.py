"""Real-World Case Studies: Practical ML Scenarios for Learning."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from src.simulation.engine import SimulationParameters, DataGenerator


class CaseStudy:
    """Base class for case study simulations."""
    
    def __init__(self, name: str, description: str, problem_type: str):
        self.name = name
        self.description = description
        self.problem_type = problem_type
        self.generator = DataGenerator()
    
    def get_recommended_parameters(self) -> SimulationParameters:
        """Get parameters recommended for this case study."""
        raise NotImplementedError
    
    def generate_data(self, params: SimulationParameters) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data for this case study."""
        raise NotImplementedError


class SpamDetectionCase(CaseStudy):
    """Email spam detection case study."""
    
    def __init__(self):
        super().__init__(
            name="Email Spam Detection",
            description="Build a classifier to identify spam vs legitimate emails. Features represent word frequencies and email characteristics.",
            problem_type="classification"
        )
    
    def get_recommended_parameters(self) -> SimulationParameters:
        """Parameters tuned for spam detection."""
        return SimulationParameters(
            n_samples=5000,
            n_features=50,
            n_informative=40,
            n_redundant=10,
            n_classes=2,
            noise_level=0.15,
            missing_rate=0.02,
            class_imbalance=[0.85, 0.15],  # Spam is rarer
            random_state=42
        )
    
    def generate_data(self, params: SimulationParameters) -> Tuple[np.ndarray, np.ndarray]:
        """Generate email spam data."""
        params.n_classes = 2
        return self.generator.generate_classification(params, dataset_type='standard')
    
    def get_insights(self) -> Dict[str, str]:
        """Get case-study specific insights."""
        return {
            'challenge': "Class imbalance: legitimate emails (85%) greatly outnumber spam (15%)",
            'metrics_to_watch': "Precision and Recall - False positives (blocking legitimate email) can be worse than false negatives",
            'real_world_impact': "Each false positive frustrates a user; high recall ensures we catch most spam",
            'data_challenges': "Spammers constantly evolve tactics, so models need regular retraining",
        }


class StockPriceMovementCase(CaseStudy):
    """Stock price movement prediction case study."""
    
    def __init__(self):
        super().__init__(
            name="Stock Price Movement Prediction",
            description="Predict whether stock price will go up or down. Features represent technical indicators and market factors.",
            problem_type="classification"
        )
    
    def get_recommended_parameters(self) -> SimulationParameters:
        """Parameters for stock prediction."""
        return SimulationParameters(
            n_samples=2000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            noise_level=0.2,
            missing_rate=0.0,
            feature_correlation=0.3,
            random_state=42
        )
    
    def generate_data(self, params: SimulationParameters) -> Tuple[np.ndarray, np.ndarray]:
        """Generate stock price data."""
        params.n_classes = 2
        return self.generator.generate_classification(params, dataset_type='standard')
    
    def get_insights(self) -> Dict[str, str]:
        return {
            'challenge': "High noise from market volatility and unexpected events",
            'metrics_to_watch': "Precision and F1 score - balance between false positives and negatives",
            'real_world_impact': "A 51% accuracy might still be profitable if fees are low",
            'data_challenges': "Market regimes change; historical patterns may not hold in the future",
        }


class CreditRiskCase(CaseStudy):
    """Credit default prediction case study."""
    
    def __init__(self):
        super().__init__(
            name="Credit Default Prediction",
            description="Predict loan default risk. Features include credit history, income, debt, and personal characteristics.",
            problem_type="classification"
        )
    
    def get_recommended_parameters(self) -> SimulationParameters:
        """Parameters for credit risk."""
        return SimulationParameters(
            n_samples=10000,
            n_features=30,
            n_informative=25,
            n_redundant=5,
            n_classes=2,
            noise_level=0.1,
            missing_rate=0.05,
            class_imbalance=[0.95, 0.05],  # Most loans don't default
            outlier_rate=0.02,
            random_state=42
        )
    
    def generate_data(self, params: SimulationParameters) -> Tuple[np.ndarray, np.ndarray]:
        """Generate credit data."""
        params.n_classes = 2
        return self.generator.generate_classification(params, dataset_type='standard')
    
    def get_insights(self) -> Dict[str, str]:
        return {
            'challenge': "Severe class imbalance and regulatory constraints",
            'metrics_to_watch': "Recall (catch defaulters) and compliance with fair lending laws",
            'real_world_impact': "False positives deny credit to good customers; false negatives cost money from defaults",
            'fairness_concern': "Models must not discriminate by protected attributes (age, race, gender)",
        }


class StudentPerformanceCase(CaseStudy):
    """Student academic performance prediction case study."""
    
    def __init__(self):
        super().__init__(
            name="Student Performance Prediction",
            description="Predict student success/failure. Features include study hours, attendance, prior GPA, socioeconomic factors.",
            problem_type="classification"
        )
    
    def get_recommended_parameters(self) -> SimulationParameters:
        """Parameters for student performance."""
        return SimulationParameters(
            n_samples=3000,
            n_features=15,
            n_informative=12,
            n_redundant=3,
            n_classes=2,
            noise_level=0.12,
            missing_rate=0.03,
            class_imbalance=[0.8, 0.2],  # Most students succeed
            random_state=42
        )
    
    def generate_data(self, params: SimulationParameters) -> Tuple[np.ndarray, np.ndarray]:
        """Generate student data."""
        params.n_classes = 2
        return self.generator.generate_classification(params, dataset_type='standard')
    
    def get_insights(self) -> Dict[str, str]:
        return {
            'challenge': "Predicting human outcomes; many unmeasured factors affect performance",
            'metrics_to_watch': "Precision and fairness across demographic groups",
            'real_world_impact': "Early prediction enables timely interventions to help struggling students",
            'ethical_note': "Predictions should support, not replace, human judgment from educators",
        }


class WeatherForecastingCase(CaseStudy):
    """Weather prediction regression case study."""
    
    def __init__(self):
        super().__init__(
            name="Weather Forecasting",
            description="Predict temperature or rainfall amount. Features are atmospheric measurements and seasonal indicators.",
            problem_type="regression"
        )
    
    def get_recommended_parameters(self) -> SimulationParameters:
        """Parameters for weather."""
        return SimulationParameters(
            n_samples=5000,
            n_features=25,
            n_informative=20,
            n_redundant=5,
            noise_level=0.15,
            missing_rate=0.05,
            feature_correlation=0.4,
            random_state=42
        )
    
    def generate_data(self, params: SimulationParameters) -> Tuple[np.ndarray, np.ndarray]:
        """Generate weather data."""
        return self.generator.generate_regression(params, pattern='polynomial')
    
    def get_insights(self) -> Dict[str, str]:
        return {
            'challenge': "Complex non-linear relationships; chaotic dynamics",
            'metrics_to_watch': "MAE and RMSE - what prediction error is acceptable?",
            'real_world_impact': "Better forecasts save lives and enable better planning",
            'data_challenges': "Sensor errors and spatial heterogeneity require careful preprocessing",
        }


class HousePricingCase(CaseStudy):
    """Real estate pricing prediction case study."""
    
    def __init__(self):
        super().__init__(
            name="House Price Prediction",
            description="Predict house prices from features like location, size, age, amenities.",
            problem_type="regression"
        )
    
    def get_recommended_parameters(self) -> SimulationParameters:
        """Parameters for house pricing."""
        return SimulationParameters(
            n_samples=4000,
            n_features=20,
            n_informative=18,
            n_redundant=2,
            noise_level=0.1,
            missing_rate=0.02,
            outlier_rate=0.02,
            feature_correlation=0.3,
            random_state=42
        )
    
    def generate_data(self, params: SimulationParameters) -> Tuple[np.ndarray, np.ndarray]:
        """Generate house price data."""
        return self.generator.generate_regression(params, pattern='polynomial')
    
    def get_insights(self) -> Dict[str, str]:
        return {
            'challenge': "Non-linear pricing with location-dependent factors and market shifts",
            'metrics_to_watch': "MAPE (Mean Absolute Percentage Error) - percentage errors matter for expensive items",
            'real_world_impact': "Prices directly affect buyers/sellers; models must be transparent",
            'data_challenges': "Outliers (luxury properties) can skew models; location data needs careful encoding",
        }


class TimeSeriesForecastingCase(CaseStudy):
    """Time series forecasting case study."""
    
    def __init__(self):
        super().__init__(
            name="Sales Forecasting",
            description="Forecast future sales from historical time series data with seasonal patterns and trends.",
            problem_type="time_series"
        )
    
    def get_recommended_parameters(self) -> SimulationParameters:
        """Parameters for time series."""
        return SimulationParameters(
            n_samples=730,  # ~2 years of daily data
            noise_level=0.1,
            random_state=42
        )
    
    def generate_data(self, params: SimulationParameters) -> Tuple[pd.DataFrame, None]:
        """Generate time series data."""
        return self.generator.generate_time_series(params, pattern='trend_seasonal'), None
    
    def get_insights(self) -> Dict[str, str]:
        return {
            'challenge': "Temporal dependencies, seasonality, trends, and external events",
            'metrics_to_watch': "MAPE for percentage errors; forecast intervals for uncertainty",
            'real_world_impact': "Forecasts drive inventory, staffing, and supply chain decisions",
            'data_challenges': "External factors (promotions, holidays, competition) require feature engineering",
        }


class CaseStudyLibrary:
    """Library of predefined case studies."""
    
    CASES = {
        'spam_detection': SpamDetectionCase,
        'stock_prediction': StockPriceMovementCase,
        'credit_risk': CreditRiskCase,
        'student_performance': StudentPerformanceCase,
        'weather_forecast': WeatherForecastingCase,
        'house_pricing': HousePricingCase,
        'sales_forecast': TimeSeriesForecastingCase,
    }
    
    @classmethod
    def get_case(cls, case_id: str) -> CaseStudy:
        """Get a case study by ID."""
        if case_id not in cls.CASES:
            raise ValueError(f"Unknown case study: {case_id}")
        return cls.CASES[case_id]()
    
    @classmethod
    def list_cases(cls) -> Dict[str, str]:
        """List available case studies with descriptions."""
        cases = {}
        for case_id, case_class in cls.CASES.items():
            case = case_class()
            cases[case_id] = {
                'name': case.name,
                'description': case.description,
                'type': case.problem_type
            }
        return cases
    
    @classmethod
    def get_learning_path(cls) -> Dict[str, list]:
        """Get recommended learning path through case studies."""
        return {
            'Beginner': [
                'student_performance',
                'house_pricing',
            ],
            'Intermediate': [
                'spam_detection',
                'credit_risk',
                'weather_forecast',
            ],
            'Advanced': [
                'stock_prediction',
                'sales_forecast',
            ]
        }
