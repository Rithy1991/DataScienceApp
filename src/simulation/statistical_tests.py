"""Statistical Testing and A/B Testing Simulators."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.stats import (
    ttest_ind, mannwhitneyu, chi2_contingency,
    shapiro, levene, f_oneway, kruskal
)
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StatTestResult:
    """Results from statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float]
    interpretation: str
    recommendation: str


@dataclass
class ABTestResult:
    """Results from A/B test simulation."""
    control_mean: float
    treatment_mean: float
    control_std: float
    treatment_std: float
    n_control: int
    n_treatment: int
    uplift: float
    uplift_pct: float
    p_value: float
    significant: bool
    confidence_interval: Tuple[float, float]
    power: float
    minimum_detectable_effect: float


class StatisticalTestSimulator:
    """Simulate various statistical tests."""
    
    @staticmethod
    def t_test(
        group1: np.ndarray,
        group2: np.ndarray,
        alpha: float = 0.05
    ) -> StatTestResult:
        """Perform independent samples t-test."""
        
        statistic, p_value = ttest_ind(group1, group2)
        significant = p_value < alpha
        
        # Cohen's d for effect size
        pooled_std = np.sqrt(((len(group1) - 1) * np.std(group1, ddof=1)**2 + 
                               (len(group2) - 1) * np.std(group2, ddof=1)**2) / 
                              (len(group1) + len(group2) - 2))
        effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
        
        interpretation = f"Groups {'differ' if significant else 'do not differ'} significantly (p={p_value:.4f})"
        
        if abs(effect_size) < 0.2:
            magnitude = "negligible"
        elif abs(effect_size) < 0.5:
            magnitude = "small"
        elif abs(effect_size) < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        recommendation = f"Effect size is {magnitude} (Cohen's d = {effect_size:.3f})"
        
        return StatTestResult(
            test_name="Independent Samples T-Test",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            effect_size=effect_size,
            interpretation=interpretation,
            recommendation=recommendation
        )
    
    @staticmethod
    def mann_whitney_test(
        group1: np.ndarray,
        group2: np.ndarray,
        alpha: float = 0.05
    ) -> StatTestResult:
        """Perform Mann-Whitney U test (non-parametric)."""
        
        statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        significant = p_value < alpha
        
        # Rank-biserial correlation for effect size
        n1, n2 = len(group1), len(group2)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        
        interpretation = f"Groups {'differ' if significant else 'do not differ'} significantly (p={p_value:.4f})"
        recommendation = f"Use when normality assumptions are violated. Effect size: {effect_size:.3f}"
        
        return StatTestResult(
            test_name="Mann-Whitney U Test",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            effect_size=effect_size,
            interpretation=interpretation,
            recommendation=recommendation
        )
    
    @staticmethod
    def chi_square_test(
        observed: np.ndarray,
        alpha: float = 0.05
    ) -> StatTestResult:
        """Perform Chi-square test of independence."""
        
        statistic, p_value, dof, expected = chi2_contingency(observed)
        significant = p_value < alpha
        
        # Cramér's V for effect size
        n = observed.sum()
        min_dim = min(observed.shape[0], observed.shape[1]) - 1
        effect_size = np.sqrt(statistic / (n * min_dim)) if min_dim > 0 else 0
        
        interpretation = f"Variables {'are' if significant else 'are not'} independent (p={p_value:.4f})"
        recommendation = f"Cramér's V = {effect_size:.3f}. Expected frequencies should be > 5."
        
        return StatTestResult(
            test_name="Chi-Square Test of Independence",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            effect_size=effect_size,
            interpretation=interpretation,
            recommendation=recommendation
        )
    
    @staticmethod
    def anova_test(
        *groups: np.ndarray,
        alpha: float = 0.05
    ) -> StatTestResult:
        """Perform one-way ANOVA."""
        
        statistic, p_value = f_oneway(*groups)
        significant = p_value < alpha
        
        # Eta-squared for effect size
        grand_mean = np.mean(np.concatenate(groups))
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = sum(sum((x - grand_mean)**2 for x in g) for g in groups)
        effect_size = ss_between / ss_total if ss_total > 0 else 0
        
        interpretation = f"Groups {'differ' if significant else 'do not differ'} significantly (p={p_value:.4f})"
        recommendation = f"Eta-squared = {effect_size:.3f}. Post-hoc tests needed if significant."
        
        return StatTestResult(
            test_name="One-Way ANOVA",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            effect_size=effect_size,
            interpretation=interpretation,
            recommendation=recommendation
        )
    
    @staticmethod
    def normality_test(
        data: np.ndarray,
        alpha: float = 0.05
    ) -> StatTestResult:
        """Test for normality using Shapiro-Wilk test."""
        
        statistic, p_value = shapiro(data)
        significant = p_value < alpha
        
        interpretation = f"Data {'is not' if significant else 'is'} normally distributed (p={p_value:.4f})"
        recommendation = "Use parametric tests if normal, non-parametric otherwise."
        
        return StatTestResult(
            test_name="Shapiro-Wilk Normality Test",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            effect_size=None,
            interpretation=interpretation,
            recommendation=recommendation
        )
    
    @staticmethod
    def variance_test(
        group1: np.ndarray,
        group2: np.ndarray,
        alpha: float = 0.05
    ) -> StatTestResult:
        """Test for equal variances using Levene's test."""
        
        statistic, p_value = levene(group1, group2)
        significant = p_value < alpha
        
        interpretation = f"Variances {'differ' if significant else 'are equal'} (p={p_value:.4f})"
        recommendation = "Use Welch's t-test if variances differ."
        
        return StatTestResult(
            test_name="Levene's Test for Equality of Variances",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            effect_size=None,
            interpretation=interpretation,
            recommendation=recommendation
        )


class ABTestSimulator:
    """Comprehensive A/B testing simulator."""
    
    @staticmethod
    def calculate_sample_size(
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.80
    ) -> int:
        """Calculate required sample size for A/B test."""
        
        from scipy.stats import norm
        
        # Z-scores
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)
        
        p_pooled = (p1 + p2) / 2
        
        n = ((z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) + 
              z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2 / 
             (p2 - p1)**2)
        
        return int(np.ceil(n))
    
    @staticmethod
    def simulate_ab_test(
        control_mean: float,
        treatment_mean: float,
        control_std: float,
        treatment_std: float,
        n_samples: int,
        alpha: float = 0.05
    ) -> ABTestResult:
        """Simulate A/B test with continuous metric."""
        
        # Generate samples
        control = np.random.normal(control_mean, control_std, n_samples)
        treatment = np.random.normal(treatment_mean, treatment_std, n_samples)
        
        # Calculate statistics
        t_stat, p_value = ttest_ind(treatment, control)
        significant = p_value < alpha
        
        uplift = treatment_mean - control_mean
        uplift_pct = (uplift / control_mean * 100) if control_mean != 0 else 0
        
        # Confidence interval for difference
        se = np.sqrt(control_std**2 / n_samples + treatment_std**2 / n_samples)
        ci_margin = 1.96 * se
        ci = (uplift - ci_margin, uplift + ci_margin)
        
        # Statistical power
        effect_size = uplift / np.sqrt((control_std**2 + treatment_std**2) / 2)
        from scipy.stats import nct
        ncp = effect_size * np.sqrt(n_samples / 2)
        critical_value = stats.t.ppf(1 - alpha / 2, 2 * n_samples - 2)
        power = 1 - nct.cdf(critical_value, 2 * n_samples - 2, ncp)
        
        # MDE
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(0.8)
        mde = (z_alpha + z_beta) * se
        
        return ABTestResult(
            control_mean=control.mean(),
            treatment_mean=treatment.mean(),
            control_std=control.std(),
            treatment_std=treatment.std(),
            n_control=n_samples,
            n_treatment=n_samples,
            uplift=uplift,
            uplift_pct=uplift_pct,
            p_value=p_value,
            significant=significant,
            confidence_interval=ci,
            power=power,
            minimum_detectable_effect=mde
        )
    
    @staticmethod
    def simulate_conversion_test(
        control_rate: float,
        treatment_rate: float,
        n_samples: int,
        alpha: float = 0.05
    ) -> ABTestResult:
        """Simulate A/B test with binary conversion metric."""
        
        # Generate samples
        control = np.random.binomial(1, control_rate, n_samples)
        treatment = np.random.binomial(1, treatment_rate, n_samples)
        
        # Z-test for proportions
        p1 = control.mean()
        p2 = treatment.mean()
        p_pooled = (control.sum() + treatment.sum()) / (2 * n_samples)
        
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_samples + 1/n_samples))
        z_stat = (p2 - p1) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        significant = p_value < alpha
        
        uplift = p2 - p1
        uplift_pct = (uplift / p1 * 100) if p1 > 0 else 0
        
        # Confidence interval
        ci_margin = 1.96 * se
        ci = (uplift - ci_margin, uplift + ci_margin)
        
        # Power calculation
        from statsmodels.stats.power import zt_ind_solve_power
        effect_size = (treatment_rate - control_rate) / np.sqrt(control_rate * (1 - control_rate))
        try:
            power = zt_ind_solve_power(
                effect_size=effect_size,
                nobs1=n_samples,
                alpha=alpha,
                ratio=1
            )
        except:
            power = 0.5
        
        # MDE
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(0.8)
        mde = (z_alpha + z_beta) * se
        
        return ABTestResult(
            control_mean=p1,
            treatment_mean=p2,
            control_std=np.sqrt(p1 * (1 - p1)),
            treatment_std=np.sqrt(p2 * (1 - p2)),
            n_control=n_samples,
            n_treatment=n_samples,
            uplift=uplift,
            uplift_pct=uplift_pct,
            p_value=p_value,
            significant=significant,
            confidence_interval=ci,
            power=power,
            minimum_detectable_effect=mde
        )
    
    @staticmethod
    def sequential_testing_simulation(
        control_rate: float,
        treatment_rate: float,
        max_samples: int = 10000,
        check_frequency: int = 100,
        alpha: float = 0.05
    ) -> Dict:
        """Simulate sequential A/B testing with multiple looks."""
        
        results = {
            'sample_sizes': [],
            'p_values': [],
            'effect_sizes': [],
            'stopped_early': False,
            'final_n': 0,
            'decision': None
        }
        
        control_data = []
        treatment_data = []
        
        for n in range(check_frequency, max_samples + 1, check_frequency):
            # Generate new data
            control_data.extend(np.random.binomial(1, control_rate, check_frequency))
            treatment_data.extend(np.random.binomial(1, treatment_rate, check_frequency))
            
            # Test
            p1 = np.mean(control_data)
            p2 = np.mean(treatment_data)
            
            if p1 > 0 and p2 > 0:
                p_pooled = (sum(control_data) + sum(treatment_data)) / (2 * n)
                se = np.sqrt(p_pooled * (1 - p_pooled) * (2 / n))
                z_stat = (p2 - p1) / se if se > 0 else 0
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                
                results['sample_sizes'].append(n)
                results['p_values'].append(p_value)
                results['effect_sizes'].append(p2 - p1)
                
                # Check for early stopping (with adjusted alpha for multiple testing)
                adjusted_alpha = alpha / np.log(n / check_frequency + 1)
                if p_value < adjusted_alpha:
                    results['stopped_early'] = True
                    results['final_n'] = n
                    results['decision'] = 'significant'
                    break
        
        if not results['stopped_early']:
            results['final_n'] = max_samples
            results['decision'] = 'not_significant'
        
        return results
    
    @staticmethod
    def power_analysis(
        baseline_rate: float,
        effect_sizes: List[float],
        sample_sizes: List[int],
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """Calculate statistical power for various effect sizes and sample sizes."""
        
        from statsmodels.stats.power import zt_ind_solve_power
        
        results = []
        
        for effect_pct in effect_sizes:
            for n in sample_sizes:
                treatment_rate = baseline_rate * (1 + effect_pct)
                effect_size = (treatment_rate - baseline_rate) / np.sqrt(baseline_rate * (1 - baseline_rate))
                
                try:
                    power = zt_ind_solve_power(
                        effect_size=effect_size,
                        nobs1=n,
                        alpha=alpha,
                        ratio=1
                    )
                except:
                    power = 0.0
                
                results.append({
                    'baseline_rate': baseline_rate,
                    'effect_pct': effect_pct * 100,
                    'sample_size': n,
                    'power': power
                })
        
        return pd.DataFrame(results)


class BayesianABTesting:
    """Bayesian approach to A/B testing."""
    
    @staticmethod
    def beta_binomial_test(
        control_conversions: int,
        control_total: int,
        treatment_conversions: int,
        treatment_total: int,
        n_simulations: int = 10000
    ) -> Dict:
        """Bayesian A/B test using Beta-Binomial model."""
        
        # Prior: Beta(1, 1) - uniform
        # Posterior: Beta(alpha + conversions, beta + non-conversions)
        
        control_alpha = 1 + control_conversions
        control_beta = 1 + (control_total - control_conversions)
        
        treatment_alpha = 1 + treatment_conversions
        treatment_beta = 1 + (treatment_total - treatment_conversions)
        
        # Sample from posteriors
        control_samples = np.random.beta(control_alpha, control_beta, n_simulations)
        treatment_samples = np.random.beta(treatment_alpha, treatment_beta, n_simulations)
        
        # Probability that treatment is better
        prob_treatment_better = (treatment_samples > control_samples).mean()
        
        # Expected loss if we choose wrong variant
        expected_loss_treatment = np.maximum(control_samples - treatment_samples, 0).mean()
        expected_loss_control = np.maximum(treatment_samples - control_samples, 0).mean()
        
        # Credible intervals
        control_ci = np.percentile(control_samples, [2.5, 97.5])
        treatment_ci = np.percentile(treatment_samples, [2.5, 97.5])
        
        # Uplift
        uplift_samples = (treatment_samples - control_samples) / control_samples
        uplift_mean = uplift_samples.mean()
        uplift_ci = np.percentile(uplift_samples, [2.5, 97.5])
        
        return {
            'prob_treatment_better': prob_treatment_better,
            'expected_loss_treatment': expected_loss_treatment,
            'expected_loss_control': expected_loss_control,
            'control_rate_mean': control_samples.mean(),
            'treatment_rate_mean': treatment_samples.mean(),
            'control_ci': control_ci,
            'treatment_ci': treatment_ci,
            'uplift_mean': uplift_mean,
            'uplift_ci': uplift_ci,
            'control_samples': control_samples,
            'treatment_samples': treatment_samples
        }
