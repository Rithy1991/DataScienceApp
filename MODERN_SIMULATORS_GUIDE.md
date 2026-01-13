# 🚀 Modern ML Simulators - Comprehensive Guide

**Last Updated**: January 13, 2026  
**Version**: 2.0.0 - Enhanced with Modern ML Techniques

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [New Simulators Added](#new-simulators-added)
3. [Advanced Scenarios](#advanced-scenarios)
4. [Quick Start Guide](#quick-start-guide)
5. [Detailed Simulator Documentation](#detailed-simulator-documentation)
6. [Integration Examples](#integration-examples)
7. [Best Practices](#best-practices)

---

## Overview

This enhanced simulation suite now includes **10+ modern ML simulators** covering:
- **Distributed Learning** (Federated Learning with Privacy)
- **Model Explainability** (SHAP/LIME-like techniques)
- **Fairness & Bias Detection** (Comprehensive fairness metrics)
- **Active Learning** (Efficient labeling strategies)
- **Transfer Learning** (Domain adaptation techniques)
- **Multi-Modal Learning** (Combining different data types)
- **Adversarial ML** (Attack simulations)
- **Drift Detection** (Concept & data drift)
- **Advanced Clustering** (Multiple algorithms & dimensionality reduction)
- **Neural Architecture Search** (AutoML for neural networks)
- **Anomaly Detection** (Multiple detection methods)
- **Ensemble Methods** (Comparison & analysis)

---

## New Simulators Added

### 1. **Federated Learning Simulator** 🌐
**File**: `src/simulation/advanced_simulators.py`

Simulate decentralized machine learning with privacy preservation.

**Key Features**:
- Multi-client training (split data across clients)
- IID and non-IID data distributions
- Differential Privacy integration
- Federated Averaging (FedAvg) algorithm
- Privacy budget tracking
- Communication round monitoring

**Use Cases**:
- Mobile device learning
- Hospital networks (healthcare privacy)
- Financial institution collaboration
- Edge device optimization

**Example**:
```python
from src.simulation.advanced_simulators import FederatedLearningSimulator

sim = FederatedLearningSimulator()

# Generate data for 5 clients
client_data = sim.split_data_to_clients(X, y, n_clients=5, iid=False)

# Run federated training with differential privacy
result = sim.federated_averaging(
    X_train_clients=client_data,
    X_test=X_test,
    y_test=y_test,
    n_rounds=20,
    privacy_mechanism='differential_privacy',
    epsilon=1.0
)

print(f"Global Accuracy: {result.global_accuracies[-1]:.4f}")
print(f"Convergence Round: {result.convergence_round}")
```

---

### 2. **Explainability Simulator** 💡
**File**: `src/simulation/advanced_simulators.py`

Understand model decisions with SHAP-like and LIME-like explanations.

**Key Features**:
- Feature importance (tree-based models)
- Permutation importance
- LIME-like local linear explanations
- Instance-level explanations
- Global model understanding

**Use Cases**:
- Model debugging
- Stakeholder communication
- Regulatory compliance (GDPR, explainability requirements)
- Trust building

**Example**:
```python
from src.simulation.advanced_simulators import ExplainabilitySimulator

explainer = ExplainabilitySimulator()

# Explain entire model
result = explainer.explain_model(
    model=trained_model,
    X=X_test,
    y=y_test,
    method='permutation',
    feature_names=['age', 'income', 'credit_score', ...]
)

print("Global Feature Importance:")
for feat, imp in sorted(result.feature_importance.items(), 
                        key=lambda x: abs(x[1]), reverse=True)[:5]:
    print(f"  {feat}: {imp:.4f}")

# Explain single instance
for explanation in result.sample_explanations:
    print(f"\nInstance {explanation['instance_idx']}:")
    print(f"  Prediction: {explanation['prediction']}")
    print(f"  Top contributing features: {explanation['local_coefficients']}")
```

---

### 3. **Fairness Simulator** ⚖️
**File**: `src/simulation/advanced_simulators.py`

Detect and quantify bias in ML models.

**Key Features**:
- Demographic parity measurement
- Equalized odds analysis
- Disparate impact ratio
- Group-specific accuracy analysis
- Fairness metrics comparison

**Fairness Metrics**:
| Metric | Definition | Range |
|--------|-----------|-------|
| **Demographic Parity** | Difference in positive rates between groups | 0 (fair) - 1 (unfair) |
| **Equalized Odds** | TPR and FPR differences | 0 (fair) - 1 (unfair) |
| **Disparate Impact** | Ratio of positive rates (4/5 rule) | 0 (biased) - 1 (fair) |
| **Equal Opportunity** | TPR difference only | 0 (fair) - 1 (unfair) |

**Example**:
```python
from src.simulation.advanced_simulators import FairnessSimulator

fair_sim = FairnessSimulator()

# Generate biased data
X, y, protected_attr = fair_sim.generate_biased_data(
    n_samples=1000,
    protected_attribute_bias=0.3
)

# Evaluate fairness
result = fair_sim.evaluate_fairness(
    model=trained_model,
    X=X,
    y=y,
    protected=protected_attr
)

print(f"Overall Accuracy: {result.overall_accuracy:.4f}")
print(f"Group Accuracies: {result.group_accuracies}")
print(f"Demographic Parity: {result.demographic_parity:.4f}")
print(f"Fairness Score (lower better): {result.fairness_score:.4f}")
```

---

### 4. **Active Learning Simulator** 🎯
**File**: `src/simulation/advanced_simulators.py`

Minimize labeling costs with intelligent sample selection.

**Key Strategies**:
- **Uncertainty Sampling**: Select high-uncertainty predictions
- **Margin Sampling**: Pick samples with smallest margin between top classes
- **Entropy Sampling**: Choose high-entropy predictions
- **Random Sampling**: Baseline comparison

**Use Cases**:
- Limited annotation budgets
- Medical image labeling
- NLP dataset creation
- Active feedback loops

**Example**:
```python
from src.simulation.advanced_simulators import ActiveLearningSimulator

al_sim = ActiveLearningSimulator()

result = al_sim.simulate_active_learning(
    X=X_full,
    y=y_full,
    initial_samples=20,
    n_iterations=10,
    samples_per_iteration=10,
    strategy='uncertainty'
)

print(f"Learning Curve (Uncertainty Strategy):")
for size, acc in zip(result.sample_sizes, result.accuracies):
    print(f"  Samples: {size:3d}, Accuracy: {acc:.4f}")
```

---

### 5. **Transfer Learning Simulator** 🔄
**File**: `src/simulation/advanced_simulators.py`

Leverage source domain knowledge for target domain tasks.

**Key Strategies**:
- **Freeze All**: Use source model as-is (zero-shot)
- **Freeze Partial**: Freeze early layers, fine-tune last layer
- **Fine-Tune All**: Retrain all layers on target data

**Use Cases**:
- Small target datasets
- Domain adaptation
- Pre-trained model utilization
- Cost-effective learning

**Example**:
```python
from src.simulation.advanced_simulators import TransferLearningSimulator

tl_sim = TransferLearningSimulator()

# Generate source and target domain data
X_src, y_src, X_tgt_train, y_tgt_train = tl_sim.generate_source_target_data(
    n_samples_source=1000,
    n_samples_target=200,
    domain_shift=1.0
)

# Compare transfer strategies
X_tgt_test, y_tgt_test = ..., ...

result = tl_sim.simulate_transfer_learning(
    X_source=X_src,
    y_source=y_src,
    X_target_train=X_tgt_train,
    y_target_train=y_tgt_train,
    X_target_test=X_tgt_test,
    y_target_test=y_tgt_test,
    strategy='freeze_partial'
)

print(f"Source Accuracy: {result.source_accuracy:.4f}")
print(f"Target (No Transfer): {result.target_accuracy_no_transfer:.4f}")
print(f"Target (With Transfer): {result.target_accuracy_with_transfer:.4f}")
print(f"Improvement: {result.improvement:.4f}")
```

---

### 6. **Multi-Modal Learning Simulator** 🎨🔊
**File**: `src/simulation/advanced_simulators.py`

Combine multiple data modalities for better predictions.

**Fusion Strategies**:
- **Early Fusion**: Concatenate features before training
- **Late Fusion**: Train separate models, combine predictions
- **Hybrid Fusion**: Extract features from each modality, then combine

**Use Cases**:
- Vision + Audio (video understanding)
- Text + Images (social media analysis)
- Multimodal medical data
- Sensor fusion (IoT)

**Example**:
```python
from src.simulation.advanced_simulators import MultiModalSimulator

mm_sim = MultiModalSimulator()

# Generate multi-modal data
X_mod1, X_mod2, y = mm_sim.generate_multimodal_data(
    n_samples=1000,
    n_features_mod1=10,
    n_features_mod2=15,
    correlation=0.7
)

# Compare fusion strategies
result = mm_sim.compare_fusion_strategies(X_mod1, X_mod2, y)

print(f"Modality 1 Only: {result.accuracy_modality1_only:.4f}")
print(f"Modality 2 Only: {result.accuracy_modality2_only:.4f}")
print(f"Combined ({result.fusion_strategy}): {result.accuracy_combined:.4f}")
print(f"Improvement: {result.fusion_improvement:.4f}")
```

---

### 7. **Adversarial ML Simulator** ⚔️
**File**: `src/simulation/scenarios.py`

Test model robustness against attacks.

**Attack Types**:
- **Evasion Attacks**: Fool model at test time
- **Data Poisoning**: Corrupt training data
- **Backdoor Attacks**: Inject hidden triggers

**Use Cases**:
- Security testing
- Robustness validation
- Adversarial training
- Red-teaming

**Example**:
```python
from src.simulation.scenarios import AdversarialMLSimulator

adv_sim = AdversarialMLSimulator()

# Test evasion attacks
evasion_result = adv_sim.evasion_attack(
    model=trained_model,
    X=X_test,
    y=y_test,
    epsilon=0.1,
    n_samples=100
)

print(f"Original Accuracy: {evasion_result['original_accuracy']:.4f}")
print(f"Under Attack Accuracy: {evasion_result['adversarial_accuracy']:.4f}")
print(f"Attack Success Rate: {evasion_result['attack_success_rate']:.4f}")

# Test data poisoning
poison_result = adv_sim.poisoning_attack(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    poison_rate=0.1
)

print(f"\nPoison Rate: {poison_result['poison_rate']:.1%}")
print(f"Clean Accuracy: {poison_result['clean_accuracy']:.4f}")
print(f"Poisoned Accuracy: {poison_result['poisoned_accuracy']:.4f}")
print(f"Attack Effectiveness: {poison_result['attack_effectiveness']:.4f}")
```

---

### 8. **Drift Detection Simulator** 📊
**File**: `src/simulation/scenarios.py`

Monitor and detect distribution changes over time.

**Drift Types**:
- **Sudden Drift**: Abrupt distribution change
- **Gradual Drift**: Slow, continuous change
- **Recurring Drift**: Cyclical patterns
- **Incremental Drift**: New data influences slowly

**Detection Methods**:
- **DDM** (Drift Detection Method): Monitors error rate
- **ADWIN** (Adaptive Windowing): Adapts window size

**Use Cases**:
- Online learning
- Model monitoring
- Automatic retraining
- Production ML systems

**Example**:
```python
from src.simulation.scenarios import DriftDetectionSimulator

drift_sim = DriftDetectionSimulator()

# Generate data with drift
X, y, drift_indicator = drift_sim.generate_drift_data(
    n_samples=2000,
    drift_type='gradual',
    drift_position=0.5
)

# Monitor with drift detection
result = drift_sim.simulate_drift_monitoring(
    X=X,
    y=y,
    drift_indicator=drift_indicator,
    retrain_on_drift=True
)

print(f"Number of Detected Drifts: {result['n_retrains']}")
print(f"Retraining Triggered At: {result['retrain_points']}")
print(f"True Drift Points: {result['true_drift_points']}")
print(f"Final Accuracies: {result['accuracies'][-10:]}")
```

---

### 9. **Clustering Simulator** 🎯
**File**: `src/simulation/advanced_simulators.py`

Compare clustering algorithms and explore data structure.

**Algorithms**:
- K-Means
- DBSCAN
- Hierarchical Clustering

**Dimensionality Reduction**:
- PCA
- t-SNE
- Isomap

**Metrics**:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

**Example**:
```python
from src.simulation.advanced_simulators import ClusteringSimulator

cluster_sim = ClusteringSimulator()

# Generate non-trivial data
X = cluster_sim.generate_clustering_data(
    n_samples=500,
    n_clusters=4,
    pattern='moons'
)

# Compare algorithms
for algo in ['kmeans', 'dbscan', 'hierarchical']:
    result = cluster_sim.run_clustering(
        X=X,
        algorithm=algo,
        reduce_dim='tsne'
    )
    print(f"{algo}: Silhouette={result.silhouette:.3f}")
```

---

### 10. **Neural Network Architecture Search** 🧠
**File**: `src/simulation/advanced_simulators.py`

Automatically search for optimal neural network architectures.

**Features**:
- Random architecture search
- Hyperparameter tuning
- Activation function comparison
- Learning rate optimization
- Convergence tracking

**Example**:
```python
from src.simulation.advanced_simulators import NeuralNetworkSimulator

nn_sim = NeuralNetworkSimulator()

results = nn_sim.simulate_architecture_search(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    problem_type='classification',
    n_trials=20
)

# Top 3 architectures
for i, result in enumerate(results[:3]):
    print(f"\n{i+1}. Architecture: {result.architecture}")
    print(f"   Activation: {result.activation}")
    print(f"   Val Score: {result.final_score:.4f}")
```

---

## Advanced Scenarios

### Adversarial ML Scenarios

#### Scenario: E-commerce Recommendation Attack
```python
# An attacker poisons training data to promote specific items
adv_sim = AdversarialMLSimulator()
result = adv_sim.backdoor_attack(
    X_train=user_behavior,
    y_train=purchased,
    X_test=test_behavior,
    y_test=test_purchased,
    backdoor_rate=0.05,
    target_class=1  # Force positive recommendation
)
```

#### Scenario: Fraud Detection Evasion
```python
# Fraudsters craft transactions to evade detection
evasion_result = adv_sim.evasion_attack(
    model=fraud_detector,
    X=fraudulent_txns,
    epsilon=0.05  # Small perturbations
)
print(f"Evasion Success Rate: {evasion_result['attack_success_rate']:.1%}")
```

---

## Quick Start Guide

### Installation
```bash
# All simulators are already integrated
from src.simulation import *
```

### Basic Usage Pattern
```python
# 1. Create simulator instance
simulator = YourSimulator()

# 2. Prepare data
X_train, X_test, y_train, y_test = prepare_data()

# 3. Run simulation
result = simulator.simulate_or_analyze(
    X=X_train,
    y=y_train,
    **parameters
)

# 4. Analyze results
print(result)  # Structured result object
```

### Full Example: Complete ML Fairness Audit
```python
from src.simulation.advanced_simulators import (
    FairnessSimulator,
    ExplainabilitySimulator,
    AdversarialMLSimulator
)

# 1. Fairness check
fair_sim = FairnessSimulator()
X, y, protected = fair_sim.generate_biased_data(n_samples=1000)
model = train_model(X, y)
fairness_result = fair_sim.evaluate_fairness(model, X, y, protected)

# 2. Explainability check
explainer = ExplainabilitySimulator()
explain_result = explainer.explain_model(model, X, y, method='permutation')

# 3. Adversarial robustness check
adv_sim = AdversarialMLSimulator()
evasion = adv_sim.evasion_attack(model, X, y, epsilon=0.1)

# 4. Summary report
print(f"""
=== ML Audit Report ===
Fairness Score: {fairness_result.fairness_score:.4f} (lower is better)
Top Features: {list(explain_result.feature_importance.keys())[:3]}
Adversarial Vulnerability: {evasion['accuracy_drop']:.4f}
""")
```

---

## Detailed Simulator Documentation

### FederatedLearningSimulator

```python
class FederatedLearningSimulator:
    def split_data_to_clients(n_clients, iid=True)
        # Split data for distributed training
        
    def add_differential_privacy(gradients, epsilon=1.0)
        # Add privacy protection
        
    def federated_averaging(X_train_clients, X_test, y_test, 
                           n_rounds=20, privacy_mechanism='none')
        # Run FedAvg algorithm
```

**Parameters**:
- `n_clients`: Number of participating clients
- `iid`: Data distribution (True=IID, False=non-IID/biased)
- `epsilon`: Privacy budget (lower = more private)
- `n_rounds`: Communication rounds

**Returns**:
- `global_accuracies`: List of test accuracies per round
- `client_accuracies`: Individual client accuracies
- `convergence_round`: When training converged

---

### ExplainabilitySimulator

```python
class ExplainabilitySimulator:
    def compute_feature_importance(model, X, feature_names=None)
        # Tree feature importance
        
    def permutation_importance(model, X, y, n_repeats=10)
        # Permutation-based importance
        
    def explain_instance_local_linear(model, X, instance_idx, n_samples=5000)
        # LIME-like local explanations
        
    def explain_model(model, X, y, method='permutation', 
                     feature_names=None, sample_indices=None)
        # Comprehensive explanation
```

---

### FairnessSimulator

```python
class FairnessSimulator:
    def generate_biased_data(n_samples, protected_attribute_bias)
        # Create test data with bias
        
    def compute_demographic_parity(y_pred, protected)
        # Measure prediction rate difference
        
    def compute_equalized_odds(y_true, y_pred, protected)
        # Measure TPR and FPR parity
        
    def evaluate_fairness(model, X, y, protected)
        # Full fairness evaluation
```

---

### ActiveLearningSimulator

```python
class ActiveLearningSimulator:
    def uncertainty_sampling(model, X_pool, n_samples)
        # Select uncertain samples
        
    def margin_sampling(model, X_pool, n_samples)
        # Select margin samples
        
    def simulate_active_learning(X, y, initial_samples=20,
                                 n_iterations=10, strategy='uncertainty')
        # Full AL simulation
```

---

### TransferLearningSimulator

```python
class TransferLearningSimulator:
    def generate_source_target_data(n_samples_source, n_samples_target,
                                    domain_shift)
        # Create domain-shifted data
        
    def simulate_transfer_learning(X_source, y_source,
                                  X_target_train, y_target_train,
                                  X_target_test, y_target_test,
                                  strategy='freeze_partial')
        # Compare transfer strategies
```

---

### MultiModalSimulator

```python
class MultiModalSimulator:
    def generate_multimodal_data(n_samples, n_features_mod1, 
                                 n_features_mod2, correlation)
        # Create correlated modalities
        
    def early_fusion(X_mod1, X_mod2, y)
        # Concatenate and train
        
    def late_fusion(X_mod1, X_mod2, y)
        # Train separately, combine
        
    def compare_fusion_strategies(X_mod1, X_mod2, y)
        # Compare all methods
```

---

## Integration Examples

### Example 1: Production ML System Monitoring
```python
from src.simulation.scenarios import DriftDetectionSimulator
from src.simulation.advanced_simulators import FairnessSimulator

class MLMonitor:
    def __init__(self):
        self.drift_detector = DriftDetectionSimulator()
        self.fairness_checker = FairnessSimulator()
    
    def monitor_batch(self, X, y_pred, y_true, protected_attr):
        # Check for drift
        drift_result = self.drift_detector.simulate_drift_monitoring(
            X, y_true, model_class=None
        )
        
        # Check fairness
        fairness_result = self.fairness_checker.evaluate_fairness(
            model_or_predictions=y_pred,
            X=X, y=y_true,
            protected=protected_attr
        )
        
        if drift_result['n_drifts'] > 0:
            print("⚠️  Drift detected! Retraining recommended.")
        
        if fairness_result.fairness_score > 0.1:
            print("⚠️  Fairness concerns detected!")
        
        return {
            'drift': drift_result,
            'fairness': fairness_result
        }
```

### Example 2: Feature Selection with Explainability
```python
from src.simulation.advanced_simulators import ExplainabilitySimulator

class FeatureSelector:
    def __init__(self, model, X, y):
        self.explainer = ExplainabilitySimulator()
        self.model = model
        self.X = X
        self.y = y
    
    def select_top_features(self, n_features=5):
        result = self.explainer.explain_model(
            self.model, self.X, self.y,
            method='permutation'
        )
        
        # Sort by importance
        features = sorted(
            result.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return [f[0] for f in features[:n_features]]
```

### Example 3: Automated Hyperparameter Tuning
```python
from src.simulation.advanced_simulators import NeuralNetworkSimulator

class AutoTuner:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.nn_sim = NeuralNetworkSimulator()
        self.data = (X_train, y_train, X_val, y_val)
    
    def find_best_architecture(self, n_trials=50):
        results = self.nn_sim.simulate_architecture_search(
            *self.data,
            n_trials=n_trials
        )
        
        best = results[0]
        print(f"Best architecture: {best.architecture}")
        print(f"Best activation: {best.activation}")
        print(f"Best score: {best.final_score:.4f}")
        
        return best
```

---

## Best Practices

### 1. **Fairness Assessment**
- ✅ Always check multiple fairness metrics
- ✅ Test on stratified groups
- ✅ Document fairness trade-offs
- ❌ Don't rely on single metric

### 2. **Explainability**
- ✅ Use permutation importance for stability
- ✅ Explain multiple instances (local vs global)
- ✅ Validate explanations with domain experts
- ❌ Don't trust explanations without validation

### 3. **Adversarial Testing**
- ✅ Test with realistic perturbation budgets
- ✅ Use multiple attack types
- ✅ Document attack assumptions
- ❌ Don't assume model is secure without testing

### 4. **Active Learning**
- ✅ Start with random samples
- ✅ Compare strategies
- ✅ Measure learning curves
- ❌ Don't assume strategy works without validation

### 5. **Transfer Learning**
- ✅ Test all freezing strategies
- ✅ Measure domain shift impact
- ✅ Fine-tune on representative data
- ❌ Don't transfer without evaluation

### 6. **Multi-Modal Learning**
- ✅ Verify modality correlation
- ✅ Compare fusion strategies
- ✅ Handle missing modalities
- ❌ Don't assume early fusion is best

### 7. **Drift Monitoring**
- ✅ Set appropriate detection thresholds
- ✅ Monitor multiple windows
- ✅ Trigger retraining early
- ❌ Don't wait for accuracy to drop

---

## Performance Considerations

| Simulator | Scalability | Speed | Memory |
|-----------|-------------|-------|--------|
| Federated Learning | ⭐⭐⭐ (scales with # clients) | ⭐⭐ | ⭐⭐⭐ |
| Explainability | ⭐⭐ | ⭐ | ⭐⭐ |
| Fairness | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Active Learning | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| Transfer Learning | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Multi-Modal | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| Adversarial ML | ⭐ | ⭐ | ⭐⭐ |
| Drift Detection | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| Clustering | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| Neural Architecture Search | ⭐ | ⭐ | ⭐⭐⭐ |

---

## Future Enhancements

### Planned Additions
1. **Continual Learning Simulator** - Learning without catastrophic forgetting
2. **Reinforcement Learning Simulator** - Policy optimization simulation
3. **Graph Neural Network Simulator** - Node and graph classification
4. **Attention Mechanism Simulator** - Transformer explanation
5. **Causal Inference Simulator** - Causal discovery and inference
6. **Meta-Learning Simulator** - Learning to learn fast

### Contributing
To add new simulators:
1. Inherit from base simulator classes
2. Follow the result dataclass pattern
3. Add documentation
4. Update `__init__.py`

---

## References & Further Reading

### Federated Learning
- McMahan et al. (2016): "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- Kairouz et al. (2021): "Advances and Open Problems in Federated Learning"

### Explainability
- Ribeiro et al. (2016): "Why Should I Trust You?" - LIME paper
- Lundberg & Lee (2017): "Unified Framework for Interpreting Model Predictions" - SHAP

### Fairness
- Buolamwini & Buolamwini (2018): "Gender Shades"
- Moritz et al. (2019): "ML Fairness Gym"

### Active Learning
- Freeman (1965): "Pool-based Active Learning"
- Settles (2009): "Active Learning Literature Survey"

### Transfer Learning
- Yosinski et al. (2014): "How Transferable Are Features in Deep Neural Networks?"
- Zhuang et al. (2020): "A Comprehensive Survey on Transfer Learning"

---

## Support & Contact

For issues, questions, or contributions:
- 📧 Email: support@mlsimulations.dev
- 🐛 Issues: GitHub Issues
- 💬 Discussions: GitHub Discussions
- 📚 Docs: Full documentation at docs/

---

**Made with ❤️ for the ML community**

*Last Updated: January 13, 2026*  
*Version: 2.0.0*
