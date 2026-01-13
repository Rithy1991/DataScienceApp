"""Advanced Visualization Components for ML Simulations."""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class AdvancedVisualizer:
    """Advanced visualization methods for ML simulations."""
    
    @staticmethod
    def plot_3d_clusters(
        X: np.ndarray,
        labels: np.ndarray,
        title: str = "3D Cluster Visualization"
    ) -> go.Figure:
        """Create interactive 3D scatter plot of clusters."""
        
        # If more than 3 dimensions, use PCA
        if X.shape[1] > 3:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            X_3d = pca.fit_transform(X)
            explained_var = sum(pca.explained_variance_ratio_)
            axis_labels = [
                f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
                f'PC3 ({pca.explained_variance_ratio_[2]:.2%})'
            ]
        else:
            X_3d = X[:, :3] if X.shape[1] == 3 else np.hstack([X, np.zeros((X.shape[0], 3 - X.shape[1]))])
            axis_labels = ['X', 'Y', 'Z']
            explained_var = None
        
        # Create 3D scatter
        fig = go.Figure()
        
        unique_labels = np.unique(labels)
        colors = px.colors.qualitative.Set2
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            fig.add_trace(go.Scatter3d(
                x=X_3d[mask, 0],
                y=X_3d[mask, 1],
                z=X_3d[mask, 2],
                mode='markers',
                name=f'Cluster {label}' if label >= 0 else 'Noise',
                marker=dict(
                    size=5,
                    color=colors[i % len(colors)],
                    opacity=0.8 if label >= 0 else 0.3
                )
            ))
        
        subtitle = f"Explained Variance: {explained_var:.2%}" if explained_var else ""
        
        fig.update_layout(
            title=f"{title}<br><sub>{subtitle}</sub>",
            scene=dict(
                xaxis_title=axis_labels[0],
                yaxis_title=axis_labels[1],
                zaxis_title=axis_labels[2]
            ),
            height=700,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_decision_boundary_2d(
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        resolution: int = 100
    ) -> go.Figure:
        """Plot decision boundary for 2D classification."""
        
        # Create mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        # Predict on mesh
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Create contour plot
        fig = go.Figure()
        
        # Add decision boundary
        fig.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, resolution),
            y=np.linspace(y_min, y_max, resolution),
            z=Z,
            colorscale='RdBu',
            opacity=0.3,
            showscale=False,
            contours=dict(
                showlabels=True,
                labelfont=dict(size=12, color='white')
            )
        ))
        
        # Add data points
        for label in np.unique(y):
            mask = y == label
            fig.add_trace(go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode='markers',
                name=f'Class {label}',
                marker=dict(
                    size=8,
                    line=dict(width=1, color='white')
                )
            ))
        
        fig.update_layout(
            title="Decision Boundary Visualization",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            height=600
        )
        
        return fig
    
    @staticmethod
    def plot_neural_architecture(
        architecture: List[int],
        activation: str = 'relu'
    ) -> go.Figure:
        """Visualize neural network architecture."""
        
        layers = [architecture[0]] + list(architecture) + [1]  # Input + hidden + output
        max_neurons = max(layers)
        
        fig = go.Figure()
        
        # Draw neurons
        for layer_idx, n_neurons in enumerate(layers):
            x = [layer_idx] * n_neurons
            y = np.linspace(0, max_neurons, n_neurons)
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='markers',
                marker=dict(size=20, color='lightblue', line=dict(width=2, color='darkblue')),
                name=f'Layer {layer_idx}',
                showlegend=False
            ))
        
        # Draw connections (sample only to avoid clutter)
        for layer_idx in range(len(layers) - 1):
            n_current = min(layers[layer_idx], 10)  # Limit connections
            n_next = min(layers[layer_idx + 1], 10)
            
            current_y = np.linspace(0, max_neurons, layers[layer_idx])
            next_y = np.linspace(0, max_neurons, layers[layer_idx + 1])
            
            for i in range(min(n_current, len(current_y))):
                for j in range(min(n_next, len(next_y))):
                    fig.add_trace(go.Scatter(
                        x=[layer_idx, layer_idx + 1],
                        y=[current_y[i], next_y[j]],
                        mode='lines',
                        line=dict(color='gray', width=0.5),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        fig.update_layout(
            title=f"Neural Network Architecture: {architecture}<br><sub>Activation: {activation}</sub>",
            xaxis=dict(title="Layer", tickmode='array', tickvals=list(range(len(layers)))),
            yaxis=dict(title="Neurons", showticklabels=False),
            height=500,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "ROC Curve"
    ) -> go.Figure:
        """Plot ROC curve with AUC."""
        
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC (AUC = {roc_auc:.3f})',
            line=dict(color='#667eea', width=3)
        ))
        
        # Random classifier line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=500,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_precision_recall_curve(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "Precision-Recall Curve"
    ) -> go.Figure:
        """Plot Precision-Recall curve."""
        
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'PR (AP = {ap:.3f})',
            line=dict(color='#f5576c', width=3),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_calibration_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> go.Figure:
        """Plot calibration curve to assess probability predictions."""
        
        from sklearn.calibration import calibration_curve
        
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        
        fig = go.Figure()
        
        # Calibration curve
        fig.add_trace(go.Scatter(
            x=prob_pred,
            y=prob_true,
            mode='lines+markers',
            name='Model',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10)
        ))
        
        # Perfect calibration
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Calibration Curve",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_learning_curves(
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        title: str = "Learning Curves"
    ) -> go.Figure:
        """Plot learning curves showing model performance vs training size."""
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig = go.Figure()
        
        # Training score
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean,
            mode='lines',
            name='Training Score',
            line=dict(color='#667eea', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Validation score
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_mean,
            mode='lines',
            name='Validation Score',
            line=dict(color='#f5576c', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(245, 87, 108, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Training Set Size",
            yaxis_title="Score",
            height=500,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_feature_importance_interactive(
        feature_names: List[str],
        importances: np.ndarray,
        top_n: int = 20
    ) -> go.Figure:
        """Interactive feature importance plot with filtering."""
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        sorted_names = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=sorted_importances,
            y=sorted_names,
            orientation='h',
            marker=dict(
                color=sorted_importances,
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        fig.update_layout(
            title=f"Top {top_n} Feature Importances",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=max(400, top_n * 25),
            yaxis=dict(autorange="reversed")
        )
        
        return fig
    
    @staticmethod
    def plot_confusion_matrix_interactive(
        cm: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> go.Figure:
        """Interactive confusion matrix with percentages."""
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(cm))]
        
        # Normalize for percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create annotations
        annotations = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f"{cm[i, j]}<br>({cm_norm[i, j]:.1%})",
                        showarrow=False,
                        font=dict(color='white' if cm_norm[i, j] > 0.5 else 'black')
                    )
                )
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            annotations=annotations,
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_partial_dependence(
        model: Any,
        X: pd.DataFrame,
        feature_name: str,
        n_points: int = 50
    ) -> go.Figure:
        """Plot partial dependence for a single feature."""
        
        from sklearn.inspection import partial_dependence as pd_func
        
        feature_idx = X.columns.get_loc(feature_name)
        
        # Calculate partial dependence
        pd_result = pd_func(
            model,
            X,
            features=[feature_idx],
            grid_resolution=n_points
        )
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=pd_result['grid_values'][0],
            y=pd_result['average'][0],
            mode='lines',
            line=dict(color='#667eea', width=3),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title=f"Partial Dependence: {feature_name}",
            xaxis_title=feature_name,
            yaxis_title="Partial Dependence",
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_anomaly_scores(
        scores: np.ndarray,
        predictions: np.ndarray,
        title: str = "Anomaly Scores"
    ) -> go.Figure:
        """Plot anomaly scores with threshold."""
        
        fig = go.Figure()
        
        # Normal points
        normal_mask = predictions == 0
        fig.add_trace(go.Scatter(
            x=np.where(normal_mask)[0],
            y=scores[normal_mask],
            mode='markers',
            name='Normal',
            marker=dict(color='green', size=5, opacity=0.6)
        ))
        
        # Anomalies
        anomaly_mask = predictions == 1
        fig.add_trace(go.Scatter(
            x=np.where(anomaly_mask)[0],
            y=scores[anomaly_mask],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=8, symbol='x')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Sample Index",
            yaxis_title="Anomaly Score",
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_ensemble_comparison(
        results: Dict[str, Dict[str, float]]
    ) -> go.Figure:
        """Compare ensemble method performance."""
        
        methods = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score')
        )
        
        colors = px.colors.qualitative.Set2
        
        for idx, metric in enumerate(metrics):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            values = [results[method][metric] for method in methods]
            
            fig.add_trace(
                go.Bar(
                    x=methods,
                    y=values,
                    name=metric.capitalize(),
                    marker_color=colors[idx],
                    showlegend=False,
                    text=[f"{v:.3f}" for v in values],
                    textposition='outside'
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Ensemble Methods Comparison",
            height=700,
            showlegend=False
        )
        
        return fig
