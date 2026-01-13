"""Modern Interactive Visualizations for ML Simulations using Plotly."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class SimulationVisualizer:
    """Create interactive, animated visualizations for ML simulations."""
    
    # Modern color schemes
    COLORS = {
        'primary': '#667eea',
        'secondary': '#f093fb',
        'success': '#48bb78',
        'warning': '#ed8936',
        'danger': '#f56565',
        'info': '#4299e1',
        'train': '#667eea',
        'test': '#f093fb',
        'baseline': '#718096',
    }
    
    GRADIENTS = {
        'purple': ['#667eea', '#764ba2'],
        'sunset': ['#f093fb', '#f5576c'],
        'ocean': ['#4299e1', '#667eea'],
        'forest': ['#48bb78', '#38a169'],
    }
    
    @classmethod
    def plot_classification_results(
        cls,
        result: Dict[str, Any],
        show_decision_boundary: bool = True
    ) -> go.Figure:
        """Create comprehensive classification results visualization."""
        
        model_results = result['model_results']
        data = result['data']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Confusion Matrix',
                'Prediction Confidence',
                'Feature Importance',
                'Train vs Test Accuracy'
            ),
            specs=[
                [{'type': 'heatmap'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'indicator'}]
            ]
        )
        
        # 1. Confusion Matrix
        cm = model_results['confusion_matrix']
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=[f'Pred {i}' for i in range(len(cm))],
                y=[f'True {i}' for i in range(len(cm))],
                colorscale='Purples',
                text=cm,
                texttemplate='%{text}',
                textfont={'size': 14},
                showscale=False
            ),
            row=1, col=1
        )
        
        # 2. Prediction Distribution
        predictions = model_results['predictions']['test']
        unique, counts = np.unique(predictions, return_counts=True)
        fig.add_trace(
            go.Bar(
                x=[f'Class {i}' for i in unique],
                y=counts,
                marker_color=cls.COLORS['primary'],
                text=counts,
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # 3. Feature Importance (if available)
        if 'feature_importance' in model_results:
            importance = model_results['feature_importance']
            top_n = min(10, len(importance))
            top_indices = np.argsort(importance)[-top_n:]
            
            fig.add_trace(
                go.Bar(
                    x=importance[top_indices],
                    y=[f'Feature {i}' for i in top_indices],
                    orientation='h',
                    marker_color=cls.COLORS['secondary'],
                    text=[f'{v:.3f}' for v in importance[top_indices]],
                    textposition='auto'
                ),
                row=2, col=1
            )
        
        # 4. Performance Indicator
        test_acc = model_results['test_accuracy']
        fig.add_trace(
            go.Indicator(
                mode='gauge+number+delta',
                value=test_acc * 100,
                title={'text': 'Test Accuracy (%)'},
                delta={'reference': model_results['train_accuracy'] * 100},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': cls.COLORS['success']},
                    'steps': [
                        {'range': [0, 50], 'color': 'lightgray'},
                        {'range': [50, 75], 'color': cls.COLORS['warning']},
                        {'range': [75, 100], 'color': cls.COLORS['success']}
                    ],
                    'threshold': {
                        'line': {'color': 'red', 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"Classification Results: {model_results['algorithm']}",
            showlegend=False,
            height=800,
            template='plotly_white'
        )
        
        return fig
    
    @classmethod
    def plot_regression_results(cls, result: Dict[str, Any]) -> go.Figure:
        """Create comprehensive regression results visualization."""
        
        model_results = result['model_results']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Actual vs Predicted',
                'Residual Plot',
                'Residual Distribution',
                'Performance Metrics'
            )
        )
        
        y_test = result['data']['y_test']
        y_pred = model_results['predictions']['test']
        residuals = model_results['residuals']['test']
        
        # 1. Actual vs Predicted
        fig.add_trace(
            go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                marker=dict(
                    color=cls.COLORS['primary'],
                    size=8,
                    opacity=0.6,
                    line=dict(width=1, color='white')
                ),
                name='Predictions'
            ),
            row=1, col=1
        )
        
        # Add perfect prediction line
        min_val, max_val = y_test.min(), y_test.max()
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Fit'
            ),
            row=1, col=1
        )
        
        # 2. Residual Plot
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                marker=dict(
                    color=cls.COLORS['secondary'],
                    size=8,
                    opacity=0.6
                ),
                name='Residuals'
            ),
            row=1, col=2
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash='dash', line_color='gray', row=1, col=2)
        
        # 3. Residual Distribution
        fig.add_trace(
            go.Histogram(
                x=residuals,
                marker_color=cls.COLORS['info'],
                nbinsx=30,
                name='Distribution'
            ),
            row=2, col=1
        )
        
        # 4. Performance Metrics Bar Chart
        metrics = {
            'R² Score': model_results['test_r2'],
            'RMSE': -model_results['test_rmse'] / max(abs(y_test.min()), abs(y_test.max())),  # Normalized
            'MAE': -model_results['test_mae'] / max(abs(y_test.min()), abs(y_test.max())),
        }
        
        fig.add_trace(
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                marker_color=[cls.COLORS['success'], cls.COLORS['warning'], cls.COLORS['info']],
                text=[f'{v:.3f}' for v in metrics.values()],
                textposition='auto'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text='Actual Values', row=1, col=1)
        fig.update_yaxes(title_text='Predicted Values', row=1, col=1)
        fig.update_xaxes(title_text='Predicted Values', row=1, col=2)
        fig.update_yaxes(title_text='Residuals', row=1, col=2)
        fig.update_xaxes(title_text='Residual Value', row=2, col=1)
        fig.update_yaxes(title_text='Frequency', row=2, col=1)
        
        fig.update_layout(
            title_text=f"Regression Results: {model_results['algorithm']}",
            showlegend=False,
            height=800,
            template='plotly_white'
        )
        
        return fig
    
    @classmethod
    def plot_model_comparison(
        cls,
        comparison: Dict[str, Any],
        problem_type: str = 'classification'
    ) -> go.Figure:
        """Create side-by-side model comparison visualization."""
        
        algorithms = comparison['algorithms']
        summary = comparison['summary']
        
        # Extract metrics
        if problem_type == 'classification':
            metric_name = 'test_accuracy'
            metric_label = 'Test Accuracy'
        else:
            metric_name = 'test_r2'
            metric_label = 'Test R² Score'
        
        metrics = [summary[algo][metric_name] for algo in algorithms]
        train_times = [summary[algo]['train_time'] for algo in algorithms]
        overfitting = [summary[algo]['overfitting'] for algo in algorithms]
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                f'{metric_label}',
                'Training Time (seconds)',
                'Overfitting Score'
            )
        )
        
        # 1. Performance Comparison
        colors = [cls.COLORS['success'] if m == max(metrics) else cls.COLORS['primary'] 
                  for m in metrics]
        
        fig.add_trace(
            go.Bar(
                x=algorithms,
                y=metrics,
                marker_color=colors,
                text=[f'{m:.3f}' for m in metrics],
                textposition='auto',
                name='Performance'
            ),
            row=1, col=1
        )
        
        # 2. Training Time
        colors = [cls.COLORS['success'] if t == min(train_times) else cls.COLORS['warning'] 
                  for t in train_times]
        
        fig.add_trace(
            go.Bar(
                x=algorithms,
                y=train_times,
                marker_color=colors,
                text=[f'{t:.3f}s' for t in train_times],
                textposition='auto',
                name='Time'
            ),
            row=1, col=2
        )
        
        # 3. Overfitting
        colors = [cls.COLORS['danger'] if o > 0.1 else cls.COLORS['success'] 
                  for o in overfitting]
        
        fig.add_trace(
            go.Bar(
                x=algorithms,
                y=overfitting,
                marker_color=colors,
                text=[f'{o:.3f}' for o in overfitting],
                textposition='auto',
                name='Overfitting'
            ),
            row=1, col=3
        )
        
        # Update layout
        fig.update_xaxes(tickangle=-45)
        fig.update_layout(
            title_text='Model Comparison Dashboard',
            showlegend=False,
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    @classmethod
    def plot_overfitting_analysis(cls, results: Dict[str, Any]) -> go.Figure:
        """Visualize bias-variance tradeoff and overfitting."""
        
        complexity_values = results['complexity_values']
        train_scores = results['train_scores']
        test_scores = results['test_scores']
        
        fig = go.Figure()
        
        # Training scores
        fig.add_trace(
            go.Scatter(
                x=complexity_values,
                y=train_scores,
                mode='lines+markers',
                name='Training Score',
                line=dict(color=cls.COLORS['train'], width=3),
                marker=dict(size=10)
            )
        )
        
        # Test scores
        fig.add_trace(
            go.Scatter(
                x=complexity_values,
                y=test_scores,
                mode='lines+markers',
                name='Test Score',
                line=dict(color=cls.COLORS['test'], width=3),
                marker=dict(size=10)
            )
        )
        
        # Highlight optimal complexity
        optimal = results['optimal_complexity']
        optimal_idx = optimal['index']
        
        fig.add_trace(
            go.Scatter(
                x=[complexity_values[optimal_idx]],
                y=[optimal['test_score']],
                mode='markers',
                name='Optimal Point',
                marker=dict(
                    size=20,
                    color=cls.COLORS['success'],
                    symbol='star',
                    line=dict(width=2, color='white')
                )
            )
        )
        
        # Add regions
        fig.add_vrect(
            x0=complexity_values[0],
            x1=complexity_values[max(0, optimal_idx - 1)],
            fillcolor=cls.COLORS['warning'],
            opacity=0.1,
            annotation_text='Underfitting',
            annotation_position='top left'
        )
        
        if optimal_idx < len(complexity_values) - 1:
            fig.add_vrect(
                x0=complexity_values[optimal_idx + 1],
                x1=complexity_values[-1],
                fillcolor=cls.COLORS['danger'],
                opacity=0.1,
                annotation_text='Overfitting',
                annotation_position='top right'
            )
        
        fig.update_layout(
            title=f'Bias-Variance Tradeoff: {results["complexity_param"]}',
            xaxis_title=results['complexity_param'],
            yaxis_title='Score',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    @classmethod
    def plot_what_if_scenario(cls, results: Dict[str, Any]) -> go.Figure:
        """Visualize what-if scenario analysis."""
        
        variations = results['variations']
        metrics = results['metrics']
        
        # Create labels from variations
        labels = []
        for var in variations:
            label_parts = [f'{k}={v}' for k, v in var.items()]
            labels.append('<br>'.join(label_parts))
        
        # Create gradient colors based on performance
        normalized_metrics = (np.array(metrics) - min(metrics)) / (max(metrics) - min(metrics) + 1e-10)
        colors = [f'rgb({int(255 * (1-m))}, {int(200 * m)}, {int(100 * m)})' 
                  for m in normalized_metrics]
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=labels,
                y=metrics,
                marker=dict(
                    color=colors,
                    line=dict(width=2, color='white')
                ),
                text=[f'{m:.3f}' for m in metrics],
                textposition='auto'
            )
        )
        
        # Add trend line
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=metrics,
                mode='lines+markers',
                line=dict(color=cls.COLORS['danger'], width=2, dash='dash'),
                marker=dict(size=10, color='white', line=dict(width=2, color=cls.COLORS['danger'])),
                name='Trend'
            )
        )
        
        fig.update_layout(
            title=f'What-If Analysis: {results["scenario"]}',
            xaxis_title='Scenario Variation',
            yaxis_title='Performance Metric',
            template='plotly_white',
            height=500,
            showlegend=False
        )
        
        return fig
    
    @classmethod
    def plot_time_series(cls, data: pd.DataFrame, analysis: Dict[str, Any]) -> go.Figure:
        """Create interactive time series visualization."""
        
        fig = go.Figure()
        
        # Original series
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['value'],
                mode='lines',
                name='Time Series',
                line=dict(color=cls.COLORS['primary'], width=2)
            )
        )
        
        # Add trend line if available
        if 'trend' in analysis:
            trend_line = analysis['trend'] * np.arange(len(data)) + data['value'].mean()
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=trend_line,
                    mode='lines',
                    name='Trend',
                    line=dict(color=cls.COLORS['danger'], width=2, dash='dash')
                )
            )
        
        # Add moving average
        window = min(30, len(data) // 10)
        moving_avg = data['value'].rolling(window=window).mean()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=moving_avg,
                mode='lines',
                name=f'MA({window})',
                line=dict(color=cls.COLORS['success'], width=2)
            )
        )
        
        fig.update_layout(
            title='Time Series Analysis',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    @classmethod
    def plot_parameter_impact_animation(
        cls,
        param_name: str,
        param_values: List[Any],
        scores: List[float],
        score_name: str = 'Accuracy'
    ) -> go.Figure:
        """Create animated visualization of parameter impact."""
        
        fig = go.Figure()
        
        # Create frames for animation
        frames = []
        for i in range(len(param_values)):
            frame_data = go.Bar(
                x=param_values[:i+1],
                y=scores[:i+1],
                marker=dict(
                    color=scores[:i+1],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=score_name)
                ),
                text=[f'{s:.3f}' for s in scores[:i+1]],
                textposition='auto'
            )
            frames.append(go.Frame(data=[frame_data], name=str(i)))
        
        # Initial frame
        fig.add_trace(
            go.Bar(
                x=[param_values[0]],
                y=[scores[0]],
                marker=dict(
                    color=[scores[0]],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=score_name)
                ),
                text=[f'{scores[0]:.3f}'],
                textposition='auto'
            )
        )
        
        fig.frames = frames
        
        # Add play button
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 500, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate'
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate'
                        }]
                    }
                ]
            }],
            title=f'Parameter Impact: {param_name} vs {score_name}',
            xaxis_title=param_name,
            yaxis_title=score_name,
            template='plotly_white',
            height=500
        )
        
        return fig
