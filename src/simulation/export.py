"""Export Simulation Results as Reports and Interactive Dashboards."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class SimulationExporter:
    """Export simulation results in various formats."""
    
    def __init__(self, output_dir: str = "simulation_exports"):
        """
        Initialize exporter.
        
        Args:
            output_dir: Directory to save exports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_to_html(
        self,
        result: Dict[str, Any],
        explanations: Optional[Dict[str, str]] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Export simulation results as interactive HTML report.
        
        Args:
            result: Simulation result dictionary
            explanations: Educational explanations
            filename: Output filename (auto-generated if None)
        
        Returns:
            Path to saved HTML file
        """
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_report_{timestamp}.html"
        
        filepath = self.output_dir / filename
        
        # Build HTML content
        html_content = self._generate_html_report(result, explanations)
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(filepath)
    
    def export_to_json(
        self,
        result: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        Export simulation results as JSON.
        
        Args:
            result: Simulation result dictionary
            filename: Output filename
        
        Returns:
            Path to saved JSON file
        """
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_data_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Prepare data (remove non-serializable objects)
        export_data = self._prepare_for_json(result)
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return str(filepath)
    
    def export_to_csv(
        self,
        results: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> str:
        """
        Export multiple simulation results as CSV.
        
        Args:
            results: List of simulation results
            filename: Output filename
        
        Returns:
            Path to saved CSV file
        """
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_comparison_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Convert to DataFrame
        df = self._results_to_dataframe(results)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        
        return str(filepath)
    
    def create_dashboard(
        self,
        results: Dict[str, Any],
        figures: List[go.Figure]
    ) -> str:
        """
        Create comprehensive interactive dashboard.
        
        Args:
            results: Simulation results
            figures: List of Plotly figures
        
        Returns:
            Path to saved dashboard HTML
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_dashboard_{timestamp}.html"
        filepath = self.output_dir / filename
        
        # Build dashboard HTML
        html = self._generate_dashboard_html(results, figures)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return str(filepath)
    
    def _generate_html_report(
        self,
        result: Dict[str, Any],
        explanations: Optional[Dict[str, str]]
    ) -> str:
        """Generate HTML report content."""
        
        model_results = result.get('model_results', {})
        params = result.get('params', {})
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Simulation Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 40px 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 36px;
            font-weight: 800;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 18px;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            font-size: 24px;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        .metric-label {{
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}
        
        .metric-value {{
            font-size: 32px;
            font-weight: 800;
            color: #667eea;
        }}
        
        .param-list {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }}
        
        .param-item {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .param-item:last-child {{
            border-bottom: none;
        }}
        
        .param-name {{
            font-weight: 600;
            color: #555;
        }}
        
        .param-value {{
            color: #667eea;
            font-weight: 500;
        }}
        
        .explanation {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        
        .explanation h3 {{
            color: #856404;
            margin-bottom: 10px;
        }}
        
        .explanation p {{
            color: #856404;
            line-height: 1.6;
        }}
        
        .timestamp {{
            text-align: center;
            color: #999;
            font-size: 14px;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid #e0e0e0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 ML Simulation Report</h1>
            <p>Interactive Machine Learning Simulation Results</p>
        </div>
        
        <div class="content">
            <div class="section">
                <h2 class="section-title">📊 Performance Metrics</h2>
                <div class="metric-grid">
"""
        
        # Add metrics
        if 'test_accuracy' in model_results:
            html += f"""
                    <div class="metric-card">
                        <div class="metric-label">Test Accuracy</div>
                        <div class="metric-value">{model_results['test_accuracy']:.2%}</div>
                    </div>
"""
        
        if 'f1_score' in model_results:
            html += f"""
                    <div class="metric-card">
                        <div class="metric-label">F1 Score</div>
                        <div class="metric-value">{model_results['f1_score']:.3f}</div>
                    </div>
"""
        
        if 'test_r2' in model_results:
            html += f"""
                    <div class="metric-card">
                        <div class="metric-label">R² Score</div>
                        <div class="metric-value">{model_results['test_r2']:.3f}</div>
                    </div>
"""
        
        if 'train_time' in model_results:
            html += f"""
                    <div class="metric-card">
                        <div class="metric-label">Training Time</div>
                        <div class="metric-value">{model_results['train_time']:.3f}s</div>
                    </div>
"""
        
        html += """
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">⚙️ Simulation Parameters</h2>
                <div class="param-list">
"""
        
        # Add parameters
        for key, value in params.items():
            if not key.startswith('_'):
                html += f"""
                    <div class="param-item">
                        <span class="param-name">{key.replace('_', ' ').title()}</span>
                        <span class="param-value">{value}</span>
                    </div>
"""
        
        html += """
                </div>
            </div>
"""
        
        # Add explanations
        if explanations:
            html += """
            <div class="section">
                <h2 class="section-title">💡 Insights & Explanations</h2>
"""
            for title, content in explanations.items():
                content_html = str(content).replace("\n", "<br>")
                html += f"""
                <div class="explanation">
                    <h3>{title.replace('_', ' ').title()}</h3>
                    <p>{content_html}</p>
                </div>
"""
            html += """
            </div>
"""
        
        html += f"""
            <div class="timestamp">
                Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def _generate_dashboard_html(
        self,
        results: Dict[str, Any],
        figures: List[go.Figure]
    ) -> str:
        """Generate dashboard HTML with embedded Plotly figures."""
        
        html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ML Simulation Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        .dashboard-header {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .dashboard-header h1 {
            font-size: 42px;
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .chart-container {
            margin-bottom: 40px;
        }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="dashboard-header">
            <h1>🎯 ML Simulation Dashboard</h1>
            <p>Interactive Analysis and Visualizations</p>
        </div>
"""
        
        # Add each figure
        for i, fig in enumerate(figures):
            html += f"""
        <div class="chart-container">
            <div id="chart-{i}"></div>
        </div>
"""
        
        html += """
    </div>
    
    <script>
"""
        
        # Add Plotly rendering code
        for i, fig in enumerate(figures):
            fig_json = fig.to_json()
            html += f"""
        Plotly.newPlot('chart-{i}', {fig_json});
"""
        
        html += """
    </script>
</body>
</html>
"""
        
        return html
    
    @staticmethod
    def _prepare_for_json(result: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare result dictionary for JSON serialization."""
        
        export_data = {}
        
        for key, value in result.items():
            if key == 'data':
                # Skip raw data arrays
                continue
            elif key == 'model_results':
                # Export model results but skip model object
                model_results = {}
                for k, v in value.items():
                    if k == 'model':
                        continue
                    elif isinstance(v, (np.ndarray, np.generic)):
                        model_results[k] = v.tolist()
                    else:
                        model_results[k] = v
                export_data[key] = model_results
            elif isinstance(value, (np.ndarray, np.generic)):
                export_data[key] = value.tolist()
            elif isinstance(value, dict):
                export_data[key] = value
            else:
                export_data[key] = value
        
        return export_data
    
    @staticmethod
    def _results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert simulation results to DataFrame."""
        
        rows = []
        
        for result in results:
            row = {}
            
            # Extract parameters
            if 'params' in result:
                for key, value in result['params'].items():
                    if not key.startswith('_'):
                        row[f'param_{key}'] = value
            
            # Extract metrics
            if 'model_results' in result:
                model_results = result['model_results']
                for key, value in model_results.items():
                    if key not in ['model', 'predictions', 'confusion_matrix', 'residuals']:
                        if isinstance(value, (int, float, np.number)):
                            row[f'metric_{key}'] = value
            
            # Extract data quality
            if 'data_quality' in result:
                for key, value in result['data_quality'].items():
                    if isinstance(value, (int, float)):
                        row[f'quality_{key}'] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)


class ReportGenerator:
    """Generate comprehensive text reports."""
    
    @staticmethod
    def generate_summary(result: Dict[str, Any]) -> str:
        """Generate text summary of simulation."""
        
        model_results = result.get('model_results', {})
        params = result.get('params', {})
        
        summary = "=" * 60 + "\n"
        summary += "ML SIMULATION SUMMARY\n"
        summary += "=" * 60 + "\n\n"
        
        summary += "ALGORITHM:\n"
        summary += f"  {model_results.get('algorithm', 'Unknown')}\n\n"
        
        summary += "PERFORMANCE METRICS:\n"
        if 'test_accuracy' in model_results:
            summary += f"  Test Accuracy: {model_results['test_accuracy']:.2%}\n"
            summary += f"  Train Accuracy: {model_results.get('train_accuracy', 0):.2%}\n"
        if 'test_r2' in model_results:
            summary += f"  Test R²: {model_results['test_r2']:.3f}\n"
            summary += f"  Train R²: {model_results.get('train_r2', 0):.3f}\n"
        if 'train_time' in model_results:
            summary += f"  Training Time: {model_results['train_time']:.3f}s\n"
        
        summary += "\nKEY PARAMETERS:\n"
        for key in ['n_samples', 'n_features', 'max_depth', 'n_estimators', 'noise_level']:
            if key in params:
                summary += f"  {key}: {params[key]}\n"
        
        summary += "\n" + "=" * 60 + "\n"
        
        return summary
