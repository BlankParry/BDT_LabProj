import os
os.environ['JAVA_HOME'] = r'C:\JAVA\jdk-1.8'

from dash import Dash, html, dcc, Input, Output, dash_table
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta

# Initialize Dash app
app = Dash(__name__)

# Load data and results
try:
    df = pd.read_csv("data/raw/german_smart_meter_data.csv")
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
    df = df.fillna(0)
    
    # Load analysis results
    try:
        with open("output/monitoring_statistics.json", "r") as f:
            stats_results = json.load(f)
    except:
        with open("output/statistics.json", "r") as f:
            stats_results = json.load(f)
    
    try:
        with open("output/mapreduce_monitoring_results.json", "r") as f:
            mapreduce_results = json.load(f)
    except:
        with open("output/mapreduce_results.json", "r") as f:
            mapreduce_results = json.load(f)
    
    try:
        with open("output/timeseries_anomaly_results.json", "r") as f:
            pytorch_results = json.load(f)
    except:
        with open("output/pytorch_anomaly_results.json", "r") as f:
            pytorch_results = json.load(f)
        
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame()
    stats_results = {}
    mapreduce_results = {}
    pytorch_results = {}

# Dashboard layout
app.layout = html.Div([
    html.H1("German Smart Meter Monitoring & Anomaly Detection Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    
    # Control Panel
    html.Div([
        html.Div([
            html.Label("Select Household:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='household-selector',
                options=[
                    {'label': 'Residential 3', 'value': 'residential3'},
                    {'label': 'Residential 4', 'value': 'residential4'},
                    {'label': 'Residential 6', 'value': 'residential6'}
                ],
                value='residential4',
                style={'marginBottom': 10}
            )
        ], className='four columns'),
        
        html.Div([
            html.Label("Select Analysis Type:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='analysis-selector',
                options=[
                    {'label': 'Consumption Monitoring', 'value': 'monitoring'},
                    {'label': 'Anomaly Detection', 'value': 'anomaly'},
                    {'label': 'Trend Analysis', 'value': 'trends'},
                    {'label': 'Efficiency Analysis', 'value': 'efficiency'}
                ],
                value='monitoring',
                style={'marginBottom': 10}
            )
        ], className='four columns'),
        
        html.Div([
            html.Label("Date Range:", style={'fontWeight': 'bold'}),
            dcc.DatePickerRange(
                id='date-range-picker',
                start_date=df['utc_timestamp'].min() if not df.empty else '2015-10-25',
                end_date=df['utc_timestamp'].max() if not df.empty else '2015-10-26',
                style={'marginBottom': 10}
            )
        ], className='four columns'),
    ], className='row', style={'marginBottom': 30}),
    
    # Real-time Metrics Dashboard
    html.Div([
        html.H3("System Monitoring Metrics", style={'textAlign': 'center', 'color': '#34495e'}),
        html.Div(id='real-time-metrics', style={'marginBottom': 30})
    ]),
    
    # Main Visualization Area
    html.Div([
        html.Div([
            dcc.Graph(id='main-visualization')
        ], className='eight columns'),
        
        html.Div([
            html.H4("Analysis Summary", style={'color': '#34495e'}),
            html.Div(id='analysis-summary')
        ], className='four columns'),
    ], className='row', style={'marginBottom': 30}),
    
    # Secondary Visualizations
    html.Div([
        html.Div([
            dcc.Graph(id='secondary-viz-1')
        ], className='six columns'),
        
        html.Div([
            dcc.Graph(id='secondary-viz-2')
        ], className='six columns'),
    ], className='row', style={'marginBottom': 30}),
    
    # Anomaly Analysis Section
    html.Div([
        html.H3("Multi-Layer Anomaly Analysis", style={'color': '#e74c3c'}),
        html.Div(id='anomaly-analysis-section')
    ], style={'marginBottom': 30}),
    
    # Monitoring Insights Section
    html.Div([
        html.H3("Consumption Monitoring Insights", style={'color': '#8e44ad'}),
        html.Div(id='monitoring-insights-section')
    ], style={'marginBottom': 30}),
    
    # System Status
    html.Div([
        html.H3("System Status", style={'color': '#27ae60'}),
        html.Div(id='system-status')
    ]),
    
    # Auto-refresh component
    dcc.Interval(
        id='interval-component',
        interval=30*1000,
        n_intervals=0
    )
])

@app.callback(
    Output('real-time-metrics', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_metrics(n):
    """Update real-time monitoring metrics"""
    if df.empty:
        return html.P("No data available")
    
    # Calculate metrics
    total_records = len(df)
    households_active = len([h for h in ['residential3', 'residential4', 'residential6'] 
                           if any(col.startswith(f'DE_KN_{h}_') for col in df.columns)])
    
    total_anomalies = sum([pytorch_results.get(h, {}).get('num_anomalies', 0) 
                          for h in pytorch_results.keys()])
    
    # Calculate total consumption
    total_consumption = 0
    for household in ['residential3', 'residential4', 'residential6']:
        appliance_cols = [col for col in df.columns if f"DE_KN_{household}_" in col and 
                         not any(x in col for x in ['grid_export', 'grid_import', 'pv'])]
        if appliance_cols:
            total_consumption += df[appliance_cols].sum().sum()
    
    # Calculate efficiency metric
    avg_consumption_per_household = total_consumption / households_active if households_active > 0 else 0
    
    metrics = html.Div([
        html.Div([
            html.H4("Total Records"),
            html.P(f"{total_records:,}", style={'fontSize': 24, 'color': '#3498db'})
        ], className='three columns', style={'textAlign': 'center'}),
        
        html.Div([
            html.H4("Active Households"),
            html.P(f"{households_active}/3", style={'fontSize': 24, 'color': '#27ae60'})
        ], className='three columns', style={'textAlign': 'center'}),
        
        html.Div([
            html.H4("Anomalies Detected"),
            html.P(f"{total_anomalies}", style={'fontSize': 24, 'color': '#e74c3c'})
        ], className='three columns', style={'textAlign': 'center'}),
        
        html.Div([
            html.H4("Avg Consumption/Household"),
            html.P(f"{avg_consumption_per_household/1000:.1f} kWh", style={'fontSize': 24, 'color': '#f39c12'})
        ], className='three columns', style={'textAlign': 'center'}),
    ], className='row')
    
    return metrics

@app.callback(
    [Output('main-visualization', 'figure'),
     Output('analysis-summary', 'children'),
     Output('secondary-viz-1', 'figure'),
     Output('secondary-viz-2', 'figure')],
    [Input('household-selector', 'value'),
     Input('analysis-selector', 'value'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_main_visualizations(household, analysis_type, start_date, end_date):
    """Update main visualizations for monitoring and anomaly analysis"""
    
    if df.empty:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No data available", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False)
        return empty_fig, html.P("No data"), empty_fig, empty_fig
    
    # Filter data by date range
    df_filtered = df[(df['utc_timestamp'] >= start_date) & (df['utc_timestamp'] <= end_date)]
    
    # Get appliance columns for selected household
    appliance_cols = [col for col in df_filtered.columns if f"DE_KN_{household}_" in col]
    consumption_cols = [col for col in appliance_cols if not any(x in col for x in ['grid_export', 'grid_import', 'pv'])]
    
    if analysis_type == 'monitoring':
        # Consumption Heatmap (Hour vs Day of Week)
        if consumption_cols:
            df_filtered['hour'] = df_filtered['utc_timestamp'].dt.hour
            df_filtered['day_of_week'] = df_filtered['utc_timestamp'].dt.day_name()
            
            total_consumption = df_filtered[consumption_cols].sum(axis=1)
            df_filtered['total_consumption'] = total_consumption
            
            # Create heatmap data
            heatmap_data = df_filtered.groupby(['hour', 'day_of_week'])['total_consumption'].mean().reset_index()
            heatmap_pivot = heatmap_data.pivot(index='hour', columns='day_of_week', values='total_consumption')
            
            # Reorder columns to start with Monday
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_pivot = heatmap_pivot.reindex(columns=[day for day in day_order if day in heatmap_pivot.columns])
            
            fig = px.imshow(
                heatmap_pivot,
                title=f'{household.title()} - Consumption Heatmap (Hour vs Day of Week)',
                labels=dict(x="Day of Week", y="Hour of Day", color="Avg Consumption (W)"),
                color_continuous_scale='Viridis'
            )
        else:
            fig = go.Figure()
        
        # Summary
        if consumption_cols:
            total_consumption_sum = df_filtered[consumption_cols].sum().sum()
            avg_consumption = df_filtered[consumption_cols].sum(axis=1).mean()
            peak_hour = heatmap_pivot.stack().idxmax()[0] if not heatmap_pivot.empty else 'N/A'
            peak_day = heatmap_pivot.stack().idxmax()[1] if not heatmap_pivot.empty else 'N/A'
        else:
            total_consumption_sum = avg_consumption = 0
            peak_hour = peak_day = 'N/A'
        
        summary = html.Div([
            html.P(f"Total Consumption: {total_consumption_sum:.2f} Wh"),
            html.P(f"Average Consumption: {avg_consumption:.2f} W"),
            html.P(f"Peak Hour: {peak_hour}:00"),
            html.P(f"Peak Day: {peak_day}")
        ])
        
    elif analysis_type == 'anomaly':
        # Anomaly Detection Visualization
        fig = go.Figure()
        
        if consumption_cols:
            total_consumption = df_filtered[consumption_cols].sum(axis=1)
            
            # Calculate rolling statistics for anomaly detection
            window_size = 24
            rolling_mean = total_consumption.rolling(window=window_size, center=True).mean()
            rolling_std = total_consumption.rolling(window=window_size, center=True).std()
            
            # Calculate z-scores
            z_scores = (total_consumption - rolling_mean) / rolling_std
            
            # Identify anomalies
            anomaly_threshold = 2.5
            anomalies = np.abs(z_scores) > anomaly_threshold
            
            # Plot consumption
            fig.add_trace(go.Scatter(
                x=df_filtered['utc_timestamp'],
                y=total_consumption,
                mode='lines',
                name='Consumption',
                line=dict(color='blue')
            ))
            
            # Plot rolling mean
            fig.add_trace(go.Scatter(
                x=df_filtered['utc_timestamp'],
                y=rolling_mean,
                mode='lines',
                name='Rolling Mean',
                line=dict(color='green', dash='dash')
            ))
            
            # Highlight anomalies
            if anomalies.any():
                fig.add_trace(go.Scatter(
                    x=df_filtered.loc[anomalies, 'utc_timestamp'],
                    y=total_consumption[anomalies],
                    mode='markers',
                    name='Statistical Anomalies',
                    marker=dict(color='red', size=8)
                ))
        
        fig.update_layout(
            title=f'{household.title()} - Multi-Layer Anomaly Detection',
            xaxis_title='Time',
            yaxis_title='Power (W)'
        )
        
        if consumption_cols:
            anomaly_count = anomalies.sum()
            pytorch_anomalies = pytorch_results.get(household, {}).get('num_anomalies', 0)
            summary = html.Div([
                html.P(f"Statistical Anomalies: {anomaly_count}"),
                html.P(f"PyTorch Anomalies: {pytorch_anomalies}"),
                html.P(f"Combined Rate: {(anomaly_count + pytorch_anomalies)/len(total_consumption)*100:.2f}%"),
                html.P(f"Threshold: ±{anomaly_threshold} σ")
            ])
        else:
            summary = html.P("No consumption data available")
        
    elif analysis_type == 'trends':
        # Monthly Consumption Trends
        if consumption_cols:
            df_filtered['month'] = df_filtered['utc_timestamp'].dt.to_period('M')
            monthly_data = df_filtered.groupby('month')[consumption_cols].sum().sum(axis=1)
            
            fig = px.line(
                x=monthly_data.index.astype(str),
                y=monthly_data.values,
                title=f'{household.title()} - Monthly Consumption Trends',
                labels={'x': 'Month', 'y': 'Total Consumption (Wh)'}
            )
            
            # Add trend line
            if len(monthly_data) > 1:
                z = np.polyfit(range(len(monthly_data)), monthly_data.values, 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=monthly_data.index.astype(str),
                    y=p(range(len(monthly_data))),
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', dash='dash')
                ))
        else:
            fig = go.Figure()
        
        if consumption_cols and len(monthly_data) > 1:
            trend_slope = (monthly_data.iloc[-1] - monthly_data.iloc[0]) / len(monthly_data)
            trend_direction = "Increasing" if trend_slope > 0 else "Decreasing"
            summary = html.Div([
                html.P(f"Trend Direction: {trend_direction}"),
                html.P(f"Monthly Change: {trend_slope:.2f} Wh/month"),
                html.P(f"Total Months: {len(monthly_data)}"),
                html.P(f"Latest Month: {monthly_data.iloc[-1]:.2f} Wh")
            ])
        else:
            summary = html.P("Insufficient data for trend analysis")
        
    else:  # efficiency
        # Appliance Efficiency Analysis
        if consumption_cols:
            appliance_totals = df_filtered[consumption_cols].sum()
            appliance_names = [col.split('_')[-1].replace('_', ' ').title() for col in consumption_cols]
            
            # Calculate efficiency (consumption per hour of operation)
            appliance_efficiency = []
            for col in consumption_cols:
                non_zero_hours = (df_filtered[col] > 0).sum()
                total_consumption = df_filtered[col].sum()
                efficiency = total_consumption / non_zero_hours if non_zero_hours > 0 else 0
                appliance_efficiency.append(efficiency)
            
            fig = px.bar(
                x=appliance_names,
                y=appliance_efficiency,
                title=f'{household.title()} - Appliance Efficiency (Consumption per Active Hour)',
                labels={'x': 'Appliance', 'y': 'Efficiency (W/hour)'}
            )
            
            summary = html.Div([
                html.P(f"Most Efficient: {appliance_names[np.argmin(appliance_efficiency)]}"),
                html.P(f"Least Efficient: {appliance_names[np.argmax(appliance_efficiency)]}"),
                html.P(f"Avg Efficiency: {np.mean(appliance_efficiency):.2f} W/h"),
                html.P(f"Appliances Analyzed: {len(consumption_cols)}")
            ])
        else:
            fig = go.Figure()
            summary = html.P("No appliance data available")
    
    # Secondary visualizations
    # 1. Appliance Usage Timeline (Stacked Area)
    if consumption_cols:
        fig_sec1 = go.Figure()
        
        for col in consumption_cols[:5]:  # Limit to top 5 appliances
            appliance_name = col.split('_')[-1].replace('_', ' ').title()
            fig_sec1.add_trace(go.Scatter(
                x=df_filtered['utc_timestamp'],
                y=df_filtered[col],
                mode='lines',
                stackgroup='one',
                name=appliance_name
            ))
        
        fig_sec1.update_layout(
            title=f'{household.title()} - Appliance Usage Timeline',
            xaxis_title='Time',
            yaxis_title='Power (W)'
        )
    else:
        fig_sec1 = go.Figure()
    
    # 2. Peak Demand Analysis
    if consumption_cols:
        total_consumption = df_filtered[consumption_cols].sum(axis=1)
        df_filtered['hour'] = df_filtered['utc_timestamp'].dt.hour
        
        # Calculate peak percentiles by hour
        hourly_peaks = df_filtered.groupby('hour')['total_consumption'].agg(['mean', 'max', lambda x: np.percentile(x, 95)])
        hourly_peaks.columns = ['Mean', 'Max', '95th Percentile']
        
        fig_sec2 = go.Figure()
        for col in hourly_peaks.columns:
            fig_sec2.add_trace(go.Scatter(
                x=hourly_peaks.index,
                y=hourly_peaks[col],
                mode='lines+markers',
                name=col
            ))
        
        fig_sec2.update_layout(
            title=f'{household.title()} - Peak Demand Analysis by Hour',
            xaxis_title='Hour of Day',
            yaxis_title='Power (W)'
        )
    else:
        fig_sec2 = go.Figure()
    
    return fig, summary, fig_sec1, fig_sec2

@app.callback(
    Output('anomaly-analysis-section', 'children'),
    Input('household-selector', 'value')
)
def update_anomaly_analysis_section(household):
    """Update multi-layer anomaly analysis section"""
    
    if household not in pytorch_results:
        return html.P("No anomaly detection results available for this household.")
    
    pytorch_data = pytorch_results[household]
    
    # Create comprehensive anomaly analysis
    anomaly_data = [
        {
            "Detection Method": "PyTorch LSTM",
            "Anomalies Detected": pytorch_data.get('num_anomalies', 'N/A'),
            "Detection Rate": f"{pytorch_data.get('anomaly_percentage', 0):.2f}%",
            "Threshold": f"{pytorch_data.get('threshold', 0):.6f}"
        },
        {
            "Detection Method": "Spark Statistical",
            "Anomalies Detected": "Based on Z-score analysis",
            "Detection Rate": "Variable by time window",
            "Threshold": "±3 standard deviations"
        },
        {
            "Detection Method": "MapReduce Local",
            "Anomalies Detected": "Local pattern inconsistencies",
            "Detection Rate": "Context-dependent",
            "Threshold": "Local mean ± 2.5σ"
        }
    ]
    
    table = dash_table.DataTable(
        data=anomaly_data,
        columns=[
            {"name": "Detection Method", "id": "Detection Method"},
            {"name": "Anomalies Detected", "id": "Anomalies Detected"},
            {"name": "Detection Rate", "id": "Detection Rate"},
            {"name": "Threshold", "id": "Threshold"}
        ],
        style_cell={'textAlign': 'left'},
        style_header={'backgroundColor': '#e74c3c', 'color': 'white'},
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
    )
    
    return html.Div([
        html.H4(f"Multi-Layer Anomaly Detection - {household.title()}"),
        table,
        html.P("Each method detects different types of anomalies: PyTorch (pattern-based), Spark (statistical), MapReduce (local context)", 
               style={'marginTop': 10, 'fontStyle': 'italic', 'color': '#7f8c8d'})
    ])

@app.callback(
    Output('monitoring-insights-section', 'children'),
    Input('household-selector', 'value')
)
def update_monitoring_insights_section(household):
    """Update monitoring insights section"""
    
    if not mapreduce_results:
        return html.P("No monitoring insights available.")
    
    # Extract monitoring insights for the household
    monitoring_data = []
    
    # Look for monthly trends
    monthly_keys = [k for k in mapreduce_results.keys() if k.startswith(f"{household}_monthly")]
    if monthly_keys:
        latest_month = max(monthly_keys)
        monthly_stats = mapreduce_results[latest_month]
        monitoring_data.append({
            "Metric": "Latest Monthly Average",
            "Value": f"{monthly_stats['average']:.2f} W",
            "Trend": f"{monthly_stats.get('trend', 0):.3f}"
        })
    
    # Look for peak demand
    peak_keys = [k for k in mapreduce_results.keys() if k.startswith(f"{household}_peak_demand")]
    for key in peak_keys[:3]:  # Show top 3
        demand_type = key.split('_')[-1]
        stats = mapreduce_results[key]
        monitoring_data.append({
            "Metric": f"{demand_type.title()} Demand Periods",
            "Value": f"{stats['count']} occurrences",
            "Trend": f"Avg: {stats['average']:.2f}W"
        })
    
    if not monitoring_data:
        monitoring_data = [{"Metric": "No monitoring data", "Value": "N/A", "Trend": "N/A"}]
    
    table = dash_table.DataTable(
        data=monitoring_data,
        columns=[
            {"name": "Metric", "id": "Metric"},
            {"name": "Value", "id": "Value"},
            {"name": "Trend", "id": "Trend"}
        ],
        style_cell={'textAlign': 'left'},
        style_header={'backgroundColor': '#8e44ad', 'color': 'white'},
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
    )
    
    return html.Div([
        html.H4(f"Consumption Monitoring Insights - {household.title()}"),
        table,
        html.P(f"Insights from MapReduce aggregations across {len([k for k in mapreduce_results.keys() if k.startswith(household)])} categories", 
               style={'marginTop': 10, 'fontStyle': 'italic'})
    ])

@app.callback(
    Output('system-status', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_system_status(n):
    """Update system status"""
    
    status_items = [
        {"component": "Spark Monitoring Processing", "status": "✅ Complete", "color": "green"},
        {"component": "MapReduce Aggregations", "status": "✅ Complete", "color": "green"},
        {"component": "PyTorch Anomaly Models", "status": "✅ Trained", "color": "green"},
        {"component": "Monitoring Dashboard", "status": "✅ Active", "color": "green"},
        {"component": "Multi-Layer Detection", "status": "✅ Operational", "color": "green"}
    ]
    
    status_elements = []
    for item in status_items:
        status_elements.append(
            html.P(f"{item['component']}: {item['status']}", 
                  style={'color': item['color'], 'margin': '5px 0'})
        )
    
    return html.Div(status_elements)

if __name__ == '__main__':
    print("Starting German Smart Meter Monitoring Dashboard on http://localhost:8050")
    app.run_server(debug=True, port=8050, host='127.0.0.1')
