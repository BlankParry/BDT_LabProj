import os
import sys
import time
from datetime import datetime
import psutil

# Set environment variables for your actual setup
os.environ['JAVA_HOME'] = r'C:\JAVA\jdk-1.8'
os.environ['SPARK_HOME'] = r'C:\Spark\spark-3.3.4-bin-hadoop3'
os.environ['PYSPARK_PYTHON'] = r'C:\Users\kmamo\AppData\Local\Programs\Python\Python310\python.exe'

# Add src directory to path
sys.path.append('src')

def run_monitoring_pipeline():
    """Run the complete monitoring and anomaly detection pipeline"""
    
    print("=" * 70)
    print("GERMAN SMART METER MONITORING & ANOMALY DETECTION SYSTEM")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Focus: Comprehensive monitoring with multi-layer anomaly detection")
    print()
    
    # Import monitoring modules
    try:
        from data_processing.spark_processor import main as spark_main
        from mapreduce.mapreduce_processor import main as mapreduce_main
        from machine_learning.pytorch_anomaly import main as pytorch_main
        from monitoring.heat_monitor import SystemHeatMonitor
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure all updated monitoring modules are in the src/ directory")
        return
    
    # Initialize heat monitoring
    heat_monitor = SystemHeatMonitor()
    
    def execute_monitoring_pipeline():
        """Execute the complete monitoring pipeline"""
        results = {}
        
        print("Step 1: Spark Monitoring & Anomaly Processing")
        print("-" * 50)
        print("• Time-series anomaly detection with rolling windows")
        print("• Comprehensive monitoring analysis (trends, patterns)")
        print("• Monthly, weekly, and hourly aggregations")
        spark_result = spark_main()
        results['spark'] = spark_result is not None
        print(f"Spark processing: {'✅ Success' if results['spark'] else '❌ Failed'}")
        print()
        
        print("Step 2: MapReduce Monitoring Aggregations")
        print("-" * 50)
        print("• Temporal aggregations (hourly, daily, monthly)")
        print("• Peak demand analysis and efficiency metrics")
        print("• Appliance-level monitoring insights")
        mapreduce_result = mapreduce_main()
        results['mapreduce'] = mapreduce_result is not None
        print(f"MapReduce processing: {'✅ Success' if results['mapreduce'] else '❌ Failed'}")
        print()
        
        print("Step 3: PyTorch Time-Series Pattern Learning")
        print("-" * 50)
        print("• LSTM autoencoder for behavioral pattern learning")
        print("• Adaptive threshold anomaly detection")
        print("• GPU-accelerated sequence reconstruction")
        pytorch_result = pytorch_main()
        results['pytorch'] = pytorch_result is not None
        print(f"PyTorch processing: {'✅ Success' if results['pytorch'] else '❌ Failed'}")
        print()
        
        return results
    
    # Execute pipeline with heat monitoring
    print("🔥 Starting heat-monitored monitoring pipeline...")
    analysis, pipeline_results = heat_monitor.analyze_processing_heat(execute_monitoring_pipeline)
    
    # Display monitoring results
    print("\n" + "=" * 70)
    print("MONITORING & ANOMALY DETECTION RESULTS")
    print("=" * 70)
    
    # Load and display monitoring statistics
    try:
        import json
        with open("output/monitoring_statistics.json", "r") as f:
            monitoring_stats = json.load(f)
        
        print("Comprehensive Monitoring Results:")
        for household, stats in monitoring_stats.items():
            basic = stats.get('basic_stats', {})
            anomalies = stats.get('anomaly_stats', {})
            trends = stats.get('monitoring_trends', {})
            print(f"\n{household.upper()}:")
            print(f"  Avg Consumption: {basic.get('avg_consumption', 0):.2f}W")
            print(f"  Time-Series Anomalies: {anomalies.get('total_anomalies', 0)} ({anomalies.get('anomaly_rate', 0):.2f}%)")
            print(f"  Peak 95th Percentile: {trends.get('peak_demand', {}).get('peak_95th', 0):.2f}W")
            print(f"  Monthly Trends: {len(trends.get('monthly_trends', []))} months analyzed")
    except Exception as e:
        print(f"Could not load monitoring statistics: {e}")
    
    # Display PyTorch anomaly results
    try:
        with open("output/timeseries_anomaly_results.json", "r") as f:
            pytorch_results = json.load(f)
        
        print("\nPyTorch LSTM Autoencoder Results:")
        for household, stats in pytorch_results.items():
            print(f"\n{household.upper()}:")
            print(f"  Sequences Analyzed: {stats.get('num_sequences', 0)}")
            print(f"  Pattern-Based Anomalies: {stats.get('num_anomalies', 0)} ({stats.get('anomaly_percentage', 0):.2f}%)")
            print(f"  Reconstruction Threshold: {stats.get('threshold', 0):.6f}")
            print(f"  Avg Reconstruction Error: {stats.get('avg_reconstruction_error', 0):.6f}")
    except Exception as e:
        print(f"Could not load PyTorch results: {e}")
    
    # Display MapReduce monitoring insights
    try:
        with open("output/mapreduce_monitoring_analysis.json", "r") as f:
            mapreduce_analysis = json.load(f)
        
        print("\nMapReduce Monitoring Insights:")
        for household in ['residential3', 'residential4', 'residential6']:
            if household in mapreduce_analysis.get('consumption_trends', {}):
                trends = mapreduce_analysis['consumption_trends'][household]
                if trends:
                    latest_month = max(trends.keys())
                    print(f"\n{household.upper()}:")
                    print(f"  Latest Month Avg: {trends[latest_month]['avg_consumption']:.2f}W")
                    print(f"  Consumption Trend: {trends[latest_month]['trend']:.3f}")
                    print(f"  Volatility: {trends[latest_month]['volatility']:.3f}")
    except Exception as e:
        print(f"Could not load MapReduce analysis: {e}")
    
    # Display heat analysis results
    print("\n" + "=" * 70)
    print("HEAT ANALYSIS RESULTS")
    print("=" * 70)
    print(f"Processing Duration: {analysis['processing_duration']:.2f} seconds")
    print(f"Temperature Increase: {analysis['temperature_increase']:.2f}°C")
    print(f"Peak CPU Usage: {analysis['peak']['cpu_usage']:.2f}%")
    print(f"Peak Memory Usage: {analysis['peak']['memory_usage']:.2f}%")
    print(f"Thermal Efficiency: {analysis['thermal_efficiency']:.2f} sec/°C")
    
    # Generate visualizations
    heat_monitor.visualize_monitoring_data()
    
    # Generate comprehensive report
    report = heat_monitor.generate_heat_report(analysis)
    
    print("\n" + "=" * 70)
    print("SYSTEM RECOMMENDATIONS")
    print("=" * 70)
    for rec in report['recommendations']:
        print(f"• {rec}")
    
    # Final status
    print("\n" + "=" * 70)
    print("MONITORING PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    
    if pipeline_results:
        for component, status in pipeline_results.items():
            status_icon = "✅ Success" if status else "❌ Failed"
            print(f"{component.upper()}: {status_icon}")
    
    # Display system capabilities
    print("\n📊 MONITORING SYSTEM CAPABILITIES:")
    print("• Multi-layer anomaly detection (Statistical, Pattern-based, Local)")
    print("• Comprehensive consumption monitoring and trend analysis")
    print("• Real-time efficiency metrics and peak demand tracking")
    print("• Appliance-level usage insights and optimization recommendations")
    print("• Interactive dashboard with heatmaps and time-series visualizations")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Launch monitoring dashboard
    print("\n🚀 Launching comprehensive monitoring dashboard...")
    print("Dashboard features:")
    print("• Consumption heatmap (Hour vs Day of Week)")
    print("• Multi-layer anomaly analysis and consensus view")
    print("• Monthly trends and efficiency analysis")
    print("• Appliance usage timeline and peak demand tracking")
    print("• Real-time monitoring metrics and system status")
    print("\nDashboard will be available at: http://localhost:8050")
    print("Press Ctrl+C to stop the dashboard")
    
    try:
        from dashboard.dash_app import app
        app.run_server(debug=False, port=8050, host='127.0.0.1')
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        print("You can manually run: python src/dashboard/dash_app.py")

def validate_environment():
    """Validate environment setup before running pipeline"""
    print("Validating environment setup...")
    
    # Check Java
    java_home = os.environ.get('JAVA_HOME')
    if not java_home or not os.path.exists(java_home):
        print("❌ JAVA_HOME not set correctly")
        return False
    print(f"✅ Java: {java_home}")
    
    # Check Spark
    spark_home = os.environ.get('SPARK_HOME')
    if not spark_home or not os.path.exists(spark_home):
        print("❌ SPARK_HOME not set correctly")
        return False
    print(f"✅ Spark: {spark_home}")
    
    # Check Python
    python_path = os.environ.get('PYSPARK_PYTHON')
    if not python_path or not os.path.exists(python_path):
        print("❌ PYSPARK_PYTHON not set correctly")
        return False
    print(f"✅ Python: {python_path}")
    
    # Check data file
    if not os.path.exists("data/raw/german_smart_meter_data.csv"):
        print("❌ German smart meter data file not found")
        print("   Please ensure your dataset is at: data/raw/german_smart_meter_data.csv")
        return False
    print("✅ Data file found")
    
    # Check output directory
    if not os.path.exists("output"):
        os.makedirs("output")
        print("✅ Created output directory")
    else:
        print("✅ Output directory exists")
    
    return True

if __name__ == "__main__":
    print("German Smart Meter Monitoring & Anomaly Detection System")
    print("=" * 70)
    # Validate environment first
    if not validate_environment():
        print("\n❌ Environment validation failed. Please fix the issues above.")
        sys.exit(1)
    print("✅ Environment validation passed")
    print()
    from src.monitoring.heat_monitor import SystemHeatMonitor
    heat_monitor = SystemHeatMonitor()
    def pipeline_with_stats():
        from src.data_processing.spark_processor import main as spark_main
        spark_main()
        from src.mapreduce.mapreduce_processor import main as mapreduce_main
        mapreduce_main()
        from src.machine_learning.pytorch_anomaly import main as pytorch_main
        pytorch_main()
    # Run pipeline with heat monitoring
    analysis, _ = heat_monitor.analyze_processing_heat(pipeline_with_stats)
    print("\nSystem Heat Monitoring Results:")
    print(f"  Processing Duration: {analysis['processing_duration']:.2f} seconds")
    print(f"  Temperature Increase: {analysis['temperature_increase']:.2f}°C")
    print(f"  Peak CPU Usage: {analysis['peak']['cpu_usage']:.2f}%")
    print(f"  Peak Memory Usage: {analysis['peak']['memory_usage']:.2f}%")
    print(f"  Thermal Efficiency: {analysis['thermal_efficiency']:.2f} sec/°C")
    # Visualize and save system stats
    heat_monitor.visualize_monitoring_data(save_path="output/timeseries_system_monitoring.png")
    heat_monitor.generate_heat_report(analysis, save_path="output/timeseries_heat_analysis_report.json")
    print("System monitoring plot saved to output/timeseries_system_monitoring.png")
    print("System heat analysis report saved to output/timeseries_heat_analysis_report.json")
    print("\nAll processing complete. You can now launch the Streamlit dashboard:")
    print("  streamlit run streamlit_dashboard.py")
