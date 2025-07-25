import os
os.environ['JAVA_HOME'] = r'C:\JAVA\jdk-1.8'
os.environ['SPARK_HOME'] = r'C:\Spark\spark-3.3.4-bin-hadoop3'
os.environ['PYSPARK_PYTHON'] = r'C:\Users\kmamo\AppData\Local\Programs\Python\Python310\python.exe'

import findspark
findspark.init(r'C:\Spark\spark-3.3.4-bin-hadoop3')

from functools import reduce
from operator import add
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import pandas as pd

def create_spark_session():
    """Initialize Spark session with optimized configuration"""
    spark = SparkSession.builder \
        .appName("GermanSmartMeterMonitoring") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    
    return spark

def load_and_preprocess_data(spark, file_path):
    """Load and preprocess German smart meter data for monitoring"""
    print("Loading German smart meter data for monitoring analysis...")
    
    # Load data
    df = spark.read.option("header", "true").csv(file_path)
    
    # Convert timestamp (only use UTC timestamp)
    df = df.withColumn("timestamp", to_timestamp(col("utc_timestamp"), "yyyy-MM-dd'T'HH:mm:ss'Z'"))
    df = df.withColumn("row_id", monotonically_increasing_id())
    
    # Fill null values with 0 for all appliance columns
    appliance_columns = [col_name for col_name in df.columns if col_name.startswith("DE_KN_")]
    for col_name in appliance_columns:
        df = df.withColumn(col_name, when(col(col_name).isNull(), 0.0).otherwise(col(col_name).cast("double")))
    
    # Add comprehensive time-based features for monitoring
    df = df.withColumn("hour", hour(col("timestamp"))) \
           .withColumn("day_of_week", dayofweek(col("timestamp"))) \
           .withColumn("date", to_date(col("timestamp"))) \
           .withColumn("month", month(col("timestamp"))) \
           .withColumn("year", year(col("timestamp"))) \
           .withColumn("week_of_year", weekofyear(col("timestamp"))) \
           .withColumn("is_weekend", when(dayofweek(col("timestamp")).isin([1, 7]), 1).otherwise(0))
    
    # Calculate total consumption per household (excluding grid and PV)
    residential3_cols = [col_name for col_name in appliance_columns if "residential3" in col_name and 
                        not any(x in col_name for x in ['grid_export', 'grid_import', 'pv'])]
    residential4_cols = [col_name for col_name in appliance_columns if "residential4" in col_name and 
                        not any(x in col_name for x in ['grid_export', 'grid_import', 'pv'])]
    residential6_cols = [col_name for col_name in appliance_columns if "residential6" in col_name and 
                        not any(x in col_name for x in ['grid_export', 'grid_import', 'pv'])]
    
    # Use reduce with add operator for column summation
    if residential3_cols:
        df = df.withColumn("residential3_total", reduce(add, [col(c) for c in residential3_cols]))
    if residential4_cols:
        df = df.withColumn("residential4_total", reduce(add, [col(c) for c in residential4_cols]))
    if residential6_cols:
        df = df.withColumn("residential6_total", reduce(add, [col(c) for c in residential6_cols]))
    
    return df

def detect_timeseries_anomalies(df, window_size=5):
    """Detect anomalies based on inconsistency with time-series range before and after"""
    print(f"Detecting time-series anomalies using window size of {window_size}...")
    
    households = ['residential3', 'residential4', 'residential6']
    
    for household in households:
        total_col = f"{household}_total"
        if total_col not in df.columns:
            continue
        
        # FIXED: Add proper partitioning to avoid single partition warning
        window_spec = Window.partitionBy("date").orderBy("timestamp").rowsBetween(-window_size, window_size)
        
        # Calculate rolling statistics over the window
        df = df.withColumn(f"{household}_rolling_mean", 
                          avg(col(total_col)).over(window_spec))
        df = df.withColumn(f"{household}_rolling_std", 
                          stddev(col(total_col)).over(window_spec))
        
        # For percentile operations, use a different approach to avoid window issues
        # Calculate median using approx_percentile over groups instead of windows
        daily_stats = df.groupBy("date") \
                       .agg(expr(f"percentile_approx({total_col}, 0.5)").alias(f"{household}_daily_median"))
        
        df = df.join(daily_stats, "date", "left")
        
        # Calculate z-score for current value against surrounding values
        df = df.withColumn(f"{household}_z_score", 
                          (col(total_col) - col(f"{household}_rolling_mean")) / 
                          when(col(f"{household}_rolling_std") > 0, col(f"{household}_rolling_std")).otherwise(1))
        
        # Calculate deviation from median (more robust to outliers)
        df = df.withColumn(f"{household}_median_deviation", 
                          abs(col(total_col) - col(f"{household}_daily_median")))
        
        # Flag as anomaly using multiple criteria
        df = df.withColumn(f"{household}_is_anomaly", 
                          when((abs(col(f"{household}_z_score")) > 3) |  # Standard z-score
                               (col(f"{household}_median_deviation") > col(f"{household}_rolling_mean") * 1.5), 1)
                          .otherwise(0))
        
        # Clean up intermediate columns to avoid duplication
        df = df.drop(f"{household}_rolling_mean", f"{household}_rolling_std", f"{household}_daily_median")
    
    return df

def perform_monitoring_analysis(df):
    """Perform comprehensive monitoring analysis"""
    print("Performing comprehensive monitoring analysis...")
    
    households = ['residential3', 'residential4', 'residential6']
    results = {}
    
    for household in households:
        total_col = f"{household}_total"
        if total_col not in df.columns:
            continue
            
        # Basic statistics
        basic_stats = df.select(
            avg(total_col).alias("avg_consumption"),
            stddev(total_col).alias("std_consumption"),
            max(total_col).alias("max_consumption"),
            min(total_col).alias("min_consumption"),
            count(total_col).alias("total_readings")
        ).collect()[0]
        
        # Time-series anomaly statistics
        anomaly_stats = df.select(
            sum(f"{household}_is_anomaly").alias("total_anomalies"),
            avg(f"{household}_z_score").alias("avg_z_score"),
            max(f"{household}_z_score").alias("max_z_score"),
            min(f"{household}_z_score").alias("min_z_score")
        ).collect()[0]
        
        # Monitoring trends - Monthly consumption
        monthly_trends = df.groupBy("year", "month") \
                          .agg(avg(total_col).alias("monthly_avg"),
                               sum(total_col).alias("monthly_total"),
                               count(total_col).alias("monthly_readings")) \
                          .orderBy("year", "month") \
                          .collect()
        
        # Hourly patterns for monitoring
        hourly_patterns = df.groupBy("hour") \
                           .agg(avg(total_col).alias("hourly_avg"),
                                count(total_col).alias("hourly_count"),
                                sum(f"{household}_is_anomaly").alias("hourly_anomalies")) \
                           .orderBy("hour") \
                           .collect()
        
        # Weekly patterns for monitoring
        weekly_patterns = df.groupBy("day_of_week") \
                           .agg(avg(total_col).alias("weekly_avg"),
                                count(total_col).alias("weekly_count")) \
                           .orderBy("day_of_week") \
                           .collect()
        
        # Peak demand analysis
        peak_stats = df.select(
            expr(f"percentile_approx({total_col}, 0.95)").alias("peak_95th"),
            expr(f"percentile_approx({total_col}, 0.99)").alias("peak_99th")
        ).collect()[0]
        
        results[household] = {
            'basic_stats': {
                'avg_consumption': float(basic_stats['avg_consumption']),
                'std_consumption': float(basic_stats['std_consumption']),
                'max_consumption': float(basic_stats['max_consumption']),
                'min_consumption': float(basic_stats['min_consumption']),
                'total_readings': int(basic_stats['total_readings'])
            },
            'anomaly_stats': {
                'total_anomalies': int(anomaly_stats['total_anomalies']),
                'anomaly_rate': float(anomaly_stats['total_anomalies']) / float(basic_stats['total_readings']) * 100,
                'avg_z_score': float(anomaly_stats['avg_z_score']),
                'max_z_score': float(anomaly_stats['max_z_score']),
                'min_z_score': float(anomaly_stats['min_z_score'])
            },
            'monitoring_trends': {
                'monthly_trends': [
                    {'year': int(row['year']), 'month': int(row['month']),
                     'avg_consumption': float(row['monthly_avg']),
                     'total_consumption': float(row['monthly_total']),
                     'readings': int(row['monthly_readings'])}
                    for row in monthly_trends
                ],
                'hourly_patterns': [
                    {'hour': int(row['hour']), 
                     'avg_consumption': float(row['hourly_avg']),
                     'count': int(row['hourly_count']),
                     'anomalies': int(row['hourly_anomalies'])}
                    for row in hourly_patterns
                ],
                'weekly_patterns': [
                    {'day_of_week': int(row['day_of_week']),
                     'avg_consumption': float(row['weekly_avg']),
                     'count': int(row['weekly_count'])}
                    for row in weekly_patterns
                ],
                'peak_demand': {
                    'peak_95th': float(peak_stats['peak_95th']),
                    'peak_99th': float(peak_stats['peak_99th'])
                }
            }
        }
    
    return results

def main():
    """Main processing function with monitoring and anomaly detection"""
    spark = create_spark_session()
    try:
        # Load and preprocess data
        df = load_and_preprocess_data(spark, "data/raw/german_smart_meter_data.csv")
        # Save processed data as parquet for Streamlit
        df_pd = df.toPandas()
        df_pd.to_parquet("data/processed/processed_data.parquet", index=False)
        print("Processed data saved to data/processed/processed_data.parquet")
    except Exception as e:
        print(f"Error in Spark processing: {e}")
    finally:
        spark.stop()
    return

if __name__ == "__main__":
    processed_df = main()
