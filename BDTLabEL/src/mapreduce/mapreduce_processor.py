from collections import defaultdict
import multiprocessing as mp
import pandas as pd
import json
from datetime import datetime
import numpy as np

def map_monitoring_smart_meter_data(data_chunk):
    """Enhanced map phase for monitoring and aggregation analysis"""
    mapped_results = []
    
    # Sort data by timestamp for proper time-series analysis
    data_chunk = data_chunk.sort_values('utc_timestamp').reset_index(drop=True)
    
    for idx, record in data_chunk.iterrows():
        try:
            timestamp = pd.to_datetime(record['utc_timestamp'])
            hour = timestamp.hour
            date = timestamp.date()
            month = timestamp.month
            year = timestamp.year
            day_of_week = timestamp.dayofweek
            week_of_year = timestamp.isocalendar()[1]
            
            # Map for each household
            households = ['residential3', 'residential4', 'residential6']
            
            for household in households:
                # Get appliance columns for this household
                appliance_cols = [col for col in record.index if f"DE_KN_{household}_" in col]
                consumption_cols = [col for col in appliance_cols if not any(x in col for x in ['grid_export', 'grid_import', 'pv'])]
                
                total_consumption = 0
                appliance_consumptions = {}
                
                for col_name in consumption_cols:
                    value = record[col_name] if pd.notna(record[col_name]) else 0
                    total_consumption += value
                    appliance = col_name.split('_')[-1]
                    appliance_consumptions[appliance] = value
                
                # MONITORING AGGREGATIONS
                
                # 1. Temporal aggregations for trend monitoring
                mapped_results.append(((household, 'hourly', hour), total_consumption))
                mapped_results.append(((household, 'daily', str(date)), total_consumption))
                mapped_results.append(((household, 'weekly', f"{year}_W{week_of_year}"), total_consumption))
                mapped_results.append(((household, 'monthly', f"{year}_{month:02d}"), total_consumption))
                mapped_results.append(((household, 'yearly', year), total_consumption))
                
                # 2. Day of week patterns for monitoring
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                mapped_results.append(((household, 'day_pattern', day_names[day_of_week]), total_consumption))
                
                # 3. Peak demand analysis
                if total_consumption > 1500:  # High consumption threshold
                    mapped_results.append(((household, 'peak_demand', 'high'), total_consumption))
                elif total_consumption < 200:  # Low consumption threshold
                    mapped_results.append(((household, 'peak_demand', 'low'), total_consumption))
                else:
                    mapped_results.append(((household, 'peak_demand', 'normal'), total_consumption))
                
                # 4. Appliance-level monitoring aggregations
                for appliance, consumption in appliance_consumptions.items():
                    mapped_results.append(((household, 'appliance_hourly', f"{appliance}_{hour}"), consumption))
                    mapped_results.append(((household, 'appliance_daily', f"{appliance}_{date}"), consumption))
                    mapped_results.append(((household, 'appliance_total', appliance), consumption))
                
                # 5. Efficiency metrics (consumption per hour)
                mapped_results.append(((household, 'efficiency', 'consumption_per_hour'), total_consumption))
                
                # 6. Time-series context analysis for anomaly detection
                window_size = 3
                start_idx = max(0, idx - window_size)
                end_idx = min(len(data_chunk), idx + window_size + 1)
                
                surrounding_values = []
                for i in range(start_idx, end_idx):
                    if i != idx:  # Exclude current value
                        surrounding_total = 0
                        for col_name in consumption_cols:
                            val = data_chunk.iloc[i][col_name] if pd.notna(data_chunk.iloc[i][col_name]) else 0
                            surrounding_total += val
                        surrounding_values.append(surrounding_total)
                
                # Local anomaly detection mapping
                if surrounding_values:
                    local_mean = np.mean(surrounding_values)
                    local_std = np.std(surrounding_values)
                    
                    if local_std > 0:
                        z_score = (total_consumption - local_mean) / local_std
                        mapped_results.append(((household, 'anomaly_score', 'z_score'), z_score))
                        
                        if abs(z_score) > 2.5:
                            mapped_results.append(((household, 'anomaly_detection', 'local_anomaly'), total_consumption))
                
        except Exception as e:
            print(f"Error processing record at index {idx}: {e}")
            continue
    
    return mapped_results

def reduce_monitoring_smart_meter_data(mapped_data):
    """Enhanced reduce phase for monitoring aggregations"""
    reduced_results = defaultdict(list)
    
    # Group by keys
    for key, value in mapped_data:
        reduced_results[key].append(value)
    
    # Calculate comprehensive aggregations
    final_results = {}
    for key, values in reduced_results.items():
        final_results[key] = {
            'total': sum(values),
            'average': sum(values) / len(values),
            'max': max(values),
            'min': min(values),
            'count': len(values),
            'std_deviation': calculate_std(values),
            'median': calculate_median(values)
        }
        
        # Add additional metrics for monitoring
        if len(values) > 1:
            final_results[key]['trend'] = calculate_trend(values)
            final_results[key]['volatility'] = calculate_volatility(values)
    
    return final_results

def calculate_std(values):
    """Calculate standard deviation"""
    if len(values) < 2:
        return 0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5

def calculate_median(values):
    """Calculate median"""
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n % 2 == 0:
        return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
    else:
        return sorted_values[n//2]

def calculate_trend(values):
    """Calculate simple trend (positive/negative/stable)"""
    if len(values) < 2:
        return 0
    
    first_half = values[:len(values)//2]
    second_half = values[len(values)//2:]
    
    first_avg = sum(first_half) / len(first_half)
    second_avg = sum(second_half) / len(second_half)
    
    return (second_avg - first_avg) / first_avg if first_avg > 0 else 0

def calculate_volatility(values):
    """Calculate volatility as coefficient of variation"""
    if len(values) < 2:
        return 0
    
    mean_val = sum(values) / len(values)
    std_val = calculate_std(values)
    
    return std_val / mean_val if mean_val > 0 else 0

def fix_mapreduce_keys(mapped_data):
    """Convert tuple keys to string keys for JSON serialization"""
    fixed_results = {}
    for key, value in mapped_data.items():
        if isinstance(key, tuple):
            new_key = '_'.join(str(k) for k in key)
        else:
            new_key = str(key)
        fixed_results[new_key] = value
    return fixed_results

def parallel_mapreduce(data, num_processes=4):
    """Execute monitoring MapReduce in parallel"""
    print(f"Starting monitoring MapReduce with {num_processes} processes...")
    
    # Split data into chunks
    chunk_size = len(data) // num_processes
    data_chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Parallel map phase
    with mp.Pool(num_processes) as pool:
        mapped_results = pool.map(map_monitoring_smart_meter_data, data_chunks)
    
    # Flatten mapped results
    all_mapped = [item for sublist in mapped_results for item in sublist]
    
    # Reduce phase
    final_results = reduce_monitoring_smart_meter_data(all_mapped)
    
    return final_results

def analyze_monitoring_results(results):
    """Analyze MapReduce results for monitoring insights"""
    analysis = {
        'consumption_trends': {},
        'peak_demand_analysis': {},
        'appliance_insights': {},
        'anomaly_summary': {}
    }
    
    for key, stats in results.items():
        key_parts = key.split('_')
        household = key_parts[0]
        
        if household not in ['residential3', 'residential4', 'residential6']:
            continue
        
        # Consumption trends analysis
        if 'monthly' in key:
            if household not in analysis['consumption_trends']:
                analysis['consumption_trends'][household] = {}
            month = '_'.join(key_parts[2:])
            analysis['consumption_trends'][household][month] = {
                'avg_consumption': stats['average'],
                'total_consumption': stats['total'],
                'trend': stats.get('trend', 0),
                'volatility': stats.get('volatility', 0)
            }
        
        # Peak demand analysis
        if 'peak_demand' in key:
            if household not in analysis['peak_demand_analysis']:
                analysis['peak_demand_analysis'][household] = {}
            demand_type = key_parts[2]
            analysis['peak_demand_analysis'][household][demand_type] = {
                'count': stats['count'],
                'avg_consumption': stats['average'],
                'max_consumption': stats['max']
            }
        
        # Appliance insights
        if 'appliance_total' in key:
            if household not in analysis['appliance_insights']:
                analysis['appliance_insights'][household] = {}
            appliance = key_parts[2]
            analysis['appliance_insights'][household][appliance] = {
                'total_consumption': stats['total'],
                'avg_consumption': stats['average'],
                'usage_frequency': stats['count']
            }
        
        # Anomaly summary
        if 'anomaly_detection' in key:
            if household not in analysis['anomaly_summary']:
                analysis['anomaly_summary'][household] = {}
            anomaly_type = key_parts[2]
            analysis['anomaly_summary'][household][anomaly_type] = {
                'count': stats['count'],
                'avg_value': stats['average']
            }
    
    return analysis

def main():
    """Main monitoring MapReduce execution"""
    print("Loading data for monitoring MapReduce processing...")
    try:
        # Load data
        df = pd.read_csv("data/raw/german_smart_meter_data.csv")
        df = df.fillna(0)  # Fill null values with 0
        print(f"Loaded {len(df)} records for monitoring MapReduce processing")
        # Execute MapReduce
        results = parallel_mapreduce(df, num_processes=4)
        # Fix keys for JSON serialization BEFORE analysis
        results = fix_mapreduce_keys(results)
        # Analyze results for monitoring insights
        analysis = analyze_monitoring_results(results)
        # Save results
        with open("output/mapreduce_monitoring_results.json", "w") as f:
            json.dump(results, f, indent=2)
        # Save monitoring analysis
        with open("output/mapreduce_monitoring_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        # Also save as mapreduce_results.json for compatibility
        with open("output/mapreduce_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Monitoring MapReduce completed! Generated {len(results)} aggregated results")
        print("Results saved to output/mapreduce_monitoring_results.json")
        return results, analysis
    except Exception as e:
        print(f"Error in monitoring MapReduce processing: {e}")
        # Always write empty files if error
        with open("output/mapreduce_monitoring_results.json", "w") as f:
            json.dump({}, f)
        with open("output/mapreduce_monitoring_analysis.json", "w") as f:
            json.dump({}, f)
        with open("output/mapreduce_results.json", "w") as f:
            json.dump({}, f)
        return {{}}, {{}}
