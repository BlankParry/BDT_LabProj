import psutil
import time
import threading
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import json
import numpy as np

class SystemHeatMonitor:
    def __init__(self):
        self.cpu_temps = deque(maxlen=1000)
        self.cpu_usage = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=1000)
        self.disk_usage = deque(maxlen=1000)
        self.timestamps = deque(maxlen=1000)
        self.monitoring = False
        
    def start_monitoring(self):
        """Start system monitoring for time-series processing"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_system)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("System monitoring started for time-series processing...")
        
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        print("System monitoring stopped.")
            
    def _monitor_system(self):
        """Internal monitoring loop optimized for time-series workloads"""
        while self.monitoring:
            try:
                # CPU temperature (simulated based on usage if sensors not available)
                try:
                    temps = psutil.sensors_temperatures()
                    if temps and 'coretemp' in temps:
                        cpu_temp = temps['coretemp'][0].current
                    else:
                        # Simulate temperature based on CPU usage
                        cpu_percent = psutil.cpu_percent()
                        cpu_temp = 30 + (cpu_percent * 0.6)  # Base temp + usage factor
                except:
                    cpu_percent = psutil.cpu_percent()
                    cpu_temp = 30 + (cpu_percent * 0.6)
                
                # System metrics
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Store metrics
                self.cpu_temps.append(cpu_temp)
                self.cpu_usage.append(psutil.cpu_percent())
                self.memory_usage.append(memory.percent)
                self.disk_usage.append(disk.percent)
                self.timestamps.append(time.time())
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                break
    
    def analyze_processing_heat(self, processing_function, *args, **kwargs):
        """Analyze heat generation during time-series processing"""
        print("Starting heat analysis for time-series processing task...")
        
        # Start monitoring
        self.start_monitoring()
        
        # Record baseline for 5 seconds
        time.sleep(5)
        baseline_metrics = self._get_current_averages()
        
        # Execute processing function
        start_time = time.time()
        try:
            print("Executing time-series processing function...")
            result = processing_function(*args, **kwargs)
            success = True
        except Exception as e:
            print(f"Processing error: {e}")
            result = None
            success = False
        end_time = time.time()
        
        # Continue monitoring for cool-down period
        print("Monitoring cool-down period...")
        time.sleep(10)
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Analyze results
        processing_duration = end_time - start_time
        peak_metrics = self._get_peak_metrics()
        avg_metrics = self._get_average_metrics()
        
        analysis = {
            'processing_duration': processing_duration,
            'success': success,
            'baseline': baseline_metrics,
            'peak': peak_metrics,
            'average': avg_metrics,
            'temperature_increase': peak_metrics['cpu_temp'] - baseline_metrics['cpu_temp'],
            'cpu_increase': peak_metrics['cpu_usage'] - baseline_metrics['cpu_usage'],
            'memory_increase': peak_metrics['memory_usage'] - baseline_metrics['memory_usage'],
            'thermal_efficiency': processing_duration / max(peak_metrics['cpu_temp'] - baseline_metrics['cpu_temp'], 0.1),
            'workload_type': 'time_series_processing'
        }
        
        return analysis, result
    
    def _get_current_averages(self):
        """Get current average metrics"""
        if len(self.cpu_temps) < 5:
            return {'cpu_temp': 30, 'cpu_usage': 0, 'memory_usage': 0, 'disk_usage': 0}
        
        return {
            'cpu_temp': sum(list(self.cpu_temps)[-5:]) / 5,
            'cpu_usage': sum(list(self.cpu_usage)[-5:]) / 5,
            'memory_usage': sum(list(self.memory_usage)[-5:]) / 5,
            'disk_usage': sum(list(self.disk_usage)[-5:]) / 5
        }
    
    def _get_peak_metrics(self):
        """Get peak metrics during monitoring"""
        if not self.cpu_temps:
            return {'cpu_temp': 30, 'cpu_usage': 0, 'memory_usage': 0, 'disk_usage': 0}
        
        return {
            'cpu_temp': max(self.cpu_temps),
            'cpu_usage': max(self.cpu_usage),
            'memory_usage': max(self.memory_usage),
            'disk_usage': max(self.disk_usage)
        }
    
    def _get_average_metrics(self):
        """Get average metrics during monitoring"""
        if not self.cpu_temps:
            return {'cpu_temp': 30, 'cpu_usage': 0, 'memory_usage': 0, 'disk_usage': 0}
        
        return {
            'cpu_temp': sum(self.cpu_temps) / len(self.cpu_temps),
            'cpu_usage': sum(self.cpu_usage) / len(self.cpu_usage),
            'memory_usage': sum(self.memory_usage) / len(self.memory_usage),
            'disk_usage': sum(self.disk_usage) / len(self.disk_usage)
        }
    
    def visualize_monitoring_data(self, save_path="output/timeseries_system_monitoring.png"):
        """Create visualization of monitoring data for time-series processing"""
        if not self.timestamps:
            print("No monitoring data available for visualization")
            return
        
        # Convert timestamps to relative time - FIXED
        start_time = self.timestamps[0]
        times = [(t - start_time) / 60 for t in self.timestamps]  # Convert to minutes
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # CPU Temperature - FIXED temperature symbol
        ax1.plot(times, list(self.cpu_temps), 'r-', linewidth=2, label='CPU Temperature')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('CPU Temperature During Time-Series Processing')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # CPU Usage
        ax2.plot(times, list(self.cpu_usage), 'b-', linewidth=2, label='CPU Usage')
        ax2.set_ylabel('CPU Usage (%)')
        ax2.set_title('CPU Usage During Time-Series Processing')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Memory Usage
        ax3.plot(times, list(self.memory_usage), 'g-', linewidth=2, label='Memory Usage')
        ax3.set_ylabel('Memory Usage (%)')
        ax3.set_xlabel('Time (minutes)')
        ax3.set_title('Memory Usage During Time-Series Processing')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Disk Usage
        ax4.plot(times, list(self.disk_usage), 'm-', linewidth=2, label='Disk Usage')
        ax4.set_ylabel('Disk Usage (%)')
        ax4.set_xlabel('Time (minutes)')
        ax4.set_title('Disk Usage During Time-Series Processing')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Time-series monitoring visualization saved to {save_path}")
    
    def generate_heat_report(self, analysis, save_path="output/timeseries_heat_analysis_report.json"):
        """Generate comprehensive heat analysis report for time-series processing"""
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'workload_type': 'time_series_anomaly_detection',
            'processing_analysis': analysis,
            'system_specifications': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'disk_total_gb': round(psutil.disk_usage('/').total / (1024**3), 2)
            },
            'recommendations': self._generate_timeseries_recommendations(analysis)
        }
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Time-series heat analysis report saved to {save_path}")
        return report
    
    def _generate_timeseries_recommendations(self, analysis):
        """Generate optimization recommendations for time-series processing"""
        recommendations = []
        
        if analysis['temperature_increase'] > 20:
            recommendations.append("High temperature increase detected during time-series processing. Consider improving cooling.")
        
        if analysis['cpu_increase'] > 80:
            recommendations.append("High CPU usage during LSTM training. Consider distributed processing or smaller batch sizes.")
        
        if analysis['memory_increase'] > 70:
            recommendations.append("High memory usage during sequence processing. Consider reducing sequence length or batch size.")
        
        if analysis['processing_duration'] > 300:  # 5 minutes
            recommendations.append("Long time-series processing duration. Consider optimizing LSTM architecture or using GPU acceleration.")
        
        if analysis['thermal_efficiency'] < 10:
            recommendations.append("Low thermal efficiency for time-series workload. Processing generates significant heat relative to duration.")
        
        # Time-series specific recommendations
        if analysis.get('workload_type') == 'time_series_processing':
            recommendations.append("Time-series processing detected. Ensure adequate cooling for sustained LSTM training workloads.")
        
        if not recommendations:
            recommendations.append("System performance is optimal for time-series anomaly detection workload.")
        
        return recommendations

def simulate_timeseries_processing():
    """Simulate time-series processing workload for testing"""
    print("Simulating time-series LSTM processing...")
    
    import numpy as np
    
    # Simulate LSTM training workload
    sequence_length = 24
    n_features = 6
    n_sequences = 10000
    
    # Generate synthetic time-series data
    data = np.random.randn(n_sequences, sequence_length, n_features)
    
    # Simulate LSTM operations
    results = []
    for epoch in range(20):
        # Simulate forward pass
        hidden_states = []
        for seq in range(min(1000, n_sequences)):  # Limit for simulation
            # Simulate LSTM cell operations
            hidden = np.random.randn(64)  # Hidden state
            cell = np.random.randn(64)    # Cell state
            
            for t in range(sequence_length):
                # Simulate LSTM computations
                input_gate = 1 / (1 + np.exp(-np.random.randn()))
                forget_gate = 1 / (1 + np.exp(-np.random.randn()))
                output_gate = 1 / (1 + np.exp(-np.random.randn()))
                
                # Update states
                cell = forget_gate * cell + input_gate * np.tanh(np.random.randn())
                hidden = output_gate * np.tanh(cell)
            
            hidden_states.append(hidden)
        
        # Simulate reconstruction error calculation
        reconstruction_errors = []
        for hidden in hidden_states:
            # Simulate decoder operations
            reconstructed = np.random.randn(sequence_length, n_features)
            original = data[len(reconstruction_errors)]
            error = np.mean((original - reconstructed) ** 2)
            reconstruction_errors.append(error)
        
        results.append({
            'epoch': epoch,
            'avg_loss': np.mean(reconstruction_errors),
            'hidden_states': len(hidden_states)
        })
        
        # Simulate progress
        if epoch % 5 == 0:
            print(f"Simulated epoch {epoch+1}/20 completed")
    
    print("Time-series processing simulation completed")
    return results

def main():
    """Main heat monitoring function for time-series processing"""
    print("=== Time-Series System Heat Monitoring Analysis ===")
    
    # Initialize heat monitor
    heat_monitor = SystemHeatMonitor()
    
    # Analyze heat generation during simulated processing
    analysis, processing_result = heat_monitor.analyze_processing_heat(simulate_timeseries_processing)
    
    # Display results
    print("\n=== Heat Analysis Results ===")
    print(f"Processing Duration: {analysis['processing_duration']:.2f} seconds")
    print(f"Temperature Increase: {analysis['temperature_increase']:.2f}°C")
    print(f"CPU Usage Increase: {analysis['cpu_increase']:.2f}%")
    print(f"Memory Usage Increase: {analysis['memory_increase']:.2f}%")
    print(f"Thermal Efficiency: {analysis['thermal_efficiency']:.2f} sec/°C")
    print(f"Workload Type: {analysis['workload_type']}")
    
    # Create visualizations
    heat_monitor.visualize_monitoring_data()
    
    # Generate comprehensive report
    report = heat_monitor.generate_heat_report(analysis)
    
    print("\n=== Recommendations ===")
    for rec in report['recommendations']:
        print(f"• {rec}")
    
    return analysis, report

if __name__ == "__main__":
    heat_analysis, heat_report = main()
