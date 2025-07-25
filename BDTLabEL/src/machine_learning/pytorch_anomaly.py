import os
os.environ['JAVA_HOME'] = r'C:\JAVA\jdk-1.8'

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

class TimeSeriesLSTMAutoencoder(nn.Module):
    def __init__(self, sequence_length, n_features, encoding_dim=32):
        super(TimeSeriesLSTMAutoencoder, self).__init__()
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.encoding_dim = encoding_dim
        
        # Encoder for time-series patterns
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=encoding_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Decoder for reconstruction
        self.decoder_lstm = nn.LSTM(
            input_size=encoding_dim,
            hidden_size=encoding_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.output_layer = nn.Linear(encoding_dim, n_features)
        
    def forward(self, x):
        # Encode the time series
        encoded, (hidden, cell) = self.encoder_lstm(x)
        
        # Use the last encoded state
        encoded_final = encoded[:, -1, :].unsqueeze(1)
        encoded_repeated = encoded_final.repeat(1, self.sequence_length, 1)
        
        # Decode back to original sequence
        decoded, _ = self.decoder_lstm(encoded_repeated)
        output = self.output_layer(decoded)
        
        return output

def prepare_timeseries_sequences(df, household='residential4', sequence_length=24):
    """Prepare sequences for time-series anomaly detection"""
    print(f"Preparing time-series sequences for {household}...")
    
    # Select appliance columns for the specific household
    appliance_cols = [col for col in df.columns if f"DE_KN_{household}_" in col]
    consumption_cols = [col for col in appliance_cols if not any(x in col for x in ['grid_export', 'grid_import', 'pv'])]
    
    if not consumption_cols:
        print(f"No consumption columns found for {household}")
        return None, None, None, None
    
    print(f"Using features: {consumption_cols}")
    
    # Sort by timestamp for proper time-series analysis
    df_sorted = df.sort_values('utc_timestamp').reset_index(drop=True)
    
    # Extract consumption values
    consumption_data = df_sorted[consumption_cols].fillna(0).values
    timestamps = df_sorted['utc_timestamp'].values
    
    # Normalize data for better LSTM performance
    scaler = StandardScaler()
    consumption_data_scaled = scaler.fit_transform(consumption_data)
    
    # Create overlapping sequences for time-series analysis
    sequences = []
    sequence_timestamps = []
    
    for i in range(len(consumption_data_scaled) - sequence_length + 1):
        sequences.append(consumption_data_scaled[i:i+sequence_length])
        sequence_timestamps.append(timestamps[i+sequence_length-1])
    
    sequences = np.array(sequences)
    print(f"Created {len(sequences)} time-series sequences with shape {sequences.shape}")
    
    return sequences, consumption_cols, scaler, sequence_timestamps

def detect_timeseries_anomalies(model, data_loader, timestamps, household, threshold_method='adaptive'):
    """Detect anomalies based on reconstruction error patterns"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in data_loader:
            sequences = batch[0].to(device)
            reconstructed = model(sequences)
            
            # Calculate reconstruction error for each sequence
            mse = torch.mean((sequences - reconstructed) ** 2, dim=(1, 2))
            reconstruction_errors.extend(mse.cpu().numpy())
    
    reconstruction_errors = np.array(reconstruction_errors)
    
    # Adaptive threshold based on error distribution
    if threshold_method == 'adaptive':
        # Use IQR method for robust threshold
        q75 = np.percentile(reconstruction_errors, 75)
        q25 = np.percentile(reconstruction_errors, 25)
        iqr = q75 - q25
        threshold = q75 + 1.5 * iqr
    elif threshold_method == 'statistical':
        # Use mean + 3*std
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        threshold = mean_error + 3 * std_error
    else:
        # Use percentile
        threshold = np.percentile(reconstruction_errors, 95)
    
    # Identify anomalies
    anomalies = reconstruction_errors > threshold
    
    # Additional check: look for sudden spikes in reconstruction error
    error_diff = np.diff(reconstruction_errors)
    spike_threshold = np.percentile(np.abs(error_diff), 95)
    spike_anomalies = np.abs(error_diff) > spike_threshold
    
    # Combine both types of anomalies
    combined_anomalies = anomalies.copy()
    combined_anomalies[1:] = combined_anomalies[1:] | spike_anomalies
    
    return reconstruction_errors, combined_anomalies, threshold

def visualize_timeseries_results(train_losses, reconstruction_errors, anomalies, timestamps, household):
    """Create visualizations for time-series anomaly detection"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # Training loss
    ax1.plot(train_losses, linewidth=2, color='blue')
    ax1.set_title(f'{household} - Training Loss (Time-Series)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Reconstruction errors over time
    timestamps_dt = pd.to_datetime(timestamps)
    colors = ['red' if anom else 'blue' for anom in anomalies]
    ax2.scatter(timestamps_dt, reconstruction_errors, c=colors, alpha=0.6, s=2)
    ax2.set_title(f'{household} - Reconstruction Errors Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Reconstruction Error')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Error distribution
    ax3.hist(reconstruction_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(np.percentile(reconstruction_errors, 95), color='red', linestyle='--', 
               label='95th Percentile')
    ax3.set_title(f'{household} - Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Reconstruction Error')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Time-series of anomalies
    anomaly_indices = np.where(anomalies)[0]
    if len(anomaly_indices) > 0:
        ax4.plot(range(len(reconstruction_errors)), reconstruction_errors, 'b-', alpha=0.5, label='Reconstruction Error')
        ax4.scatter(anomaly_indices, reconstruction_errors[anomaly_indices], 
                   color='red', s=20, label='Anomalies', zorder=5)
        ax4.set_title(f'{household} - Time-Series Anomaly Detection', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Sequence Index')
        ax4.set_ylabel('Reconstruction Error')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'output/{household}_timeseries_anomaly_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main time-series PyTorch anomaly detection function"""
    print("Starting PyTorch time-series anomaly detection...")
    
    try:
        # Load data
        df = pd.read_csv("data/raw/german_smart_meter_data.csv")
        print(f"Loaded {len(df)} records")
        
        # Process each household
        households = ['residential3', 'residential4', 'residential6']
        results = {}
        
        for household in households:
            print(f"\n=== Processing {household} with Time-Series Analysis ===")
            
            # Prepare time-series sequences
            sequences, feature_names, scaler, timestamps = prepare_timeseries_sequences(df, household)
            
            if sequences is None:
                print(f"Skipping {household} - no data available")
                continue
            
            # Convert to PyTorch tensors
            tensor_data = torch.FloatTensor(sequences)
            dataset = TensorDataset(tensor_data)
            
            # Split into train/test
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Initialize model
            model = TimeSeriesLSTMAutoencoder(
                sequence_length=24,
                n_features=sequences.shape[2],
                encoding_dim=64
            )
            
            # Train model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            print(f"Training on device: {device}")
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            train_losses = []
            for epoch in range(30):
                model.train()
                epoch_loss = 0
                for batch in train_loader:
                    sequences_batch = batch[0].to(device)
                    
                    reconstructed = model(sequences_batch)
                    loss = criterion(reconstructed, sequences_batch)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(train_loader)
                train_losses.append(avg_loss)
                
                if epoch % 10 == 0:
                    print(f'Epoch [{epoch}/30], Loss: {avg_loss:.6f}')
            
            # Detect time-series anomalies
            test_timestamps = timestamps[train_size:]
            reconstruction_errors, anomalies, threshold = detect_timeseries_anomalies(
                model, test_loader, test_timestamps, household, threshold_method='adaptive')

            # Extract anomaly timestamps in ISO format
            anomaly_timestamps = []
            if len(test_timestamps) == len(anomalies):
                anomaly_timestamps = [str(test_timestamps[i]) for i, is_anom in enumerate(anomalies) if is_anom]
            else:
                print(f"Warning: Length mismatch between test_timestamps and anomalies for {household}")

            # Save results
            results[household] = {
                'num_sequences': int(len(sequences)),
                'num_anomalies': int(sum(anomalies)),
                'anomaly_percentage': float(sum(anomalies) / len(anomalies) * 100) if len(anomalies) > 0 else 0.0,
                'feature_names': feature_names,
                'threshold': float(threshold),
                'avg_reconstruction_error': float(np.mean(reconstruction_errors)),
                'max_reconstruction_error': float(np.max(reconstruction_errors)),
                'anomaly_timestamps': anomaly_timestamps
            }

            print(f"Detected {sum(anomalies)} time-series anomalies out of {len(anomalies)} sequences ({sum(anomalies)/len(anomalies)*100 if len(anomalies) > 0 else 0:.2f}%)")
            print(f"Threshold: {threshold:.6f}")
            print(f"Average reconstruction error: {np.mean(reconstruction_errors):.6f}")

            # Save model and scaler
            torch.save(model.state_dict(), f'output/{household}_timeseries_model.pth')
            joblib.dump(scaler, f'output/{household}_timeseries_scaler.pkl')

            # Create visualizations
            visualize_timeseries_results(train_losses, reconstruction_errors, anomalies, 
                                       test_timestamps, household)
        
        # Save comprehensive results
        import json
        with open("output/timeseries_anomaly_results.json", "w") as f:
            json.dump(results, f, indent=2)
        # Also save for compatibility
        with open("output/pytorch_anomaly_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\n=== Time-Series Anomaly Detection Complete ===")
        print("Results saved to output/timeseries_anomaly_results.json")
        return results
    except Exception as e:
        import json
        print(f"Error in time-series PyTorch processing: {e}")
        # Always write empty file if error
        with open("output/timeseries_anomaly_results.json", "w") as f:
            json.dump({}, f)
        with open("output/pytorch_anomaly_results.json", "w") as f:
            json.dump({}, f)
        return None

if __name__ == "__main__":
    timeseries_results = main()
