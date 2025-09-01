#!/usr/bin/env python3
"""
Comprehensive Visualization Script for Neural Probe Data
Shows step-by-step data processing from pickle file to model input
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import torch

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_examine_pickle(file_path):
    """Load pickle file and examine its structure"""
    print("=== STEP 1: LOADING PICKLE FILE ===")
    print(f"File: {file_path}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data type: {type(data)}")
    print(f"Number of recordings: {len(data)}")
    
    if isinstance(data, list) and len(data) > 0:
        first_recording = data[0]
        print(f"First recording type: {type(first_recording)}")
        
        if isinstance(first_recording, tuple):
            signal, label = first_recording
            print(f"Signal type: {type(signal)}")
            print(f"Signal shape: {signal.shape if hasattr(signal, 'shape') else len(signal)}")
            print(f"Label: {label}")
            print(f"Signal dtype: {signal.dtype}")
            print(f"Signal range: [{signal.min():.6f}, {signal.max():.6f}]")
    
    return data

def visualize_individual_recordings(data, num_recordings=8):
    """Visualize individual recordings"""
    print(f"\n=== STEP 2: INDIVIDUAL RECORDINGS (First {num_recordings}) ===")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Individual Neural Recordings (First {num_recordings})', fontsize=16)
    
    for i in range(min(num_recordings, len(data))):
        row = i // 4
        col = i % 4
        
        signal, label = data[i]
        time_axis = np.arange(len(signal)) / 1250  # Assuming 1250 Hz sampling rate
        
        axes[row, col].plot(time_axis, signal, linewidth=0.8, alpha=0.8)
        axes[row, col].set_title(f'Recording {i}\n{label.split("_")[-1]}', fontsize=10)
        axes[row, col].set_xlabel('Time (s)')
        axes[row, col].set_ylabel('Amplitude')
        axes[row, col].grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        axes[row, col].text(0.02, 0.98, f'μ={mean_val:.6f}\nσ={std_val:.6f}', 
                           transform=axes[row, col].transAxes, 
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return fig

def visualize_signal_statistics(data, num_recordings=100):
    """Visualize statistics across recordings"""
    print(f"\n=== STEP 3: SIGNAL STATISTICS (First {num_recordings} recordings) ===")
    
    # Calculate statistics for first num_recordings
    means = []
    stds = []
    ranges = []
    labels = []
    
    for i in range(min(num_recordings, len(data))):
        signal, label = data[i]
        means.append(np.mean(signal))
        stds.append(np.std(signal))
        ranges.append(signal.max() - signal.min())
        labels.append(label.split('_')[-1])  # Extract brain region
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Signal Statistics Across Recordings (First {num_recordings})', fontsize=16)
    
    # Plot 1: Mean values
    axes[0, 0].hist(means, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Mean Amplitude')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Mean Values')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Standard deviations
    axes[0, 1].hist(stds, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('Standard Deviation')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Standard Deviations')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Ranges
    axes[1, 0].hist(ranges, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Range (Max - Min)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Ranges')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Brain region distribution
    region_counts = {}
    for label in labels:
        region_counts[label] = region_counts.get(label, 0) + 1
    
    regions = list(region_counts.keys())
    counts = list(region_counts.values())
    
    axes[1, 1].bar(regions, counts, alpha=0.7, color='gold', edgecolor='black')
    axes[1, 1].set_xlabel('Brain Region')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Distribution of Brain Regions')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def visualize_stacked_matrix(data, num_recordings=16):
    """Visualize the stacked matrix creation process"""
    print(f"\n=== STEP 4: CREATING STACKED MATRIX ({num_recordings} recordings) ===")
    
    # Extract signals and stack them
    signals = []
    labels = []
    
    for i in range(min(num_recordings, len(data))):
        signal, label = data[i]
        signals.append(signal)
        labels.append(label.split('_')[-1])
    
    # Stack into 2D matrix
    stacked_matrix = np.stack(signals)
    print(f"Stacked matrix shape: {stacked_matrix.shape}")
    print(f"Matrix range: [{stacked_matrix.min():.6f}, {stacked_matrix.max():.6f}]")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Stacked Matrix Creation Process ({num_recordings} recordings)', fontsize=16)
    
    # Plot 1: Heatmap of the stacked matrix
    im1 = axes[0, 0].imshow(stacked_matrix, aspect='auto', cmap='RdBu_r', 
                             vmin=stacked_matrix.min(), vmax=stacked_matrix.max())
    axes[0, 0].set_xlabel('Time Points')
    axes[0, 0].set_ylabel('Recordings')
    axes[0, 0].set_title('Stacked Matrix Heatmap')
    axes[0, 0].set_yticks(range(len(labels)))
    axes[0, 0].set_yticklabels([f'{i}: {label}' for i, label in enumerate(labels)])
    plt.colorbar(im1, ax=axes[0, 0], label='Amplitude')
    
    # Plot 2: Individual traces overlaid
    time_axis = np.arange(stacked_matrix.shape[1]) / 1250  # Assuming 1250 Hz
    for i in range(len(signals)):
        axes[0, 1].plot(time_axis, stacked_matrix[i], alpha=0.7, linewidth=0.8, 
                        label=f'{labels[i]}' if i < 5 else None)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('All Recordings Overlaid')
    axes[0, 1].legend(loc='upper right', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Correlation matrix between recordings
    corr_matrix = np.corrcoef(stacked_matrix)
    im3 = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 0].set_xlabel('Recording Index')
    axes[1, 0].set_ylabel('Recording Index')
    axes[1, 0].set_title('Correlation Matrix Between Recordings')
    plt.colorbar(im3, ax=axes[1, 0], label='Correlation')
    
    # Plot 4: Variance across time
    time_variance = np.var(stacked_matrix, axis=0)
    axes[1, 1].plot(time_axis, time_variance, linewidth=1.5, color='purple')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Variance')
    axes[1, 1].set_title('Variance Across Recordings Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig, stacked_matrix, labels

def visualize_model_input_reshaping(stacked_matrix, labels):
    """Visualize the reshaping process for model input"""
    print(f"\n=== STEP 5: MODEL INPUT RESHAPING ===")
    
    # Convert to torch tensor
    tensor_matrix = torch.tensor(stacked_matrix, dtype=torch.float32)
    print(f"Torch tensor shape: {tensor_matrix.shape}")
    
    # Reshape for model input: [1, 1, num_recordings, time_points]
    model_input = tensor_matrix.unsqueeze(0).unsqueeze(0)
    print(f"Model input shape: {model_input.shape}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Model Input Reshaping Process', fontsize=16)
    
    # Plot 1: Original stacked matrix
    im1 = axes[0, 0].imshow(stacked_matrix, aspect='auto', cmap='RdBu_r')
    axes[0, 0].set_xlabel('Time Points')
    axes[0, 0].set_ylabel('Recordings')
    axes[0, 0].set_title('Original Stacked Matrix')
    axes[0, 0].set_yticks(range(len(labels)))
    axes[0, 0].set_yticklabels([f'{i}: {labels[i]}' for i in range(len(labels))])
    plt.colorbar(im1, ax=axes[0, 0], label='Amplitude')
    
    # Plot 2: Model input tensor (first batch, first channel)
    model_data = model_input[0, 0].numpy()
    im2 = axes[0, 1].imshow(model_data, aspect='auto', cmap='RdBu_r')
    axes[0, 1].set_xlabel('Time Points')
    axes[0, 1].set_ylabel('Recordings')
    axes[0, 1].set_title('Model Input Tensor [0, 0, :, :]')
    axes[0, 1].set_yticks(range(len(labels)))
    axes[0, 1].set_yticklabels([f'{i}: {labels[i]}' for i in range(len(labels))])
    plt.colorbar(im2, ax=axes[0, 1], label='Amplitude')
    
    # Plot 3: Data preservation check
    original_slice = stacked_matrix[0, :100]  # First 100 time points of first recording
    model_slice = model_data[0, :100]
    
    axes[1, 0].plot(original_slice, label='Original', linewidth=2, alpha=0.8)
    axes[1, 0].plot(model_slice, label='Model Input', linewidth=2, alpha=0.8, linestyle='--')
    axes[1, 0].set_xlabel('Time Points (First 100)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title('Data Preservation Check (First Recording)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Shape transformation summary
    shapes = ['Original', 'Model Input']
    dimensions = [f'{stacked_matrix.shape[0]}×{stacked_matrix.shape[1]}', 
                 f'{model_input.shape[0]}×{model_input.shape[1]}×{model_input.shape[2]}×{model_input.shape[3]}']
    
    axes[1, 1].bar(shapes, [len(str(dim)) for dim in dimensions], color=['lightblue', 'lightcoral'])
    axes[1, 1].set_ylabel('Complexity (string length)')
    axes[1, 1].set_title('Shape Transformation Summary')
    
    # Add text annotations
    for i, (shape, dim) in enumerate(zip(shapes, dimensions)):
        axes[1, 1].text(i, len(str(dim)) + 0.1, dim, ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def visualize_time_series_means(data, num_recordings=100):
    """Visualize mean values across all time points for multiple recordings"""
    print(f"\n=== STEP 6: TIME SERIES MEANS ({num_recordings} recordings) ===")
    
    # Extract signals for the specified number of recordings
    signals = []
    labels = []
    
    for i in range(min(num_recordings, len(data))):
        signal, label = data[i]
        signals.append(signal)
        labels.append(label.split('_')[-1])
    
    # Stack into matrix
    stacked_signals = np.stack(signals)
    print(f"Stacked signals shape: {stacked_signals.shape}")
    
    # Calculate mean across recordings for each time point
    time_means = np.mean(stacked_signals, axis=0)
    time_stds = np.std(stacked_signals, axis=0)
    
    # Create time axis (assuming 1250 Hz sampling rate)
    time_axis = np.arange(len(time_means)) / 1250
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Time Series Analysis ({num_recordings} recordings)', fontsize=16)
    
    # Plot 1: Mean across all recordings over time
    axes[0, 0].plot(time_axis, time_means, linewidth=2, color='blue', alpha=0.8)
    axes[0, 0].fill_between(time_axis, time_means - time_stds, time_means + time_stds, 
                            alpha=0.3, color='blue', label='±1σ')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Mean Amplitude')
    axes[0, 0].set_title(f'Mean Signal Across {num_recordings} Recordings Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Standard deviation over time
    axes[0, 1].plot(time_axis, time_stds, linewidth=2, color='red', alpha=0.8)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Standard Deviation')
    axes[0, 1].set_title(f'Signal Variability Over Time ({num_recordings} recordings)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Individual recordings overlaid (first 20 for clarity)
    num_to_show = min(20, num_recordings)
    for i in range(num_to_show):
        axes[1, 0].plot(time_axis, stacked_signals[i], alpha=0.5, linewidth=0.8, 
                        label=f'{labels[i]}' if i < 5 else None)
    axes[1, 0].plot(time_axis, time_means, linewidth=3, color='black', 
                     label='Mean (all recordings)', alpha=0.9)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title(f'Individual Recordings + Mean (showing first {num_to_show})')
    axes[1, 0].legend(loc='upper right', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Statistics summary
    stats_data = {
        'Mean': [np.mean(time_means), np.std(time_means)],
        'Std': [np.mean(time_stds), np.std(time_stds)],
        'Min': [np.min(time_means), np.std(time_means)],
        'Max': [np.max(time_means), np.std(time_means)]
    }
    
    stat_names = list(stats_data.keys())
    stat_values = [stats_data[name][0] for name in stat_names]
    stat_errors = [stats_data[name][1] for name in stat_names]
    
    bars = axes[1, 1].bar(stat_names, stat_values, yerr=stat_errors, 
                           capsize=5, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Summary Statistics Across Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, stat_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(stat_values) * 0.01,
                        f'{value:.6f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return fig, time_means, time_stds

def visualize_all_recordings_means(data):
    """Visualize mean values for ALL recordings (full dataset)"""
    print(f"\n=== STEP 7: ALL RECORDINGS MEANS (Full dataset: {len(data)} recordings) ===")
    
    # This will be memory-intensive, so we'll process in chunks
    chunk_size = 1000
    num_chunks = (len(data) + chunk_size - 1) // chunk_size
    
    print(f"Processing {len(data)} recordings in {num_chunks} chunks of {chunk_size}")
    
    # Initialize arrays to store chunk means
    all_time_means = []
    all_time_stds = []
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(data))
        
        print(f"Processing chunk {chunk_idx + 1}/{num_chunks}: recordings {start_idx}-{end_idx}")
        
        # Extract signals for this chunk
        chunk_signals = []
        for i in range(start_idx, end_idx):
            signal, _ = data[i]
            chunk_signals.append(signal)
        
        # Stack and compute statistics
        chunk_matrix = np.stack(chunk_signals)
        chunk_means = np.mean(chunk_matrix, axis=0)
        chunk_stds = np.std(chunk_matrix, axis=0)
        
        all_time_means.append(chunk_means)
        all_time_stds.append(chunk_stds)
    
    # Combine all chunks
    all_means = np.array(all_time_means)
    all_stds = np.array(all_time_stds)
    
    print(f"Combined shape: {all_means.shape}")
    
    # Calculate overall statistics
    overall_mean = np.mean(all_means, axis=0)
    overall_std = np.std(all_means, axis=0)
    overall_std_of_means = np.std(all_means, axis=0)
    
    # Create time axis
    time_axis = np.arange(len(overall_mean)) / 1250
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'Complete Dataset Analysis: {len(data)} Recordings', fontsize=18)
    
    # Plot 1: Overall mean across all recordings
    axes[0, 0].plot(time_axis, overall_mean, linewidth=2, color='darkblue', alpha=0.9)
    axes[0, 0].fill_between(time_axis, overall_mean - overall_std, overall_mean + overall_std, 
                            alpha=0.3, color='darkblue', label='±1σ across chunks')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Mean Amplitude')
    axes[0, 0].set_title(f'Overall Mean Signal (All {len(data)} Recordings)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Variability across chunks
    axes[0, 1].plot(time_axis, overall_std_of_means, linewidth=2, color='darkred', alpha=0.9)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Standard Deviation of Means')
    axes[0, 1].set_title('Variability of Mean Values Across Chunks')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Heatmap of chunk means
    im = axes[1, 0].imshow(all_means, aspect='auto', cmap='RdBu_r', 
                           vmin=all_means.min(), vmax=all_means.max())
    axes[1, 0].set_xlabel('Time Points')
    axes[1, 0].set_ylabel('Chunks')
    axes[1, 0].set_title(f'Mean Values Heatmap ({num_chunks} chunks × {chunk_size} recordings)')
    plt.colorbar(im, ax=axes[1, 0], label='Mean Amplitude')
    
    # Plot 4: Statistics distribution
    chunk_final_means = all_means[:, -1]  # Last time point of each chunk
    chunk_final_stds = all_stds[:, -1]
    
    axes[1, 1].scatter(chunk_final_means, chunk_final_stds, alpha=0.7, s=50)
    axes[1, 1].set_xlabel('Mean Amplitude (Final Time Point)')
    axes[1, 1].set_ylabel('Standard Deviation (Final Time Point)')
    axes[1, 1].set_title('Chunk Statistics: Final Time Point')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(chunk_final_means, chunk_final_stds, 1)
    p = np.poly1d(z)
    axes[1, 1].plot(chunk_final_means, p(chunk_final_means), "r--", alpha=0.8, 
                     label=f'Trend: y={z[0]:.2e}x+{z[1]:.2e}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig, overall_mean, overall_std

def visualize_final_summary(data, stacked_matrix, model_input):
    """Create a final summary visualization"""
    print(f"\n=== STEP 8: FINAL SUMMARY ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Complete Data Processing Pipeline Summary', fontsize=18)
    
    # Plot 1: Data flow diagram
    axes[0, 0].text(0.5, 0.8, 'Pickle File\n(265,608 recordings)', ha='center', va='center', 
                     fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[0, 0].arrow(0.5, 0.7, 0, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
    axes[0, 0].text(0.5, 0.5, 'SessionDataset\n(Individual recordings)', ha='center', va='center', 
                     fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    axes[0, 0].arrow(0.5, 0.4, 0, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
    axes[0, 0].text(0.5, 0.2, 'DataLoader\n(Batched recordings)', ha='center', va='center', 
                     fontsize=12, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    axes[0, 0].arrow(0.5, 0.1, 0, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
    axes[0, 0].text(0.5, -0.1, 'Model Input\n(4D tensor)', ha='center', va='center', 
                     fontsize=12, bbox=dict(boxstyle='round', facecolor='gold', alpha=0.8))
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(-0.2, 1)
    axes[0, 0].set_title('Data Flow Pipeline')
    axes[0, 0].axis('off')
    
    # Plot 2: Shape evolution
    shapes = ['Pickle\nList', 'Individual\nRecording', 'Stacked\nMatrix', 'Model\nInput']
    dimensions = [len(data), f'({stacked_matrix.shape[1]},)', f'{stacked_matrix.shape[0]}×{stacked_matrix.shape[1]}', 
                 f'{model_input.shape[0]}×{model_input.shape[1]}×{model_input.shape[2]}×{model_input.shape[3]}']
    
    bars = axes[0, 1].bar(shapes, [len(str(dim)) for dim in dimensions], 
                           color=['lightblue', 'lightgreen', 'lightcoral', 'gold'])
    axes[0, 1].set_ylabel('Complexity')
    axes[0, 1].set_title('Shape Evolution')
    
    # Add text annotations
    for i, (shape, dim) in enumerate(zip(shapes, dimensions)):
        axes[0, 1].text(i, len(str(dim)) + 0.1, str(dim), ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Data statistics comparison
    stats_names = ['Min', 'Max', 'Mean', 'Std']
    original_stats = [stacked_matrix.min(), stacked_matrix.max(), np.mean(stacked_matrix), np.std(stacked_matrix)]
    model_stats = [model_input.min().item(), model_input.max().item(), 
                   model_input.mean().item(), model_input.std().item()]
    
    x = np.arange(len(stats_names))
    width = 0.35
    
    axes[0, 2].bar(x - width/2, original_stats, width, label='Original Data', alpha=0.8)
    axes[0, 2].bar(x + width/2, model_stats, width, label='Model Input', alpha=0.8)
    axes[0, 2].set_xlabel('Statistics')
    axes[0, 2].set_ylabel('Value')
    axes[0, 2].set_title('Data Statistics Comparison')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(stats_names)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Memory usage estimation
    data_types = ['Original\nPickle', 'Stacked\nMatrix', 'Model\nInput']
    memory_sizes = [len(data) * stacked_matrix.shape[1] * 4,  # 4 bytes per float32
                   stacked_matrix.nbytes,
                   model_input.element_size() * model_input.nelement()]
    
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    bars = axes[1, 0].bar(data_types, memory_sizes, color=colors)
    axes[1, 0].set_ylabel('Memory (bytes)')
    axes[1, 0].set_title('Memory Usage Estimation')
    
    # Add text annotations
    for i, (data_type, size) in enumerate(zip(data_types, memory_sizes)):
        axes[1, 0].text(i, size + max(memory_sizes) * 0.01, f'{size:,}', 
                        ha='center', va='bottom', fontsize=9)
    
    # Plot 5: Time series sample
    time_axis = np.arange(stacked_matrix.shape[1]) / 1250  # Assuming 1250 Hz
    for i in range(min(5, stacked_matrix.shape[0])):
        axes[1, 1].plot(time_axis, stacked_matrix[i], alpha=0.7, linewidth=1, 
                        label=f'Recording {i}' if i < 3 else None)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title('Sample Time Series (First 5 recordings)')
    axes[1, 1].legend(loc='upper right', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Final model input structure
    model_data = model_input[0, 0].numpy()
    im = axes[1, 2].imshow(model_data, aspect='auto', cmap='RdBu_r')
    axes[1, 2].set_xlabel('Time Points')
    axes[1, 2].set_ylabel('Recordings')
    axes[1, 2].set_title('Final Model Input Structure')
    plt.colorbar(im, ax=axes[1, 2], label='Amplitude')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def main():
    """Main function to run all visualizations"""
    print("🧠 NEURAL PROBE DATA VISUALIZATION SCRIPT 🧠")
    print("=" * 60)
    
    # File path
    pickle_file = '/Users/uttamsingh/Downloads/715093703_810755797.pickle'
    
    try:
        # Step 1: Load and examine pickle file
        data = load_and_examine_pickle(pickle_file)
        
        # Step 2: Visualize individual recordings
        fig1 = visualize_individual_recordings(data)
        
        # Step 3: Visualize signal statistics
        fig2 = visualize_signal_statistics(data)
        
        # Step 4: Visualize stacked matrix creation
        fig3, stacked_matrix, labels = visualize_stacked_matrix(data)
        
        # Step 5: Visualize model input reshaping
        fig4 = visualize_model_input_reshaping(stacked_matrix, labels)
        
        # Step 6: Visualize time series means (100 recordings)
        fig5, time_means, time_stds = visualize_time_series_means(data, num_recordings=100)
        
        # Step 7: Visualize ALL recordings means (full dataset)
        fig6, overall_mean, overall_std = visualize_all_recordings_means(data)
        
        # Step 8: Final summary
        model_input = torch.tensor(stacked_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        fig7 = visualize_final_summary(data, stacked_matrix, model_input)
        
        print("\n🎉 ALL VISUALIZATIONS COMPLETED SUCCESSFULLY! 🎉")
        print("\nSummary of what we learned:")
        print(f"• Pickle file contains {len(data)} recordings")
        print(f"• Each recording has {stacked_matrix.shape[1]} time points")
        print(f"• Stacked matrix shape: {stacked_matrix.shape}")
        print(f"• Final model input shape: {model_input.shape}")
        print(f"• Data range: [{stacked_matrix.min():.6f}, {stacked_matrix.max():.6f}]")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
