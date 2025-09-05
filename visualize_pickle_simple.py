#!/usr/bin/env python3
"""
Simple visualization script for pickle neural data
No external dependencies required - uses only matplotlib and numpy
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from pathlib import Path

class SimpleBrainVisualizer:
    def __init__(self, pickle_path, output_dir="brain_visualizations"):
        self.pickle_path = pickle_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load pickle data
        self.data = self.load_pickle_data()
    
    def load_pickle_data(self):
        """Load pickle data and extract neural information"""
        print(f"📁 Loading pickle data from: {self.pickle_path}")
        
        try:
            with open(self.pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            print(f"✅ Successfully loaded pickle data")
            print(f"   Data type: {type(data)}")
            
            if isinstance(data, dict):
                print(f"   Keys: {list(data.keys())}")
                for key, value in data.items():
                    if hasattr(value, 'shape'):
                        print(f"   {key}: {value.shape} ({value.dtype})")
                    else:
                        print(f"   {key}: {type(value)}")
            elif hasattr(data, 'shape'):
                print(f"   Shape: {data.shape} ({data.dtype})")
            
            return data
        except Exception as e:
            print(f"❌ Error loading pickle data: {e}")
            return None
    
    def extract_neural_activity(self):
        """Extract neural activity data from pickle"""
        if self.data is None:
            return None, None, None
        
        # Try to extract neural activity data
        if isinstance(self.data, dict):
            # Look for common neural data keys
            possible_keys = ['data', 'neural_data', 'activity', 'spikes', 'lfp', 'signals', 'X', 'y']
            neural_data = None
            
            for key in possible_keys:
                if key in self.data:
                    neural_data = self.data[key]
                    print(f"   Found data under key: {key}")
                    break
            
            if neural_data is None:
                # Use the first array-like value
                for key, value in self.data.items():
                    if hasattr(value, 'shape') and len(value.shape) >= 2:
                        neural_data = value
                        print(f"   Using data from key: {key}")
                        break
            
            if neural_data is not None:
                print(f"📊 Found neural data: {neural_data.shape}")
                
                # Extract time points and channels
                if len(neural_data.shape) == 2:
                    # 2D: [channels, time_points] or [time_points, channels]
                    if neural_data.shape[0] > neural_data.shape[1]:
                        # [channels, time_points]
                        channels = neural_data.shape[0]
                        time_points = neural_data.shape[1]
                        data_matrix = neural_data
                    else:
                        # [time_points, channels]
                        time_points = neural_data.shape[0]
                        channels = neural_data.shape[1]
                        data_matrix = neural_data.T
                else:
                    print(f"⚠️ Unexpected data shape: {neural_data.shape}")
                    return None, None, None
                
                return data_matrix, channels, time_points
            else:
                print("❌ No neural data found in pickle file")
                return None, None, None
        elif isinstance(self.data, list):
            # List data - try to find array-like elements
            print(f"   List with {len(self.data)} elements")
            for i, item in enumerate(self.data[:5]):  # Check first 5 items
                if hasattr(item, 'shape'):
                    print(f"   Item {i}: {item.shape} ({item.dtype})")
                elif isinstance(item, tuple):
                    print(f"   Item {i}: tuple with {len(item)} elements")
                    for j, subitem in enumerate(item[:3]):  # Check first 3 subitems
                        if hasattr(subitem, 'shape'):
                            print(f"     Subitem {j}: {subitem.shape} ({subitem.dtype})")
                        else:
                            print(f"     Subitem {j}: {type(subitem)}")
                else:
                    print(f"   Item {i}: {type(item)}")
            
            # Look for array-like data in tuples
            arrays_found = []
            for i, item in enumerate(self.data):
                if isinstance(item, tuple):
                    for j, subitem in enumerate(item):
                        if hasattr(subitem, 'shape') and len(subitem.shape) >= 2:
                            print(f"   Found 2D array in tuple {i}, subitem {j}: {subitem.shape}")
                            return subitem, subitem.shape[0], subitem.shape[1]
                        elif hasattr(subitem, 'shape') and len(subitem.shape) == 1:
                            # Collect 1D arrays to stack later
                            arrays_found.append(subitem)
                elif hasattr(item, 'shape') and len(item.shape) >= 2:
                    print(f"   Using 2D array item {i}: {item.shape}")
                    return item, item.shape[0], item.shape[1]
            
            # If we found 1D arrays, stack them into a 2D matrix
            if arrays_found:
                print(f"   Found {len(arrays_found)} 1D arrays, sampling for visualization...")
                try:
                    # Sample a reasonable number of channels (e.g., 100-500)
                    max_channels = 500
                    if len(arrays_found) > max_channels:
                        print(f"   Sampling {max_channels} channels from {len(arrays_found)} total")
                        # Sample evenly spaced channels
                        indices = np.linspace(0, len(arrays_found)-1, max_channels, dtype=int)
                        sampled_arrays = [arrays_found[i] for i in indices]
                    else:
                        sampled_arrays = arrays_found
                    
                    # Stack sampled arrays into a 2D matrix
                    data_matrix = np.stack(sampled_arrays, axis=0)  # (channels, time_points)
                    print(f"   Stacked data shape: {data_matrix.shape}")
                    return data_matrix, data_matrix.shape[0], data_matrix.shape[1]
                except Exception as e:
                    print(f"   Error stacking arrays: {e}")
                    return None, None, None
            
            print("❌ No array-like data found in list")
            return None, None, None
        else:
            # Direct array data
            if hasattr(self.data, 'shape') and len(self.data.shape) >= 2:
                return self.data, self.data.shape[0], self.data.shape[1]
            else:
                print("❌ Data is not in expected format")
                return None, None, None
    
    def create_channel_positions(self, num_channels):
        """Create 3D positions for channels (simulated probe geometry)"""
        print(f"📍 Creating 3D positions for {num_channels} channels")
        
        # Simulate a linear probe with depth variation
        # This is a simplified model - in reality, you'd use actual probe coordinates
        
        # Create a linear probe along z-axis (depth)
        z_positions = np.linspace(0, 3.8, num_channels)  # 3.8mm depth (typical mouse brain)
        
        # Add slight variation in x,y for realistic probe geometry
        np.random.seed(42)  # For reproducible results
        x_positions = np.random.normal(0, 0.1, num_channels)  # Small variation
        y_positions = np.random.normal(0, 0.1, num_channels)  # Small variation
        
        positions = np.column_stack([x_positions, y_positions, z_positions])
        
        print(f"   Probe length: {z_positions.max() - z_positions.min():.2f}mm")
        print(f"   Depth range: {z_positions.min():.2f} - {z_positions.max():.2f}mm")
        
        return positions
    
    def visualize_3d_activity(self, data_matrix, channel_positions, time_points):
        """Create 3D visualization of neural activity"""
        print("🎨 Creating 3D neural activity visualization...")
        
        # Calculate mean activity per channel
        mean_activity = np.mean(data_matrix, axis=1)
        
        # Create 3D scatter plot
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by activity level
        scatter = ax.scatter(
            channel_positions[:, 0],
            channel_positions[:, 1], 
            channel_positions[:, 2],
            c=mean_activity,
            cmap='viridis',
            s=100,
            alpha=0.8
        )
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Mean Activity')
        
        # Labels and title
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Y Position (mm)')
        ax.set_zlabel('Depth (mm)')
        ax.set_title('Neural Activity in 3D Brain Space')
        
        # Save plot
        output_path = self.output_dir / "3d_neural_activity.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved 3D visualization: {output_path}")
        
        plt.show()
    
    def visualize_activity_heatmap(self, data_matrix, time_points):
        """Create activity heatmap over time"""
        print("🔥 Creating activity heatmap...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Full heatmap
        im1 = ax1.imshow(data_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
        ax1.set_xlabel('Time Points')
        ax1.set_ylabel('Channels')
        ax1.set_title('Neural Activity Heatmap (All Channels)')
        plt.colorbar(im1, ax=ax1, label='Activity')
        
        # Mean activity over time
        mean_over_time = np.mean(data_matrix, axis=0)
        ax2.plot(mean_over_time, linewidth=2, color='red')
        ax2.set_xlabel('Time Points')
        ax2.set_ylabel('Mean Activity')
        ax2.set_title('Mean Activity Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "activity_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved heatmap: {output_path}")
        
        plt.show()
    
    def visualize_channel_distribution(self, data_matrix, channel_positions):
        """Visualize channel activity distribution"""
        print("📊 Creating channel distribution visualization...")
        
        # Calculate statistics per channel
        mean_activity = np.mean(data_matrix, axis=1)
        std_activity = np.std(data_matrix, axis=1)
        max_activity = np.max(data_matrix, axis=1)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Mean activity by depth
        ax1.scatter(channel_positions[:, 2], mean_activity, alpha=0.7, s=50)
        ax1.set_xlabel('Depth (mm)')
        ax1.set_ylabel('Mean Activity')
        ax1.set_title('Mean Activity vs Depth')
        ax1.grid(True, alpha=0.3)
        
        # Activity variability by depth
        ax2.scatter(channel_positions[:, 2], std_activity, alpha=0.7, s=50, color='orange')
        ax2.set_xlabel('Depth (mm)')
        ax2.set_ylabel('Activity Std Dev')
        ax2.set_title('Activity Variability vs Depth')
        ax2.grid(True, alpha=0.3)
        
        # Activity histogram
        ax3.hist(mean_activity, bins=30, alpha=0.7, color='green')
        ax3.set_xlabel('Mean Activity')
        ax3.set_ylabel('Number of Channels')
        ax3.set_title('Distribution of Mean Activity')
        ax3.grid(True, alpha=0.3)
        
        # Max activity by depth
        ax4.scatter(channel_positions[:, 2], max_activity, alpha=0.7, s=50, color='red')
        ax4.set_xlabel('Depth (mm)')
        ax4.set_ylabel('Max Activity')
        ax4.set_title('Max Activity vs Depth')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "channel_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved distribution plot: {output_path}")
        
        plt.show()
    
    def visualize_time_series(self, data_matrix, num_samples=5):
        """Visualize time series for selected channels"""
        print("⏰ Creating time series visualization...")
        
        # Select random channels to visualize
        num_channels = data_matrix.shape[0]
        selected_channels = np.random.choice(num_channels, min(num_samples, num_channels), replace=False)
        
        fig, axes = plt.subplots(len(selected_channels), 1, figsize=(15, 3*len(selected_channels)))
        if len(selected_channels) == 1:
            axes = [axes]
        
        for i, channel in enumerate(selected_channels):
            axes[i].plot(data_matrix[channel, :], linewidth=1, alpha=0.8)
            axes[i].set_ylabel(f'Channel {channel}')
            axes[i].set_title(f'Time Series - Channel {channel}')
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time Points')
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "time_series.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved time series: {output_path}")
        
        plt.show()
    
    def run_visualization(self):
        """Run complete visualization pipeline"""
        print("🚀 Starting brain visualization pipeline...")
        
        # Extract neural data
        data_matrix, num_channels, time_points = self.extract_neural_activity()
        if data_matrix is None:
            print("❌ Could not extract neural data")
            return
        
        print(f"📊 Data summary:")
        print(f"   Channels: {num_channels}")
        print(f"   Time points: {time_points}")
        print(f"   Data range: {data_matrix.min():.3f} - {data_matrix.max():.3f}")
        
        # Create channel positions
        channel_positions = self.create_channel_positions(num_channels)
        
        # Generate visualizations
        self.visualize_3d_activity(data_matrix, channel_positions, time_points)
        self.visualize_activity_heatmap(data_matrix, time_points)
        self.visualize_channel_distribution(data_matrix, channel_positions)
        self.visualize_time_series(data_matrix)
        
        print("✅ Visualization pipeline completed!")
        print(f"📁 All plots saved to: {self.output_dir}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize pickle neural data in mouse brain')
    parser.add_argument('pickle_path', help='Path to pickle file')
    parser.add_argument('--output_dir', default='brain_visualizations', 
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Check if pickle file exists
    if not os.path.exists(args.pickle_path):
        print(f"❌ Pickle file not found: {args.pickle_path}")
        return
    
    # Create visualizer and run
    visualizer = SimpleBrainVisualizer(args.pickle_path, args.output_dir)
    visualizer.run_visualization()

if __name__ == "__main__":
    main()
