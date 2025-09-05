#!/usr/bin/env python3
"""
Visualize pickle neural data in mouse brain using Allen Brain Atlas SDK
This script loads pickle files and visualizes the neural activity in 3D brain space
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
from pathlib import Path

# Allen Brain Atlas SDK
try:
    from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
    from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
    from allensdk.core.reference_space_cache import ReferenceSpaceCache
    from allensdk.core.structure_tree import StructureTree
    ALLEN_SDK_AVAILABLE = True
except ImportError:
    print("⚠️ Allen Brain Atlas SDK not available. Install with: pip install allensdk")
    ALLEN_SDK_AVAILABLE = False

# Alternative visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("⚠️ Plotly not available. Install with: pip install plotly")
    PLOTLY_AVAILABLE = False

class BrainVisualizer:
    def __init__(self, pickle_path, output_dir="brain_visualizations"):
        self.pickle_path = pickle_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load pickle data
        self.data = self.load_pickle_data()
        
        # Initialize Allen Brain Atlas if available
        self.mcc = None
        self.structure_tree = None
        if ALLEN_SDK_AVAILABLE:
            self.initialize_allen_atlas()
    
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
    
    def initialize_allen_atlas(self):
        """Initialize Allen Brain Atlas SDK"""
        print("🧠 Initializing Allen Brain Atlas...")
        try:
            # Initialize connectivity cache
            self.mcc = MouseConnectivityCache(manifest_file='connectivity_manifest.json')
            
            # Initialize reference space cache
            rspc = ReferenceSpaceCache(25, 'annotation/ccf_2017', manifest_file='reference_space_manifest.json')
            self.structure_tree = rspc.get_structure_tree()
            
            print("✅ Allen Brain Atlas initialized successfully")
        except Exception as e:
            print(f"⚠️ Could not initialize Allen Brain Atlas: {e}")
            self.mcc = None
            self.structure_tree = None
    
    def extract_neural_activity(self):
        """Extract neural activity data from pickle"""
        if self.data is None:
            return None, None, None
        
        # Try to extract neural activity data
        if isinstance(self.data, dict):
            # Look for common neural data keys
            possible_keys = ['data', 'neural_data', 'activity', 'spikes', 'lfp', 'signals']
            neural_data = None
            
            for key in possible_keys:
                if key in self.data:
                    neural_data = self.data[key]
                    break
            
            if neural_data is None:
                # Use the first array-like value
                for key, value in self.data.items():
                    if hasattr(value, 'shape') and len(value.shape) >= 2:
                        neural_data = value
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
    
    def create_interactive_plot(self, data_matrix, channel_positions):
        """Create interactive 3D plot using Plotly"""
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available, skipping interactive plot")
            return
        
        print("🎮 Creating interactive 3D plot...")
        
        # Calculate mean activity
        mean_activity = np.mean(data_matrix, axis=1)
        
        # Create interactive 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=channel_positions[:, 0],
            y=channel_positions[:, 1],
            z=channel_positions[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=mean_activity,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Mean Activity")
            ),
            text=[f'Channel {i}<br>Activity: {mean_activity[i]:.3f}' for i in range(len(mean_activity))],
            hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Interactive Neural Activity in 3D Brain Space',
            scene=dict(
                xaxis_title='X Position (mm)',
                yaxis_title='Y Position (mm)',
                zaxis_title='Depth (mm)'
            ),
            width=800,
            height=600
        )
        
        # Save interactive plot
        output_path = self.output_dir / "interactive_3d_plot.html"
        fig.write_html(str(output_path))
        print(f"💾 Saved interactive plot: {output_path}")
        
        # Show plot
        fig.show()
    
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
        self.create_interactive_plot(data_matrix, channel_positions)
        
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
    visualizer = BrainVisualizer(args.pickle_path, args.output_dir)
    visualizer.run_visualization()

if __name__ == "__main__":
    main()
