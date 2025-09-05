# Brain Visualization Scripts

This directory contains scripts to visualize pickle neural data in 3D mouse brain space.

## Files

- `visualize_pickle_simple.py` - Simple visualization (no external dependencies)
- `visualize_pickle_brain.py` - Advanced visualization with Allen Brain Atlas SDK
- `requirements_visualization.txt` - Required Python packages

## Quick Start

### Simple Visualization (Recommended)

```bash
# Install basic requirements
pip install matplotlib numpy

# Run visualization
python visualize_pickle_simple.py /path/to/your/data.pickle
```

### Advanced Visualization (with Allen Brain Atlas)

```bash
# Install all requirements
pip install -r requirements_visualization.txt

# Run advanced visualization
python visualize_pickle_brain.py /path/to/your/data.pickle
```

## What the Scripts Do

### 1. **3D Neural Activity Visualization**
- Creates a 3D scatter plot showing neural activity in brain space
- Each point represents a channel, colored by activity level
- Simulates probe geometry with realistic depth positioning

### 2. **Activity Heatmap**
- Shows neural activity over time for all channels
- Displays mean activity across the recording session
- Helps identify temporal patterns and channel correlations

### 3. **Channel Distribution Analysis**
- Analyzes activity distribution across different depths
- Shows mean activity, variability, and maximum activity by depth
- Helps understand spatial organization of neural activity

### 4. **Time Series Visualization**
- Displays raw time series for selected channels
- Helps identify temporal dynamics and signal quality

## Output

All visualizations are saved to the `brain_visualizations/` directory:
- `3d_neural_activity.png` - 3D scatter plot
- `activity_heatmap.png` - Activity heatmap over time
- `channel_distribution.png` - Channel distribution analysis
- `time_series.png` - Time series for selected channels
- `interactive_3d_plot.html` - Interactive 3D plot (if Plotly available)

## Data Format

The scripts automatically detect data format from your pickle file:

### Supported Formats:
- **Dictionary with neural data**: `{'data': array, 'X': array, 'neural_data': array, ...}`
- **Direct 2D array**: `[channels, time_points]` or `[time_points, channels]`
- **Any array-like data** with shape `(N, M)` where N and M are > 1

### Example Usage:
```python
# Your pickle file should contain data like:
data = {
    'X': np.random.randn(93, 3750),  # 93 channels, 3750 time points
    'y': np.random.randint(0, 10, 93),  # labels
    'metadata': {...}
}
```

## Customization

### Modify Channel Positions
Edit the `create_channel_positions()` method to use your actual probe coordinates:

```python
def create_channel_positions(self, num_channels):
    # Replace with your actual probe coordinates
    x_positions = your_x_coordinates
    y_positions = your_y_coordinates  
    z_positions = your_z_coordinates
    return np.column_stack([x_positions, y_positions, z_positions])
```

### Add Brain Atlas Integration
The advanced script includes Allen Brain Atlas SDK integration for:
- Real brain structure mapping
- Anatomical region identification
- Reference space coordinates

## Troubleshooting

### Common Issues:

1. **"No neural data found"**
   - Check your pickle file contains array-like data
   - Try different key names in the data dictionary

2. **"Unexpected data shape"**
   - Ensure your data is 2D (channels × time_points)
   - Check data orientation (may need transposition)

3. **Import errors**
   - Install missing packages: `pip install -r requirements_visualization.txt`
   - Use the simple version if Allen SDK is not available

### Getting Help:
- Check the console output for detailed error messages
- Verify your pickle file can be loaded with `pickle.load()`
- Ensure your data has the expected shape and format

## Example Output

The scripts will generate:
- **3D brain visualization** showing activity distribution
- **Heatmaps** revealing temporal patterns
- **Statistical analysis** of channel properties
- **Interactive plots** for detailed exploration

All plots are saved as high-resolution PNG files and can be used in presentations or publications.
