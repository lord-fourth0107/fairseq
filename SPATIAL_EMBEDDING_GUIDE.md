# Spatial Embedding Guide for Wav2Vec2_2D

## 🎯 Overview

The `wav2vec2_2d` model includes a **spatial embedding layer** that learns spatial representations based on recording sites. This is particularly useful for neural data where different recording sites have spatial relationships and may capture different types of neural activity patterns.

## 🏗️ Architecture

### Spatial Embedding Components

1. **Embedding Layer**: `nn.Embedding(num_recording_sites, spatial_embed_dim)`
2. **Dropout**: `nn.Dropout(spatial_embed_dropout)`
3. **Projection**: `nn.Linear(spatial_embed_dim, encoder_embed_dim)`
4. **Integration**: Added to features after post-extraction projection

### Data Flow

```
Input: (B, C, H, W) + recording_site_ids: (B,)
    ↓
2D CNN Feature Extractor
    ↓
Features: (B, H*W, C)
    ↓
Post-extraction Projection
    ↓
Spatial Embeddings: (B, encoder_embed_dim)
    ↓
Add to Features
    ↓
Transformer Encoder
```

## ⚙️ Configuration Parameters

### Spatial Embedding Parameters

```python
@dataclass
class Wav2Vec2_2DConfig:
    # Spatial embedding parameters
    use_spatial_embedding: bool = True
    num_recording_sites: int = 64
    spatial_embed_dim: int = 256
    spatial_embed_dropout: float = 0.1
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_spatial_embedding` | Enable/disable spatial embeddings | `True` |
| `num_recording_sites` | Number of unique recording sites | `64` |
| `spatial_embed_dim` | Dimension of spatial embeddings | `256` |
| `spatial_embed_dropout` | Dropout rate for spatial embeddings | `0.1` |

## 📝 Usage Examples

### Basic Usage

```python
import torch
from fairseq.models.wav2vec.wav2vec2_2d import Wav2Vec2_2DConfig

# Create configuration
cfg = Wav2Vec2_2DConfig(
    use_spatial_embedding=True,
    num_recording_sites=64,
    spatial_embed_dim=256,
    encoder_embed_dim=768,
    # ... other parameters
)

# Create model
model = cfg.build_model(cfg)

# Prepare input data
batch_size = 4
x = torch.randn(batch_size, 1, 128, 128)  # (B, C, H, W)
recording_site_ids = torch.randint(1, 64, (batch_size,))  # (B,)

# Forward pass with spatial embeddings
output = model(x, recording_site_ids=recording_site_ids, features_only=True)
```

### Without Spatial Embeddings

```python
# Disable spatial embeddings
cfg.use_spatial_embedding = False
model = cfg.build_model(cfg)

# Forward pass without spatial embeddings
output = model(x, features_only=True)
```

### Different Recording Sites

```python
# Test with different recording sites
site_ids = torch.tensor([1, 5, 10, 15])  # Different sites
output = model(x, recording_site_ids=site_ids, features_only=True)
```

## 🔧 Implementation Details

### Spatial Embedding Layer

```python
class Wav2Vec2_2DModel(BaseFairseqModel):
    def __init__(self, cfg):
        # ... other components ...
        
        # Spatial embedding layer
        self.spatial_embedding = None
        if cfg.use_spatial_embedding:
            self.spatial_embedding = nn.Embedding(
                cfg.num_recording_sites, 
                cfg.spatial_embed_dim,
                padding_idx=0  # Use 0 as padding for invalid sites
            )
            self.spatial_embed_dropout = nn.Dropout(cfg.spatial_embed_dropout)
            self.spatial_projection = nn.Linear(
                cfg.spatial_embed_dim, 
                cfg.encoder_embed_dim
            )
```

### Integration in Forward Pass

```python
def forward(self, source, recording_site_ids=None, ...):
    # ... feature extraction ...
    
    # Add spatial embeddings if available
    if self.spatial_embedding is not None and recording_site_ids is not None:
        # Get spatial embeddings
        spatial_embeds = self.spatial_embedding(recording_site_ids)
        spatial_embeds = self.spatial_embed_dropout(spatial_embeds)
        
        # Project to match feature dimension
        spatial_embeds = self.spatial_projection(spatial_embeds)
        
        # Add to features
        spatial_embeds_expanded = spatial_embeds.unsqueeze(1).expand(-1, features.size(1), -1)
        features = features + spatial_embeds_expanded
```

## 🧪 Testing

### Run Tests

```bash
# Test basic functionality
python test_wav2vec2_2d.py

# Test spatial embeddings
python test_wav2vec2_2d_spatial.py
```

### Expected Output

```
🚀 Starting wav2vec2_2d with spatial embeddings tests...

🧪 Testing Spatial Embedding...
🏗️ Model created successfully!
📥 Input shape: torch.Size([4, 1, 128, 128])
📥 Recording site IDs: tensor([12, 45, 23, 8])
📤 Output shape: torch.Size([4, 1024, 256])
📤 Output shape (no spatial): torch.Size([4, 1024, 256])
✅ Spatial embedding test passed!

🧪 Testing Spatial Embedding Analysis...
📍 Spatial embeddings shape: torch.Size([3, 64])
📍 Spatial embeddings for sites tensor([1, 5, 9]):
   Site 1: tensor([0.1234, -0.5678, 0.9012, -0.3456, 0.7890])...
   Site 5: tensor([-0.2345, 0.6789, -0.0123, 0.4567, -0.8901])...
   Site 9: tensor([0.3456, -0.7890, 0.1234, -0.5678, 0.9012])...
📍 Similarity between sites 1 and 5: 0.1234
✅ Spatial embedding analysis test passed!

🎉 All tests passed!
```

## 🎯 Use Cases

### 1. Neural Recording Sites
- **EEG/ECoG**: Different electrode positions
- **fMRI**: Different brain regions
- **MEG**: Different sensor positions

### 2. Multi-Modal Data
- **Audio + Spatial**: Audio recordings from different locations
- **Image + Spatial**: Images from different camera positions
- **Sensor + Spatial**: Sensor data from different locations

### 3. Temporal-Spatial Patterns
- **Time Series**: Temporal patterns that vary by location
- **Spatial Correlations**: Learning spatial relationships between sites

## 🔍 Analysis and Visualization

### Embedding Analysis

```python
# Analyze spatial embeddings
with torch.no_grad():
    spatial_embeds = model.spatial_embedding(torch.arange(64))
    
    # Compute similarities between sites
    similarities = torch.cosine_similarity(
        spatial_embeds.unsqueeze(1), 
        spatial_embeds.unsqueeze(0), 
        dim=2
    )
    
    # Visualize similarity matrix
    import matplotlib.pyplot as plt
    plt.imshow(similarities.numpy())
    plt.colorbar()
    plt.title('Spatial Embedding Similarities')
    plt.show()
```

### Site-Specific Analysis

```python
# Compare outputs for different sites
site_1_output = model(x, recording_site_ids=torch.tensor([1]), features_only=True)
site_2_output = model(x, recording_site_ids=torch.tensor([2]), features_only=True)

# Analyze differences
diff = site_1_output['x'] - site_2_output['x']
print(f"Average difference: {diff.abs().mean():.4f}")
```

## ⚠️ Important Notes

1. **Site ID Range**: Site IDs should be in range `[1, num_recording_sites]`. ID `0` is reserved for padding.

2. **Batch Consistency**: All samples in a batch can have different site IDs.

3. **Gradient Flow**: Spatial embeddings are trained end-to-end with the rest of the model.

4. **Memory Usage**: Spatial embeddings add minimal memory overhead.

5. **Compatibility**: The model works with or without spatial embeddings.

## 🚀 Training Tips

1. **Initialize Embeddings**: Consider initializing spatial embeddings with small random values.

2. **Learning Rate**: Spatial embeddings may need different learning rates than other components.

3. **Regularization**: Use dropout to prevent overfitting on spatial patterns.

4. **Site Balancing**: Ensure training data includes samples from all recording sites.

5. **Validation**: Monitor performance separately for different recording sites. 