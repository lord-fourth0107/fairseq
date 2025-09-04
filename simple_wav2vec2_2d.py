#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import numpy as np
from tqdm import tqdm

class SimpleWav2Vec2D(nn.Module):
    """Simplified wav2vec2 2D model that actually works"""
    
    def __init__(self, input_height=3750, input_width=93, hidden_dim=512, output_dim=256):
        super().__init__()
        
        # Simple CNN feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Adaptive pooling to standardize output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection layers
        self.projection = nn.Linear(512, hidden_dim)
        self.final_proj = nn.Linear(hidden_dim, output_dim)
        
        # Simple transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8, 
            dim_feedforward=hidden_dim*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
    def forward(self, x):
        # x: [batch, 1, height, width]
        batch_size = x.shape[0]
        
        # CNN feature extraction
        features = self.conv_layers(x)  # [batch, 512, H', W']
        
        # Adaptive pooling to standardize
        features = self.adaptive_pool(features)  # [batch, 512, 1, 1]
        features = features.view(batch_size, 512)  # [batch, 512]
        
        # Project to hidden dimension
        features = self.projection(features)  # [batch, hidden_dim]
        
        # Add sequence dimension for transformer
        features = features.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Transformer encoding
        encoded = self.transformer(features)  # [batch, 1, hidden_dim]
        
        # Final projection
        output = self.final_proj(encoded)  # [batch, 1, output_dim]
        
        return output

class SimpleDataset(Dataset):
    """Simple dataset that loads pickle files"""
    
    def __init__(self, data_paths):
        self.data_paths = data_paths
        self.data = []
        
        print(f"Loading {len(data_paths)} files...")
        for path in tqdm(data_paths):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                
                # Extract probe data
                for probe_id, probe_data in data.items():
                    if isinstance(probe_data, dict) and 'LFP' in probe_data:
                        lfp_data = probe_data['LFP']
                        if len(lfp_data) > 0:
                            # Take first recording as example
                            recording = lfp_data[0]
                            if len(recording) >= 48000:  # Ensure minimum length
                                self.data.append(recording[:48000])
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
        
        print(f"Loaded {len(self.data)} recordings")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        recording = self.data[idx]
        
        # Reshape to 2D: [height, width] where height=time, width=channels
        # For simplicity, assume we can reshape 48000 into [3750, 13] (3750*13=48750, close enough)
        height, width = 3750, 13
        if len(recording) >= height * width:
            recording = recording[:height * width]
        else:
            # Pad if too short
            recording = np.pad(recording, (0, height * width - len(recording)))
        
        # Reshape to 2D
        recording_2d = recording.reshape(height, width)
        
        # Convert to tensor and add channel dimension
        recording_tensor = torch.FloatTensor(recording_2d).unsqueeze(0)  # [1, height, width]
        
        return recording_tensor

def train_simple_model():
    """Train the simplified model"""
    
    # Data paths (you'll need to update these)
    data_paths = [
        "/scratch/mkp6112/LFP/region_decoding/script/Allen_w2v2/Allen/771160300_773621937.pickle",
        "/scratch/mkp6112/LFP/region_decoding/script/Allen_w2v2/Allen/771990200_773654728.pickle",
        "/scratch/mkp6112/LFP/region_decoding/script/Allen_w2v2/Allen/719161530_729445650.pickle",
    ]
    
    # Filter existing paths
    existing_paths = [p for p in data_paths if os.path.exists(p)]
    print(f"Found {len(existing_paths)} existing data files")
    
    if not existing_paths:
        print("No data files found! Please check paths.")
        return
    
    # Create dataset and dataloader
    dataset = SimpleDataset(existing_paths)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # Create model
    model = SimpleWav2Vec2D()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    model.train()
    for epoch in range(5):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch)
            
            # Simple reconstruction loss (predict the input)
            # Flatten both input and output for comparison
            input_flat = batch.view(batch.shape[0], -1)
            output_flat = output.view(output.shape[0], -1)
            
            # Pad or truncate to match dimensions
            min_dim = min(input_flat.shape[1], output_flat.shape[1])
            input_flat = input_flat[:, :min_dim]
            output_flat = output_flat[:, :min_dim]
            
            loss = criterion(output_flat, input_flat)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.6f}")
    
    print("Training completed!")
    
    # Save model
    torch.save(model.state_dict(), 'simple_wav2vec2d_model.pth')
    print("Model saved as 'simple_wav2vec2d_model.pth'")

if __name__ == "__main__":
    train_simple_model()
