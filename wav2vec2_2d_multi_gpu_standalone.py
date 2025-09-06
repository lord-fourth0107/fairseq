#!/usr/bin/env python3
"""
Standalone Multi-GPU training script for wav2vec2_2d with Scaled RoPE
No fairseq dependencies - uses PyTorch directly
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import math


class ScaledRoPE(nn.Module):
    """Scaled Rotary Position Embedding"""
    
    def __init__(self, dim, max_seq_len=4096, scale_factor=1.0, theta=10000.0):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for RoPE"
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.scale_factor = scale_factor
        self.theta = theta
        
        # Precompute frequency matrix
        self.register_buffer('freqs', self._compute_freqs())
        
    def _compute_freqs(self):
        positions = torch.arange(self.max_seq_len, dtype=torch.float32)
        dim_indices = torch.arange(0, self.dim, 2, dtype=torch.float32)
        freqs = 1.0 / (self.theta ** (dim_indices / self.dim))
        freqs = positions.unsqueeze(1) * freqs.unsqueeze(0)
        freqs = freqs * self.scale_factor
        return freqs
    
    def forward(self, x, positions=None):
        batch_size, seq_len, dim = x.shape
        
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
        
        positions = positions.clamp(0, self.max_seq_len - 1)
        freqs = self.freqs[positions]
        freqs = freqs.unsqueeze(0).expand(batch_size, -1, -1)
        
        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)
        
        x_reshaped = x.view(batch_size, seq_len, dim // 2, 2)
        x_rotated = torch.stack([
            x_reshaped[..., 0] * cos_freqs - x_reshaped[..., 1] * sin_freqs,
            x_reshaped[..., 0] * sin_freqs + x_reshaped[..., 1] * cos_freqs
        ], dim=-1)
        
        return x_rotated.view(batch_size, seq_len, dim)


class SimpleWav2Vec2_2D(nn.Module):
    """Simplified wav2vec2_2d model with Scaled RoPE"""
    
    def __init__(self, input_height, input_width, embed_dim=384, num_layers=6, num_heads=6):
        super().__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.embed_dim = embed_dim
        
        # 2D CNN feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Calculate output dimensions after conv layers
        conv_h = input_height // 16  # 4 stride-2 layers
        conv_w = input_width // 16
        conv_out_dim = 256 * conv_h * conv_w
        
        # Projection to embed_dim
        self.projection = nn.Linear(conv_out_dim, embed_dim)
        
        # Scaled RoPE
        self.rope = ScaledRoPE(embed_dim, max_seq_len=4096)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Quantizer (simplified)
        self.quantizer = nn.Linear(embed_dim, 320)  # 320 codebook entries
        
        # Final projection
        self.final_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, source, padding_mask=None):
        batch_size = source.size(0)
        
        # 2D CNN feature extraction
        features = self.conv_layers(source)  # [B, 256, H/16, W/16]
        
        # Flatten and project
        features = features.view(batch_size, -1)  # [B, 256*H*W]
        features = self.projection(features)  # [B, embed_dim]
        
        # Reshape for transformer (single time step)
        features = features.unsqueeze(1)  # [B, 1, embed_dim]
        
        # Apply Scaled RoPE
        positions = torch.arange(1, device=features.device)
        features = self.rope(features, positions)
        
        # Transformer encoding
        encoded = self.transformer(features)  # [B, 1, embed_dim]
        
        # Quantization
        quantized = self.quantizer(encoded)  # [B, 1, 320]
        
        # Final projection
        output = self.final_proj(encoded)  # [B, 1, embed_dim]
        
        # Create dummy targets for contrastive learning
        targets = output.clone()
        
        # Create dummy negatives
        num_negatives = 20
        negatives = torch.randn(num_negatives, batch_size, 1, self.embed_dim, device=output.device)
        
        # Combine positives and negatives
        all_features = torch.cat([output.unsqueeze(0), negatives], dim=0)  # [num_negatives+1, B, 1, embed_dim]
        
        return {
            'x': all_features,
            'y': targets,
            'padding_mask': padding_mask
        }


def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


def get_grad_norm(model):
    """Compute gradient norm"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def train_epoch(model, data_loader, optimizer, device, rank, accumulation_steps=4):
    """Train for one epoch"""
    model.train()
    losses = []
    grad_norms = []
    scaler = torch.cuda.amp.GradScaler()
    
    if rank == 0:
        progress = tqdm(data_loader, desc="Training")
    else:
        progress = data_loader
    
    for step, batch in enumerate(progress):
        try:
            # Get data
            if isinstance(batch, dict):
                source = batch['source'].to(device)
            else:
                source = batch.to(device)
            
            if len(source.shape) == 3:
                source = source.unsqueeze(1)
            
            # Forward pass
            outputs = model(source=source, padding_mask=None)
            
            # Compute contrastive loss
            x = outputs['x']  # [num_negatives+1, B, 1, embed_dim]
            y = outputs['y']  # [B, 1, embed_dim]
            
            # Reshape for loss computation
            x_flat = x.view(x.size(0), -1)  # [num_negatives+1, B*embed_dim]
            y_flat = y.view(-1)  # [B*embed_dim]
            
            # Compute cosine similarity
            logits = torch.cosine_similarity(x_flat.unsqueeze(1), y_flat.unsqueeze(0), dim=-1)
            targets = torch.zeros(logits.size(1), dtype=torch.long, device=logits.device)
            
            loss = F.cross_entropy(logits, targets)
            
            # Backward pass
            scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
            
            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = get_grad_norm(model)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                grad_norms.append(grad_norm)
            
            losses.append(loss.item())
            
            if rank == 0 and step % 10 == 0:
                avg_loss = np.mean(losses[-10:]) if losses else 0.0
                progress.set_postfix({"loss": f"{avg_loss:.4f}"})
                
        except Exception as e:
            if rank == 0:
                print(f"Training error: {e}")
            continue
    
    dist.barrier()
    
    if rank == 0:
        avg_loss = np.mean(losses) if losses else 0.0
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
        return avg_loss, avg_grad_norm
    else:
        return 0.0, 0.0


def run_training(rank, world_size, session_data, output_path, num_epochs=10):
    """Main training function"""
    
    # Setup
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"Starting Multi-GPU training on {world_size} GPUs")
        print(f"Session: {session_data['session_id']}")
    
    # Load data
    try:
        if isinstance(session_data['data'], np.ndarray):
            data = session_data['data']
        else:
            data = np.array(session_data['data'])
        
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        elif len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        
        dataset = [{'source': torch.FloatTensor(data)}]
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        data_loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True)
        
        if rank == 0:
            print(f"Data loaded: {data.shape}")
        
    except Exception as e:
        print(f"Data loading error: {e}")
        cleanup_distributed()
        return
    
    # Create model
    try:
        model = SimpleWav2Vec2_2D(
            input_height=data.shape[0],
            input_width=data.shape[1],
            embed_dim=768,  # Full size for A100
            num_layers=12,  # Full transformer
            num_heads=12    # Full attention
        )
        model = model.to(device)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        
        if rank == 0:
            print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
    except Exception as e:
        print(f"Model creation error: {e}")
        cleanup_distributed()
        return
    
    # Create optimizer
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        if rank == 0:
            print("Optimizer created")
    except Exception as e:
        print(f"Optimizer error: {e}")
        cleanup_distributed()
        return
    
    # Training loop
    if rank == 0:
        print(f"Starting {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        if rank == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
        
        sampler.set_epoch(epoch)
        train_loss, grad_norm = train_epoch(model, data_loader, optimizer, device, rank)
        
        if rank == 0:
            print(f"Epoch {epoch + 1} - Loss: {train_loss:.4f}, Grad: {grad_norm:.4f}")
    
    # Save model
    if rank == 0:
        try:
            os.makedirs(f"{output_path}/{session_data['session_id']}/ssl_model/", exist_ok=True)
            model_path = f"{output_path}/{session_data['session_id']}/ssl_model/wav2vec2_2d_multi_gpu.pt"
            
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'input_height': data.shape[0],
                'input_width': data.shape[1],
                'embed_dim': 384,
                'epoch': num_epochs,
                'train_loss': train_loss,
            }, model_path)
            
            print(f"Model saved: {model_path}")
        except Exception as e:
            print(f"Save error: {e}")
    
    cleanup_distributed()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Multi-GPU wav2vec2_2d training')
    parser.add_argument('--data_path', type=str, default='/scratch/mkp6112/LFP/region_decoding/data/Allen/data/')
    parser.add_argument('--output_path', type=str, default='./outputs')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count())
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    world_size = min(args.world_size, torch.cuda.device_count())
    print(f"Using {world_size} GPUs")
    
    # Load data
    try:
        sessions_list = []
        for filename in os.listdir(args.data_path):
            if filename.endswith('.pickle'):
                filepath = os.path.join(args.data_path, filename)
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                session_id = filename.replace('.pickle', '')
                sessions_list.append({
                    'session_id': session_id,
                    'data': data,
                    'filepath': filepath
                })
        
        if not sessions_list:
            print("No pickle files found")
            return
        
        print(f"Found {len(sessions_list)} sessions")
        
    except Exception as e:
        print(f"Data loading error: {e}")
        return
    
    # Process sessions
    for i, session_data in enumerate(sessions_list):
        print(f"Processing session {i+1}/{len(sessions_list)}: {session_data['session_id']}")
        
        mp.spawn(
            run_training,
            args=(world_size, session_data, args.output_path, args.num_epochs),
            nprocs=world_size,
            join=True
        )
        
        print(f"Completed session {i+1}/{len(sessions_list)}")


if __name__ == "__main__":
    main()
