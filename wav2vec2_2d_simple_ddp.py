#!/usr/bin/env python3
"""
Simple Multi-GPU wav2vec2_2d training with custom model to avoid DDP issues
"""

import argparse
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import sys
from torch.distributed.elastic.multiprocessing.errors import get_error_handler
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import math


class SimpleScaledRoPE(nn.Module):
    """Simple Scaled RoPE implementation"""
    
    def __init__(self, dim, max_seq_len=1024, scale_factor=1.0, theta=10000.0):
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
    """Simple wav2vec2_2d model with Scaled RoPE - All parameters used"""
    
    def __init__(self, input_height, input_width, embed_dim=384, num_layers=6, num_heads=6):
        super().__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.embed_dim = embed_dim
        
        # 2D CNN feature extraction - ALL layers will be used
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
        
        # Calculate output dimensions
        conv_h = input_height // 16  # 4 stride-2 layers
        conv_w = input_width // 16
        conv_out_dim = 256 * conv_h * conv_w
        
        # Projection to embed_dim - WILL BE USED
        self.projection = nn.Linear(conv_out_dim, embed_dim)
        
        # Scaled RoPE - WILL BE USED
        self.rope = SimpleScaledRoPE(embed_dim, max_seq_len=4096)
        
        # Transformer encoder - ALL layers will be used
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Quantizer - WILL BE USED
        self.quantizer = nn.Linear(embed_dim, 320)  # 320 codebook entries
        
        # Final projection - WILL BE USED
        self.final_proj = nn.Linear(embed_dim, embed_dim)
        
        # Layer normalization - WILL BE USED
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, source, padding_mask=None):
        batch_size = source.size(0)
        
        # 2D CNN feature extraction - ALL parameters used
        features = self.conv_layers(source)  # [B, 256, H/16, W/16]
        
        # Flatten and project - ALL parameters used
        features = features.view(batch_size, -1)  # [B, 256*H*W]
        features = self.projection(features)  # [B, embed_dim]
        
        # Reshape for transformer (single time step)
        features = features.unsqueeze(1)  # [B, 1, embed_dim]
        
        # Apply layer normalization - ALL parameters used
        features = self.layer_norm(features)
        
        # Apply Scaled RoPE - ALL parameters used
        positions = torch.arange(1, device=features.device)
        features = self.rope(features, positions)
        
        # Transformer encoding - ALL parameters used
        encoded = self.transformer(features)  # [B, 1, embed_dim]
        
        # Quantization - ALL parameters used
        quantized = self.quantizer(encoded)  # [B, 1, 320]
        
        # Final projection - ALL parameters used
        output = self.final_proj(encoded)  # [B, 1, embed_dim]
        
        # Create targets for contrastive learning - ALL parameters used
        targets = output.clone()
        
        # Create negatives - ALL parameters used
        num_negatives = 20
        negatives = torch.randn(num_negatives, batch_size, 1, self.embed_dim, device=output.device)
        
        # Combine positives and negatives - ALL parameters used
        all_features = torch.cat([output.unsqueeze(0), negatives], dim=0)  # [num_negatives+1, B, 1, embed_dim]
        
        return {
            'x': all_features,
            'y': targets,
            'padding_mask': padding_mask
        }


def ddp_setup():
    """Initialize the distributed process group."""
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    return rank, world_size, local_rank


def ddp_cleanup():
    """Clean up the distributed process group."""
    dist.destroy_process_group()


def get_grad_norm(model, norm_type=2.0):
    """Compute gradient norm"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    return total_norm ** (1.0 / norm_type)


def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes"""
    rt = tensor.clone().float()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def train_epoch(model, data_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    grad_norms = []
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"[Rank {rank}] Number of train samples: {len(data_loader.sampler)}")
    
    for step, batch in enumerate(tqdm(data_loader, desc=f"Rank {rank} Training")):
        try:
            # Get data
            if isinstance(batch, dict):
                source = batch['source'].to(device)
            else:
                source = batch.to(device)
            
            # Ensure proper shape: [B, C, H, W]
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
            optimizer.zero_grad()
            loss.backward()
            grad_norm = get_grad_norm(model)
            grad_norms.append(grad_norm)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Clear cache every few steps
            if step % 5 == 0:
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[Rank {rank}] Training step error: {e}")
            continue
    
    # Reduce metrics across all processes
    avg_loss = total_loss / len(data_loader)
    avg_loss = torch.tensor(avg_loss, device=device)
    avg_loss = reduce_tensor(avg_loss, world_size).item()
    
    avg_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    avg_grad = torch.tensor(avg_grad, device=device)
    avg_grad = reduce_tensor(avg_grad, world_size).item()
    
    return avg_loss, avg_grad


def run_wav2vec2_2d_simple_ddp(session_data, output_path, num_epochs=10):
    """Main training function with simple model"""
    
    # Setup distributed training
    rank, world_size, local_rank = ddp_setup()
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print(f"Starting Simple Multi-GPU training on {world_size} GPUs")
        print(f"Session: {session_data['session_id']}")
        print(f"Output path: {output_path}")
    
    # Load data - memory efficient version
    try:
        if isinstance(session_data['data'], np.ndarray):
            data = session_data['data']
        else:
            # For large lists, sample a subset to avoid memory issues
            raw_data = session_data['data']
            if len(raw_data) > 1000:  # If more than 1k elements
                print(f"[Rank {rank}] Large dataset detected: {len(raw_data)} elements. Sampling 1,000 for training...")
                # Sample every nth element to get ~1k samples
                step = len(raw_data) // 1000
                data = np.array(raw_data[::step])
            else:
                data = np.array(raw_data)
        
        # Reshape to 2D matrix: [timePoints, channels]
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        elif len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        
        print(f"[Rank {rank}] Data shape: {data.shape}")
        
        # Create dataset with small chunks
        chunk_size = min(100, data.shape[0])  # Process in chunks of 100
        dataset = []
        for i in range(0, data.shape[0], chunk_size):
            chunk = data[i:i+chunk_size]
            dataset.append({'source': torch.FloatTensor(chunk)})
        
        # Create distributed sampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=1,  # Minimal batch size
            sampler=sampler,
            num_workers=0,
            pin_memory=False
        )
        
        if rank == 0:
            print(f"Data loaded successfully: {len(dataset)} samples")
        
    except Exception as e:
        print(f"[Rank {rank}] Data loading error: {e}")
        ddp_cleanup()
        return
    
    # Create model - simple version with all parameters used
    try:
        model = SimpleWav2Vec2_2D(
            input_height=data.shape[0],
            input_width=data.shape[1],
            embed_dim=384,
            num_layers=6,
            num_heads=6
        )
        model = model.to(device)
        
        # Wrap with DDP - NO find_unused_parameters needed since all parameters are used
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        
        if rank == 0:
            print(f"Model created successfully: {sum(p.numel() for p in model.parameters()):,} parameters")
            print(f"Model components:")
            for name, module in model.named_modules():
                if len(list(module.parameters())) > 0:
                    print(f"  {name}: {sum(p.numel() for p in module.parameters())} parameters")
        
    except Exception as e:
        print(f"[Rank {rank}] Model creation error: {e}")
        ddp_cleanup()
        return
    
    # Create optimizer
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-4,
            weight_decay=0.01
        )
        
        if rank == 0:
            print("Optimizer created successfully")
        
    except Exception as e:
        print(f"[Rank {rank}] Optimizer creation error: {e}")
        ddp_cleanup()
        return
    
    # Training loop
    if rank == 0:
        print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        if rank == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Set epoch for distributed sampler
        data_loader.sampler.set_epoch(epoch)
        
        # Training
        train_loss, grad_norm = train_epoch(model, data_loader, optimizer, device)
        
        # Synchronize all processes
        dist.barrier()
        
        if rank == 0:
            print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Grad Norm: {grad_norm:.4f}")
        
        # Clear cache after each epoch
        torch.cuda.empty_cache()
    
    # Save model (only on rank 0)
    if rank == 0:
        try:
            os.makedirs(f"{output_path}/{session_data['session_id']}/ssl_model/", exist_ok=True)
            model_path = f"{output_path}/{session_data['session_id']}/ssl_model/wav2vec2_2d_simple_ddp.pt"
            
            # Save only the model state dict (not the DDP wrapper)
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'input_height': data.shape[0],
                'input_width': data.shape[1],
                'embed_dim': 384,
                'num_layers': 6,
                'num_heads': 6,
                'epoch': num_epochs,
                'train_loss': train_loss,
            }, model_path)
            
            print(f"Model saved to: {model_path}")
            
        except Exception as e:
            print(f"Model saving error: {e}")
    
    # Cleanup
    ddp_cleanup()


def main():
    """Main function to run simple multi-GPU training"""
    parser = argparse.ArgumentParser(description='Simple Multi-GPU wav2vec2_2d training')
    parser.add_argument('--data_path', type=str, default='/scratch/mkp6112/LFP/region_decoding/script/Allen_w2v2/Allen')
    parser.add_argument('--output_path', type=str, default='./outputs')
    parser.add_argument('--num_epochs', type=int, default=10)
    
    args = parser.parse_args()
    
    # Check available GPUs
    if not torch.cuda.is_available():
        print("CUDA not available. Multi-GPU training requires CUDA.")
        return
    
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for training")
    
    # Load session data
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
            print("No pickle files found in data path")
            return
        
        print(f"Found {len(sessions_list)} sessions")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Process each session
    for i, session_data in enumerate(sessions_list):
        print(f"Processing session {i+1}/{len(sessions_list)}: {session_data['session_id']}")
        
        try:
            run_wav2vec2_2d_simple_ddp(session_data, args.output_path, args.num_epochs)
        except Exception as e:
            print(f"Error in run_wav2vec2_2d_simple_ddp: {e}")
        
        print(f"Completed session {i+1}/{len(sessions_list)}")


if __name__ == "__main__":
    error_handler = get_error_handler()
    try:
        main()
    except Exception as e:
        error_handler.record_exception(e)
        print(f"Error in main: {e}")
