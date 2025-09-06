#!/usr/bin/env python3
"""
Final Multi-GPU Wav2Vec2 2D Training Script
Uses the actual fairseq model with working distributed training
"""

import os
import sys
import pickle
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
from tqdm import tqdm
import time

# Add fairseq to path
sys.path.insert(0, '/vast/us2193/fairseq')

# Import fairseq components
from fairseq.models.wav2vec.wav2vec2_2d import Wav2Vec2_2DConfig, Wav2Vec2_2DModel

class ModifiedSessionDataset(Dataset):
    """Dataset that handles tuple format data"""
    
    def __init__(self, data_path, max_samples=1000, chunk_size=100):
        self.data_path = data_path
        self.max_samples = max_samples
        self.chunk_size = chunk_size
        self.data = []
        self._load_data()
    
    def _load_data(self):
        """Load and process data from pickle files"""
        print("Loading data...")
        
        # Get all pickle files
        files = [f for f in os.listdir(self.data_path) if f.endswith('.pickle')]
        print(f"Found {len(files)} pickle files")
        
        # Load first file for now
        if files:
            filepath = os.path.join(self.data_path, files[0])
            print(f"Loading: {files[0]}")
            
            with open(filepath, 'rb') as f:
                raw_data = pickle.load(f)
            
            print(f"Raw data type: {type(raw_data)}")
            print(f"Raw data length: {len(raw_data)}")
            
            if isinstance(raw_data, list) and len(raw_data) > 0:
                # Sample data
                step = max(1, len(raw_data) // self.max_samples)
                sampled_data = raw_data[::step]
                print(f"Sampled {len(sampled_data)} elements")
                
                # Handle tuple format
                if isinstance(sampled_data[0], tuple):
                    print("Processing tuple format data...")
                    # Extract first element from each tuple
                    first_elements = [item[0] for item in sampled_data if len(item) > 0]
                    print(f"Extracted {len(first_elements)} first elements")
                    
                    if first_elements and hasattr(first_elements[0], 'shape'):
                        print(f"First element shape: {first_elements[0].shape}")
                        # Convert to numpy and then to list of tensors
                        np_data = np.array(first_elements)
                        print(f"Numpy data shape: {np_data.shape}")
                        
                        # Convert to list of tensors
                        self.data = [torch.FloatTensor(arr) for arr in np_data]
                        print(f"Converted to {len(self.data)} tensors")
                    else:
                        print("No valid array data found in tuples")
                        self.data = []
                else:
                    # Direct conversion if not tuples
                    np_data = np.array(sampled_data)
                    self.data = [torch.FloatTensor(arr) for arr in np_data]
                    print(f"Direct conversion to {len(self.data)} tensors")
        
        print(f"Final dataset size: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if idx >= len(self.data):
            idx = idx % len(self.data)
        return self.data[idx]

def ddp_setup(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

def train_epoch(model, dataloader, optimizer, device, rank):
    """Train one epoch with wav2vec2 model"""
    model.train()
    total_loss = 0.0
    total_contrastive = 0.0
    total_diversity = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Rank {rank} Training", disable=rank != 0)
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Move to device
            batch = batch.to(device)
            
            # Reshape data for 2D CNN: [batch_size, 1, height, width]
            # Single GPU uses [batch_size, 3750] -> [batch_size, 1, 3750, 93]
            batch_size = batch.size(0)
            if batch.size(1) == 3750:
                # Reshape to 2D: [batch_size, 1, 3750, 93] to match single GPU
                # We need to pad or repeat to get 93 channels
                if batch.size(1) == 3750:
                    # Repeat the 3750-dim vector 93 times to create 93 channels
                    batch = batch.unsqueeze(1).repeat(1, 1, 1, 93)  # [batch_size, 1, 3750, 93]
            
            # Forward pass
            result = model(batch)
            
            # Extract losses
            if isinstance(result, dict):
                loss = result.get('loss', torch.tensor(0.0, device=device))
                contrastive_loss = result.get('contrastive_loss', torch.tensor(0.0, device=device))
                diversity_loss = result.get('diversity_loss', torch.tensor(0.0, device=device))
            else:
                # Fallback if result is not a dict
                loss = torch.tensor(0.0, device=device)
                contrastive_loss = torch.tensor(0.0, device=device)
                diversity_loss = torch.tensor(0.0, device=device)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_contrastive += contrastive_loss.item()
            total_diversity += diversity_loss.item()
            num_batches += 1
            
            # Update progress bar
            if rank == 0:
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "contrastive": f"{contrastive_loss.item():.4f}",
                    "diversity": f"{diversity_loss.item():.4f}"
                })
            
            # Memory cleanup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Rank {rank} - Batch {batch_idx} error: {e}")
            continue
    
    return {
        'total_loss': total_loss / max(num_batches, 1),
        'contrastive_loss': total_contrastive / max(num_batches, 1),
        'diversity_loss': total_diversity / max(num_batches, 1)
    }

def main_worker(rank, world_size, args):
    """Main worker function for distributed training"""
    print(f"Starting worker {rank} of {world_size}")
    
    # Setup distributed training
    ddp_setup(rank, world_size)
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Create dataset
    dataset = ModifiedSessionDataset(
        data_path=args.data_path,
        max_samples=args.max_samples,
        chunk_size=args.chunk_size
    )
    
    if len(dataset) == 0:
        print(f"Rank {rank}: No data loaded, exiting")
        ddp_cleanup()
        return
    
    # Create dataloader with distributed sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    
    # Create Wav2Vec2 2D model - EXACT SAME CONFIG as single GPU
    config = Wav2Vec2_2DConfig(
        # 2D CNN feature extraction layers
        conv_2d_feature_layers="[(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)]",
        
        # Input dimensions
        input_channels=1,
        input_height=3750,  # Time points (height dimension)
        input_width=93,     # Channels (width dimension)
        
        # Transformer parameters - MUCH SMALLER
        encoder_layers=6,  # Reduced from 12
        encoder_embed_dim=384,  # Reduced from 768
        encoder_ffn_embed_dim=1536,  # Reduced from 3072
        encoder_attention_heads=6,  # Reduced from 12
        activation_fn="gelu",
        
        # Scaled RoPE parameters (replaces spatial embeddings)
        use_scaled_rope=True,
        rope_max_seq_len=4096,
        rope_scale_factor=1.0,
        rope_theta=10000.0,
        
        # Legacy spatial embedding parameters (disabled)
        use_spatial_embedding=False,
        
        # Masking parameters
        mask_prob=0.15,  # Reduced from 0.2
        mask_length=5,  # Reduced from 10
        mask_selection="static",
        
        # Other parameters
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        layer_norm_first=False,
        feature_grad_mult=1.0,
        conv_bias=False,
        extractor_mode="default",
        
        # Adaptive pooling after flattening to standardize dimensions
        flattened_pool_dim=256,  # Reduced from 512

        # 1D CNN to create multiple time steps after adaptive pooling
        temporal_conv1d_enabled=True,  # Enable 1D CNN for temporal expansion
        temporal_steps=50,  # Reduced from 100

        # Re-enable negative sampling now that we have multiple time steps
        num_negatives=20,  # Reduced from 100
        cross_sample_negatives=5,  # Reduced from 10
        codebook_negatives=10,  # Enable codebook negatives for quantization
        
        # Quantization parameters
        quantizer_depth=2,
        quantizer_factor=3,
        latent_vars=320,  # This is quantizer_k
        latent_groups=2,  # Number of codebook groups
        same_quantizer=True,
    )
    
    model = Wav2Vec2_2DModel(config).to(device)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank])
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print(f"Rank {rank}: Starting training for {args.num_epochs} epochs")
    
    for epoch in range(args.num_epochs):
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Set epoch for sampler
        sampler.set_epoch(epoch)
        
        # Train epoch
        losses = train_epoch(model, dataloader, optimizer, device, rank)
        
        if rank == 0:
            print(f"Epoch {epoch + 1} - Total Loss: {losses['total_loss']:.4f}, "
                  f"Contrastive: {losses['contrastive_loss']:.4f}, "
                  f"Diversity: {losses['diversity_loss']:.4f}")
    
    # Cleanup
    ddp_cleanup()
    print(f"Rank {rank}: Training completed")

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Wav2Vec2 2D Training')
    parser.add_argument('--data_path', type=str, 
                       default='/scratch/mkp6112/LFP/region_decoding/script/Allen_w2v2/Allen',
                       help='Path to data directory')
    parser.add_argument('--output_path', type=str, default='./output',
                       help='Path to save outputs')
    parser.add_argument('--num_epochs', type=int, default=2,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum samples to load')
    parser.add_argument('--chunk_size', type=int, default=100,
                       help='Chunk size for data processing')
    parser.add_argument('--input_dim', type=int, default=3750,
                       help='Input dimension')
    parser.add_argument('--embed_dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--world_size', type=int, default=1,
                       help='Number of GPUs to use')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    print(f"Starting multi-GPU training with {args.world_size} GPUs")
    print(f"Data path: {args.data_path}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Max samples: {args.max_samples}")
    
    # Launch distributed training
    mp.spawn(main_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)

if __name__ == "__main__":
    main()
