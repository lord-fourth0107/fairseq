#!/usr/bin/env python3
"""
Simple Multi-GPU version based on working single GPU script
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Add fairseq to path - same as single GPU script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fairseq.models.wav2vec.wav2vec2_2d import Wav2Vec2_2DConfig, Wav2Vec2_2DModel
from fairseq.optim.adam import FairseqAdam
import torch.nn.functional as F


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
    """Train for one epoch - based on single GPU version"""
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
            # Get data - same as single GPU version
            if isinstance(batch, dict):
                source = batch['source'].to(device)
            else:
                source = batch.to(device)
            
            # Ensure proper shape: [B, C, H, W] - same as single GPU
            if len(source.shape) == 3:
                source = source.unsqueeze(1)  # Add channel dimension
            
            # Forward pass - same as single GPU
            outputs = model(source=source, padding_mask=None)
            
            # Extract loss - same logic as single GPU
            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            else:
                # Manual loss computation for wav2vec2_2d
                x = outputs['x']  # [num_negatives + 1, batch_size, seq_len]
                y = outputs.get('y', x)  # Target features
                
                # Compute contrastive loss
                logits = torch.cosine_similarity(x.unsqueeze(0), y.unsqueeze(1), dim=-1)
                targets = torch.zeros(logits.size(1), dtype=torch.long, device=logits.device)
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                
                contrastive_loss = F.cross_entropy(logits_flat, targets_flat)
                
                # Compute diversity loss
                diversity_loss = 0.0
                if 'prob_perplexity' in outputs and outputs['prob_perplexity'] is not None:
                    prob_perplexity = outputs['prob_perplexity']
                    diversity_loss = -prob_perplexity
                
                loss = contrastive_loss + 0.1 * diversity_loss
            
            # Backward pass with gradient accumulation
            try:
                scaled_loss = loss / accumulation_steps
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                if (step + 1) % accumulation_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        grad_norm = get_grad_norm(model)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = get_grad_norm(model)
                        optimizer.step()
                    
                    grad_norms.append(grad_norm)
                    optimizer.zero_grad()
                
                # Store losses
                losses.append(loss.item())
                
            except Exception as e:
                if rank == 0:
                    print(f"Backward pass error: {e}")
                continue
            
            # Update progress bar (only on rank 0)
            if rank == 0 and step % 10 == 0:
                avg_loss = np.mean(losses[-10:]) if losses else 0.0
                progress.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "grad": f"{grad_norm:.2f}"
                })
                
        except Exception as e:
            if rank == 0:
                print(f"Training step error: {e}")
            continue
    
    # Synchronize across all processes
    dist.barrier()
    
    # Average metrics across all processes
    if rank == 0:
        avg_loss = np.mean(losses) if losses else 0.0
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
        print(f"Multi-GPU Training completed - Avg Loss: {avg_loss:.4f}, Avg Grad Norm: {avg_grad_norm:.4f}")
        return avg_loss, avg_grad_norm
    else:
        return 0.0, 0.0


def run_training(rank, world_size, session_data, output_path, num_epochs=10):
    """Main training function - based on single GPU version"""
    
    # Setup
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"Starting Multi-GPU training on {world_size} GPUs")
        print(f"Session: {session_data['session_id']}")
    
    # Load data - memory efficient version for large datasets
    try:
        if isinstance(session_data['data'], np.ndarray):
            data = session_data['data']
        else:
            # For large lists, sample a very small subset to avoid memory issues
            raw_data = session_data['data']
            if len(raw_data) > 1000:  # If more than 1k elements
                print(f"Large dataset detected: {len(raw_data)} elements. Sampling 1,000 for training...")
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
        
        print(f"Data shape: {data.shape}")
        
        # Create dataset with very small chunks to avoid memory issues
        chunk_size = min(100, data.shape[0])  # Process in chunks of 100
        dataset = []
        for i in range(0, data.shape[0], chunk_size):
            chunk = data[i:i+chunk_size]
            dataset.append({'source': torch.FloatTensor(chunk)})
        
        # Create distributed sampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        
        # Create data loader - minimal batch size for memory efficiency
        data_loader = DataLoader(
            dataset,
            batch_size=1,  # Minimal batch size
            sampler=sampler,
            num_workers=0,
            pin_memory=False  # Disable pin_memory for large data
        )
        
        if rank == 0:
            print(f"Data loaded successfully: {len(dataset)} samples")
            print(f"Data shape: {data.shape}")
        
    except Exception as e:
        print(f"Data loading error: {e}")
        cleanup_distributed()
        return
    
    # Model configuration - same as single GPU version
    use_spatial_embedding = False  # Disable spatial embedding (deprecated)
    use_scaled_rope = True  # Enable Scaled RoPE for positional encoding
    
    w2v2_2d_config = Wav2Vec2_2DConfig(
        # 2D CNN specific parameters for [3750 × 93] input
        conv_2d_feature_layers="[(32, 3, 2), (64, 3, 2), (128, 3, 2), (256, 3, 2)]",  # Same as single GPU
        conv_2d_stride=2,
        conv_2d_kernel_size=3,
        input_width=data.shape[1],  # Number of channels
        input_height=data.shape[0],  # Time points
        
        # Encoder parameters - same as single GPU
        encoder_layers=6,
        encoder_embed_dim=384,
        encoder_ffn_embed_dim=1536,
        encoder_attention_heads=6,
        activation_fn="gelu",
        
        # Scaled RoPE parameters
        use_scaled_rope=use_scaled_rope,
        rope_max_seq_len=4096,
        rope_scale_factor=1.0,
        rope_theta=10000.0,
        
        # Legacy spatial embedding parameters (disabled)
        use_spatial_embedding=use_spatial_embedding,
        
        # Masking parameters - same as single GPU
        mask_prob=0.15,
        mask_length=5,
        mask_selection="static",
        mask_other=0.0,
        mask_min_space=1,
        mask_channel_prob=0.0,
        mask_channel_other=0.0,
        mask_channel_min_space=1,
        mask_channel_length=64,
        no_mask_channel_overlap=False,
        mask_channel_selection="static",
        
        # Feature extractor parameters - same as single GPU
        feature_grad_mult=0.0,
        layer_norm=True,
        layerdrop=0.1,
        activation_dropout=0.0,
        dropout=0.1,
        attention_dropout=0.1,
        encoder_layerdrop=0.05,
        dropout_input=0.1,
        dropout_features=0.1,
        
        # Quantization parameters - same as single GPU
        quantizer_depth=2,
        quantizer_factor=3,
        latent_vars=320,
        latent_groups=2,
        same_quantizer=False,
        codebook_negatives=10,
        
        # Negative sampling parameters - same as single GPU
        num_negatives=20,
        cross_sample_negatives=5,
        
        # Temporal parameters - same as single GPU
        temporal_conv1d_enabled=True,
        temporal_steps=50,
        
        # Pooling parameters - same as single GPU
        flattened_pool_dim=256,
        
        # Logging - same as single GPU
        log_compression=False,
        log_temp=0.0,
        target_glu=False,
        feature_pen=0.0,
        prob_ppl_weight=0.0,
        infonce=False,
        loss_weights=[0.1, 10.0],
        log_keys=["prob_perplexity", "code_perplexity", "temp"],
        diversity_loss_weight=0.1,
    )
    
    # Create model - same as single GPU
    try:
        model = Wav2Vec2_2DModel(w2v2_2d_config)
        model = model.to(device)
        
        # Wrap with DDP
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        
        if rank == 0:
            print(f"Model created successfully on GPU {rank}")
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"Model creation error: {e}")
        cleanup_distributed()
        return
    
    # Create optimizer - same as single GPU
    try:
        optimizer = FairseqAdam(
            model.parameters(),
            lr=5e-4,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.01
        )
        
        if rank == 0:
            print(f"Optimizer created successfully")
        
    except Exception as e:
        print(f"Optimizer creation error: {e}")
        cleanup_distributed()
        return
    
    # Training loop - same as single GPU
    if rank == 0:
        print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        if rank == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Set epoch for distributed sampler
        sampler.set_epoch(epoch)
        
        # Training
        train_loss, grad_norm = train_epoch(
            model, data_loader, optimizer, device, rank, accumulation_steps=4
        )
        
        if rank == 0:
            print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Grad Norm: {grad_norm:.4f}")
    
    # Save model (only on rank 0)
    if rank == 0:
        try:
            os.makedirs(f"{output_path}/{session_data['session_id']}/ssl_model/", exist_ok=True)
            model_path = f"{output_path}/{session_data['session_id']}/ssl_model/wav2vec2_2d_multi_gpu.pt"
            
            # Save only the model state dict (not the DDP wrapper)
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'config': w2v2_2d_config,
                'epoch': num_epochs,
                'train_loss': train_loss,
            }, model_path)
            
            print(f"Model saved to: {model_path}")
            
        except Exception as e:
            print(f"Model saving error: {e}")
    
    # Cleanup
    cleanup_distributed()


def main():
    """Main function to run multi-GPU training"""
    parser = argparse.ArgumentParser(description='Multi-GPU wav2vec2_2d training')
    parser.add_argument('--data_path', type=str, default='/scratch/mkp6112/LFP/region_decoding/data/Allen/data/')
    parser.add_argument('--output_path', type=str, default='./outputs')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--world_size', type=int, default=1)  # Start with 1 GPU
    
    args = parser.parse_args()
    
    # Check available GPUs
    if not torch.cuda.is_available():
        print("CUDA not available. Multi-GPU training requires CUDA.")
        return
    
    world_size = min(args.world_size, torch.cuda.device_count())
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
        
        # Launch multi-GPU training
        mp.spawn(
            run_training,
            args=(world_size, session_data, args.output_path, args.num_epochs),
            nprocs=world_size,
            join=True
        )
        
        print(f"Completed session {i+1}/{len(sessions_list)}")


if __name__ == "__main__":
    main()
