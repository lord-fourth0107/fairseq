#!/usr/bin/env python3
"""
Multi-GPU training script for wav2vec2_2d with Scaled RoPE
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

# Add fairseq to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fairseq'))

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
            
            # Compute loss
            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            else:
                x = outputs['x']
                y = outputs.get('y', x)
                logits = torch.cosine_similarity(x.unsqueeze(0), y.unsqueeze(1), dim=-1)
                targets = torch.zeros(logits.size(1), dtype=torch.long, device=logits.device)
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                loss = F.cross_entropy(logits_flat, targets_flat)
            
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
        data_loader = DataLoader(dataset, batch_size=8, sampler=sampler, num_workers=0, pin_memory=True)
        
        if rank == 0:
            print(f"Data loaded: {data.shape}")
        
    except Exception as e:
        print(f"Data loading error: {e}")
        cleanup_distributed()
        return
    
    # Create model
    try:
        config = Wav2Vec2_2DConfig(
            conv_2d_feature_layers="[(32, 3, 2), (64, 3, 2), (128, 3, 2), (256, 3, 2)]",
            conv_2d_stride=2,
            conv_2d_kernel_size=3,
            input_width=data.shape[1],
            input_height=data.shape[0],
            encoder_layers=6,
            encoder_embed_dim=384,
            encoder_ffn_embed_dim=1536,
            encoder_attention_heads=6,
            activation_fn="gelu",
            use_scaled_rope=True,
            rope_max_seq_len=4096,
            rope_scale_factor=1.0,
            rope_theta=10000.0,
            use_spatial_embedding=False,
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
            feature_grad_mult=0.0,
            layer_norm=True,
            layerdrop=0.1,
            activation_dropout=0.0,
            dropout=0.1,
            attention_dropout=0.1,
            encoder_layerdrop=0.05,
            dropout_input=0.1,
            dropout_features=0.1,
            quantizer_depth=2,
            quantizer_factor=3,
            latent_vars=320,
            latent_groups=2,
            same_quantizer=False,
            codebook_negatives=10,
            num_negatives=20,
            cross_sample_negatives=5,
            temporal_conv1d_enabled=True,
            temporal_steps=50,
            flattened_pool_dim=256,
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
        
        model = Wav2Vec2_2DModel(config)
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
        optimizer = FairseqAdam(model.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
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
                'config': config,
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
    parser.add_argument('--data_path', type=str, default='/Users/uttamsingh/Downloads')
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
