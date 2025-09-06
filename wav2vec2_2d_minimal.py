#!/usr/bin/env python3
"""
Minimal Multi-GPU wav2vec2_2d training - ultra low memory
"""

import argparse
import os
import pickle
import torch
import numpy as np
import sys
from torch.distributed.elastic.multiprocessing.errors import get_error_handler
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import warnings
warnings.filterwarnings("ignore")

# Add fairseq to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fairseq.models.wav2vec.wav2vec2_2d import Wav2Vec2_2DConfig, Wav2Vec2_2DModel
from fairseq.optim.adam import FairseqAdam
import torch.nn.functional as F


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
    """Train for one epoch - minimal memory"""
    model.train()
    total_loss = 0
    grad_norms = []
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"[Rank {rank}] Number of train samples: {len(data_loader.sampler)}")
    
    for step, batch in enumerate(data_loader):
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
            
            # Compute loss
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


def run_wav2vec2_2d_minimal(session_data, output_path, num_epochs=5):
    """Main training function - minimal memory usage"""
    
    # Setup distributed training
    rank, world_size, local_rank = ddp_setup()
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print(f"Starting Minimal Multi-GPU training on {world_size} GPUs")
        print(f"Session: {session_data['session_id']}")
    
    # Load data - ultra minimal version
    try:
        if isinstance(session_data['data'], np.ndarray):
            data = session_data['data']
        else:
            # For large lists, sample a very small subset
            raw_data = session_data['data']
            if len(raw_data) > 100:  # If more than 100 elements
                print(f"[Rank {rank}] Large dataset detected: {len(raw_data)} elements. Sampling 100 for training...")
                # Sample every nth element to get ~100 samples
                step = len(raw_data) // 100
                data = np.array(raw_data[::step])
            else:
                data = np.array(raw_data)
        
        # Reshape to 2D matrix: [timePoints, channels]
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        elif len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        
        print(f"[Rank {rank}] Data shape: {data.shape}")
        
        # Create dataset with very small chunks
        chunk_size = min(10, data.shape[0])  # Process in chunks of 10
        dataset = []
        for i in range(0, data.shape[0], chunk_size):
            chunk = data[i:i+chunk_size]
            dataset.append({'source': torch.FloatTensor(chunk)})
        
        # Create distributed sampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        
        # Create data loader - minimal settings
        data_loader = DataLoader(
            dataset,
            batch_size=1,  # Minimal batch size
            sampler=sampler,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False
        )
        
        if rank == 0:
            print(f"Data loaded successfully: {len(dataset)} samples")
        
    except Exception as e:
        print(f"[Rank {rank}] Data loading error: {e}")
        ddp_cleanup()
        return
    
    # Model configuration - minimal size
    use_spatial_embedding = False
    use_scaled_rope = True
    
    w2v2_2d_config = Wav2Vec2_2DConfig(
        # 2D CNN specific parameters - minimal
        conv_2d_feature_layers="[(16, 3, 2), (32, 3, 2)]",  # Smaller CNN
        conv_2d_stride=2,
        conv_2d_kernel_size=3,
        input_width=data.shape[1],
        input_height=data.shape[0],
        
        # Encoder parameters - minimal
        encoder_layers=2,  # Minimal layers
        encoder_embed_dim=64,  # Very small
        encoder_ffn_embed_dim=256,  # Very small
        encoder_attention_heads=2,  # Minimal heads
        activation_fn="gelu",
        
        # Scaled RoPE parameters
        use_scaled_rope=use_scaled_rope,
        rope_max_seq_len=1024,  # Smaller
        rope_scale_factor=1.0,
        rope_theta=10000.0,
        
        # Legacy spatial embedding parameters (disabled)
        use_spatial_embedding=use_spatial_embedding,
        
        # Masking parameters - minimal
        mask_prob=0.1,  # Smaller
        mask_length=2,  # Smaller
        mask_selection="static",
        mask_other=0.0,
        mask_min_space=1,
        mask_channel_prob=0.0,
        mask_channel_other=0.0,
        mask_channel_min_space=1,
        mask_channel_length=32,  # Smaller
        no_mask_channel_overlap=False,
        mask_channel_selection="static",
        
        # Feature extractor parameters - minimal
        feature_grad_mult=0.0,
        layer_norm=True,
        layerdrop=0.0,  # No layerdrop
        activation_dropout=0.0,
        dropout=0.0,  # No dropout
        attention_dropout=0.0,
        encoder_layerdrop=0.0,  # No layerdrop
        dropout_input=0.0,
        dropout_features=0.0,
        
        # Quantization parameters - minimal
        quantizer_depth=1,  # Minimal
        quantizer_factor=2,  # Smaller
        latent_vars=64,  # Very small
        latent_groups=1,  # Minimal
        same_quantizer=False,
        codebook_negatives=5,  # Smaller
        
        # Negative sampling parameters - minimal
        num_negatives=5,  # Very small
        cross_sample_negatives=2,  # Very small
        
        # Temporal parameters - minimal
        temporal_conv1d_enabled=True,
        temporal_steps=10,  # Very small
        
        # Pooling parameters - minimal
        flattened_pool_dim=32,  # Very small
        
        # Logging - minimal
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
    
    # Create model
    try:
        model = Wav2Vec2_2DModel(w2v2_2d_config)
        model = model.to(device)
        
        # Wrap with DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        
        if rank == 0:
            print(f"Model created successfully: {sum(p.numel() for p in model.parameters()):,} parameters")
        
    except Exception as e:
        print(f"[Rank {rank}] Model creation error: {e}")
        ddp_cleanup()
        return
    
    # Create optimizer
    try:
        optimizer = FairseqAdam(
            model.parameters(),
            lr=1e-3,  # Higher learning rate for small model
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.0  # No weight decay
        )
        
        if rank == 0:
            print("Optimizer created successfully")
        
    except Exception as e:
        print(f"[Rank {rank}] Optimizer creation error: {e}")
        ddp_cleanup()
        return
    
    # Training loop - minimal epochs
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
            model_path = f"{output_path}/{session_data['session_id']}/ssl_model/wav2vec2_2d_minimal.pt"
            
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
    ddp_cleanup()


def main():
    """Main function to run minimal multi-GPU training"""
    parser = argparse.ArgumentParser(description='Minimal Multi-GPU wav2vec2_2d training')
    parser.add_argument('--data_path', type=str, default='/scratch/mkp6112/LFP/region_decoding/script/Allen_w2v2/Allen')
    parser.add_argument('--output_path', type=str, default='./outputs')
    parser.add_argument('--num_epochs', type=int, default=5)
    
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
            run_wav2vec2_2d_minimal(session_data, args.output_path, args.num_epochs)
        except Exception as e:
            print(f"Error in run_wav2vec2_2d_minimal: {e}")
        
        print(f"Completed session {i+1}/{len(sessions_list)}")


if __name__ == "__main__":
    error_handler = get_error_handler()
    try:
        main()
    except Exception as e:
        error_handler.record_exception(e)
        print(f"Error in main: {e}")
