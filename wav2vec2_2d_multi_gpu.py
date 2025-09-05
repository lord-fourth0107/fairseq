#!/usr/bin/env python3
"""
Multi-GPU training script for wav2vec2_2d with Scaled RoPE
Supports distributed training across multiple GPUs
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Add fairseq to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fairseq'))

from fairseq.models.wav2vec.wav2vec2_2d import Wav2Vec2_2DConfig, Wav2Vec2_2DModel
from fairseq.optim.adam import FairseqAdam
from fairseq.optim.lr_scheduler.polynomial_decay_lr_scheduler import PolynomialDecayLRScheduler
from fairseq.tasks.audio_pretraining import AudioPretrainingConfig
from fairseq.tasks.audio_pretraining import AudioPretrainingTask
from fairseq.data.audio.raw_audio_dataset import RawAudioDataset
from fairseq.data import Dictionary
from fairseq.criterions.wav2vec_criterion import Wav2VecCriterion, Wav2VecCriterionConfig
from fairseq.logging.meters import AverageMeter, StopwatchMeter
from fairseq.logging import metrics
from fairseq import utils
from fairseq.dataclass import FairseqDataclass
from fairseq.distributed import fsdp_wrap
from fairseq.distributed.fully_sharded_data_parallel import FullyShardedDataParallel
from fairseq.modules import GradMultiply
from fairseq.utils import buffered_arange, index_put, is_xla_tensor
import torch.nn.functional as F


def setup_distributed(rank, world_size, backend='nccl'):
    """Initialize the distributed process group"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up the distributed process group"""
    dist.destroy_process_group()


def get_grad_norm(model):
    """Compute gradient norm for monitoring"""
    total_norm = 0.0
    param_count = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    return (total_norm / max(param_count, 1)) ** 0.5


def train_2d_multi_gpu(model, data_loader, optimizer, device, rank, accumulation_steps=4):
    """Multi-GPU training function with Scaled RoPE"""
    model.train()
    
    # Initialize metrics
    grad_norm = 0.0
    grad_norms = []
    losses = []
    contrastive_losses = []
    diversity_losses = []
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Progress bar (only on rank 0)
    if rank == 0:
        progress = tqdm(data_loader, desc="Training", leave=False)
    else:
        progress = data_loader
    
    for step, batch in enumerate(progress):
        try:
            # Move batch to device
            if isinstance(batch, dict):
                source = batch['source'].to(device)
            else:
                source = batch.to(device)
            
            # Ensure proper shape: [B, C, H, W]
            if len(source.shape) == 3:
                source = source.unsqueeze(1)  # Add channel dimension
            
            # Forward pass
            outputs = model(source=source, padding_mask=None)
            
            # Extract loss components
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
                if 'contrastive_loss' in locals():
                    contrastive_losses.append(contrastive_loss.item())
                if 'diversity_loss' in locals():
                    diversity_losses.append(diversity_loss.item())
                
            except Exception as e:
                if rank == 0:
                    print(f"Backward pass error: {e}")
                continue
            
            # Update progress bar (only on rank 0)
            if rank == 0 and step % 10 == 0:
                avg_loss = np.mean(losses[-10:]) if losses else 0.0
                avg_contrastive = np.mean(contrastive_losses[-10:]) if contrastive_losses else 0.0
                avg_diversity = np.mean(diversity_losses[-10:]) if diversity_losses else 0.0
                progress.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "contrastive": f"{avg_contrastive:.4f}",
                    "diversity": f"{avg_diversity:.4f}",
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


def validate_2d_multi_gpu(model, data_loader, device, rank):
    """Multi-GPU validation function"""
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in data_loader:
            try:
                if isinstance(batch, dict):
                    source = batch['source'].to(device)
                else:
                    source = batch.to(device)
                
                if len(source.shape) == 3:
                    source = source.unsqueeze(1)
                
                outputs = model(source=source, padding_mask=None)
                
                if isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    # Manual loss computation
                    x = outputs['x']
                    y = outputs.get('y', x)
                    logits = torch.cosine_similarity(x.unsqueeze(0), y.unsqueeze(1), dim=-1)
                    targets = torch.zeros(logits.size(1), dtype=torch.long, device=logits.device)
                    logits_flat = logits.view(-1, logits.size(-1))
                    targets_flat = targets.view(-1)
                    loss = F.cross_entropy(logits_flat, targets_flat)
                
                losses.append(loss.item())
                
            except Exception as e:
                if rank == 0:
                    print(f"Validation error: {e}")
                continue
    
    # Synchronize and average across processes
    dist.barrier()
    
    if rank == 0:
        avg_loss = np.mean(losses) if losses else 0.0
        print(f"Multi-GPU Validation completed - Avg Loss: {avg_loss:.4f}")
    return avg_loss
    else:
        return 0.0


def run_wav2vec2_2d_multi_gpu(rank, world_size, sessions_list, session_data, output_path, num_epochs=10):
    """Main training function for multi-GPU setup"""
    
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"🚀 Starting Multi-GPU training on {world_size} GPUs")
        print(f"📊 Processing session: {session_data['session_id']}")
        print(f"📁 Output path: {output_path}")
    
    # Load and prepare data
    try:
        if isinstance(session_data['data'], np.ndarray):
            data = session_data['data']
        else:
            data = np.array(session_data['data'])
        
        # Reshape to 2D matrix: [timePoints, channels]
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        elif len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        
        print(f"Data shape: {data.shape}")
        
        # Create dataset
        dataset = [{'source': torch.FloatTensor(data)}]
        
        # Create distributed sampler
        sampler = DistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True
        )
        
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=8,  # Smaller batch size per GPU
            sampler=sampler,
            num_workers=0,
            pin_memory=True
        )
        
        if rank == 0:
            print(f"✅ Data loaded successfully: {len(dataset)} samples")
            print(f"📊 Data shape: {data.shape}")
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        cleanup_distributed()
        return
    
    # Model configuration
    use_spatial_embedding = False  # Disable spatial embedding (deprecated)
    use_scaled_rope = True  # Enable Scaled RoPE for positional encoding
    
    w2v2_2d_config = Wav2Vec2_2DConfig(
        # 2D CNN specific parameters
        conv_2d_feature_layers="[(32, 3, 2), (64, 3, 2), (128, 3, 2), (256, 3, 2)]",
        conv_2d_stride=2,
        conv_2d_kernel_size=3,
        input_width=data.shape[1],  # Number of channels
        input_height=data.shape[0],  # Time points
        
        # Encoder parameters
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
        
        # Masking parameters
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
        
        # Feature extractor parameters
        feature_grad_mult=0.0,
        layer_norm=True,
        layerdrop=0.1,
        activation_dropout=0.0,
        dropout=0.1,
        attention_dropout=0.1,
        encoder_layerdrop=0.05,
        dropout_input=0.1,
        dropout_features=0.1,
        
        # Quantization parameters
        quantizer_depth=2,
        quantizer_factor=3,
        latent_vars=320,
        latent_groups=2,
        same_quantizer=False,
        codebook_negatives=10,
        
        # Negative sampling parameters
        num_negatives=20,
        cross_sample_negatives=5,
        codebook_negatives=10,
        
        # Temporal parameters
        temporal_conv1d_enabled=True,
        temporal_steps=50,
        
        # Pooling parameters
        flattened_pool_dim=256,
        
        # Logging
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
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        
    if rank == 0:
            print(f"✅ Model created successfully on GPU {rank}")
            print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        cleanup_distributed()
        return
    
    # Create optimizer
    try:
        optimizer = FairseqAdam(
            model.parameters(),
            lr=5e-4,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.01
        )
        
        if rank == 0:
            print(f"✅ Optimizer created successfully")
        
    except Exception as e:
        print(f"❌ Optimizer creation error: {e}")
        cleanup_distributed()
        return
    
    # Training loop
    if rank == 0:
        print(f"🚀 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        if rank == 0:
            print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
        
        # Set epoch for distributed sampler
        sampler.set_epoch(epoch)
        
        # Training
        train_loss, grad_norm = train_2d_multi_gpu(
            model, data_loader, optimizer, device, rank, accumulation_steps=4
        )
        
        # Validation
        val_loss = validate_2d_multi_gpu(model, data_loader, device, rank)

        if rank == 0:
            print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Grad Norm: {grad_norm:.4f}")
    
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
                'val_loss': val_loss
            }, model_path)
            
            print(f"✅ Model saved to: {model_path}")
            
        except Exception as e:
            print(f"❌ Model saving error: {e}")
    
    # Cleanup
    cleanup_distributed()


def main():
    """Main function to run multi-GPU training"""
    parser = argparse.ArgumentParser(description='Multi-GPU wav2vec2_2d training')
    parser.add_argument('--data_path', type=str, default='/Users/uttamsingh/Downloads', 
                       help='Path to pickle data files')
    parser.add_argument('--output_path', type=str, default='./outputs', 
                       help='Output path for models')
    parser.add_argument('--num_epochs', type=int, default=10, 
                       help='Number of training epochs')
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), 
                       help='Number of GPUs to use')
    
    args = parser.parse_args()
    
    # Check available GPUs
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Multi-GPU training requires CUDA.")
        return
    
    world_size = min(args.world_size, torch.cuda.device_count())
    print(f"🚀 Using {world_size} GPUs for training")
    
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
            print("❌ No pickle files found in data path")
            return
        
        print(f"✅ Found {len(sessions_list)} sessions")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Process each session
    for i, session_data in enumerate(sessions_list):
        print(f"\n🔄 Processing session {i+1}/{len(sessions_list)}: {session_data['session_id']}")
        
        # Launch multi-GPU training
        mp.spawn(
            run_wav2vec2_2d_multi_gpu,
            args=(world_size, sessions_list, session_data, args.output_path, args.num_epochs),
            nprocs=world_size,
            join=True
        )
        
        print(f"✅ Completed session {i+1}/{len(sessions_list)}")


if __name__ == "__main__":
    main()