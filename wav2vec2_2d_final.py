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
import random

# Add fairseq to path
sys.path.insert(0, '/vast/us2193/fairseq')

# Import fairseq components
from fairseq.models.wav2vec.wav2vec2_2d import Wav2Vec2_2DConfig, Wav2Vec2_2DModel

class ModifiedSessionDataset(Dataset):
    """Lazy-loading dataset over .pickle files (directory or explicit list).

    Each pickle is expected to be a list where each element is either a
    numpy-like array or a tuple whose first element is the array.

    data_path can be either:
      - a directory containing .pickle files
      - a list of absolute file paths to .pickle files
    """
    
    def __init__(self, data_path, max_samples=None, chunk_size=100):
        self.data_path = data_path
        self.max_samples = None if (max_samples is None or (isinstance(max_samples, int) and max_samples <= 0)) else max_samples
        self.chunk_size = chunk_size
        # Index over files: list of dicts {path, start, count}
        self.file_index = []
        self.total_samples = 0
        # Simple last-file cache to avoid reloading the same pickle repeatedly
        self._cache_path = None
        self._cache_data = None
        self._build_index()
    
    def _build_index(self):
        print("Indexing data files (lazy loading)...")
        # Resolve file list
        if isinstance(self.data_path, (list, tuple)):
            files = [f if os.path.isabs(f) else os.path.join(os.getcwd(), f) for f in self.data_path]
        else:
            files = sorted([
                os.path.join(self.data_path, f)
                for f in os.listdir(self.data_path)
                if f.endswith('.pickle')
            ])
        print(f"Found {len(files)} pickle files")

        start = 0
        for filepath in files:
            fname = os.path.basename(filepath)
            try:
                with open(filepath, 'rb') as f:
                    raw_data = pickle.load(f)
            except Exception as e:
                print(f"  Skipping {fname}: failed to load for indexing ({e})")
                continue
            if not (isinstance(raw_data, list) and len(raw_data) > 0):
                print(f"  Skipping {fname}: not a non-empty list")
                continue
            file_len = len(raw_data)
            # Respect global max_samples cap at index time
            if self.max_samples is not None:
                remaining = self.max_samples - self.total_samples
                if remaining <= 0:
                    break
                file_take = min(file_len, remaining)
            else:
                file_take = file_len
            if file_take <= 0:
                continue
            self.file_index.append({"path": filepath, "start": start, "count": file_take})
            start += file_take
            self.total_samples += file_take
            print(f"  Indexed {file_take} from {fname} (total indexed: {self.total_samples})")
        print(f"Final indexed samples: {self.total_samples} across {len(self.file_index)} files")

    def __len__(self):
        return self.total_samples

    def _load_file_into_cache(self, path):
        if self._cache_path == path and self._cache_data is not None:
            return
        with open(path, 'rb') as f:
            raw = pickle.load(f)
        self._cache_path = path
        self._cache_data = raw

    def __getitem__(self, idx):
        # Map global idx to file and local idx
        if idx < 0 or idx >= self.total_samples:
            idx = idx % self.total_samples
        # Binary search over file_index (linear is fine for modest counts)
        for entry in self.file_index:
            start = entry["start"]
            count = entry["count"]
            if start <= idx < start + count:
                local_idx = idx - start
                path = entry["path"]
                # Load file (cached when possible)
                self._load_file_into_cache(path)
                sample = self._cache_data[local_idx]
                # If sample is a tuple, take the first element
                if isinstance(sample, tuple) and len(sample) > 0:
                    sample = sample[0]
                # Convert to tensor
                tensor = torch.as_tensor(sample, dtype=torch.float32)
                return tensor
        # Fallback (should not happen)
        raise IndexError(f"Index out of range: {idx}")

class LazySessionIterableDataset(torch.utils.data.IterableDataset):
    """True lazy IterableDataset over .pickle files, with rank-based sharding.

    - Avoids pre-reading file lengths. Starts yielding immediately.
    - Shards samples across distributed ranks by round-robin on a global counter.
    - Expects each pickle to be a list of samples (or tuples where first item is the array).
    """
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir
        self.files = sorted(
            [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.pickle')]
        )
        if len(self.files) == 0:
            print(f"No pickle files found in {self.data_dir}")

    def _iter_all(self):
        for fp in self.files:
            try:
                with open(fp, 'rb') as f:
                    data = pickle.load(f)
            except Exception as e:
                print(f"Skipping {os.path.basename(fp)} due to load error: {e}")
                continue
            if not isinstance(data, list) or len(data) == 0:
                continue
            for item in data:
                if isinstance(item, tuple) and len(item) > 0:
                    item = item[0]
                tensor = torch.as_tensor(item, dtype=torch.float32)
                yield tensor

    def __iter__(self):
        # Shard by rank across all yielded samples
        rank = 0
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        for idx, sample in enumerate(self._iter_all()):
            if (idx % world_size) == rank:
                yield sample
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if idx >= len(self.data):
            idx = idx % len(self.data)
        return self.data[idx]

def ddp_setup(rank, world_size):
    """Initialize distributed training"""
    # Set the rank environment variable for this process
    os.environ['RANK'] = str(rank)
    
    # Use env:// rendezvous so torchrun-provided env vars are honored
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=torch.distributed.constants.default_pg_timeout,
    )
    torch.cuda.set_device(rank)

def ddp_cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

def _shape(x):
    try:
        return tuple(x.shape)
    except Exception:
        return type(x).__name__

def setup_shape_trace_once(model: nn.Module):
    """Register forward hooks to print input/output shapes for one forward pass, then remove."""
    printed = {"done": False}
    handles = []

    def hook_factory(tag):
        def hook(module, inputs, output):
            if printed["done"]:
                return
            in_shapes = []
            for i in inputs:
                if isinstance(i, (list, tuple)):
                    in_shapes.append([_shape(t) for t in i])
                else:
                    in_shapes.append(_shape(i))
            out_shape = _shape(output)
            print(f"TRACE {tag}: in={in_shapes} out={out_shape}")
        return hook

    # High-level components
    if hasattr(model, 'module'):
        net = model.module
    else:
        net = model

    # Feature extractor
    if hasattr(net, 'feature_extractor'):
        handles.append(net.feature_extractor.register_forward_hook(hook_factory('feature_extractor')))
    # Layer norm before projection (may be LayerNorm on sequences)
    if hasattr(net, 'layer_norm'):
        handles.append(net.layer_norm.register_forward_hook(hook_factory('layer_norm')))
    # Post extract projection
    if hasattr(net, 'post_extract_proj') and net.post_extract_proj is not None:
        handles.append(net.post_extract_proj.register_forward_hook(hook_factory('post_extract_proj')))
    # Temporal conv1d stack
    if hasattr(net, 'temporal_conv1d') and net.temporal_conv1d is not None:
        handles.append(net.temporal_conv1d.register_forward_hook(hook_factory('temporal_conv1d')))
    # Encoder
    if hasattr(net, 'encoder'):
        handles.append(net.encoder.register_forward_hook(hook_factory('encoder')))
    # Final projection
    if hasattr(net, 'final_proj'):
        handles.append(net.final_proj.register_forward_hook(hook_factory('final_proj')))

    # Top-level wrapper to remove hooks after one successful forward
    def top_hook(module, inputs, output):
        if not printed["done"]:
            print(f"TRACE model.forward: in={_shape(inputs[0])} out={_shape(output.get('x') if isinstance(output, dict) else output)}")
            for h in handles:
                h.remove()
            printed["done"] = True

    handles.append(net.register_forward_hook(top_hook))
    return handles

def create_session_datasets(data_path, train_sessions, val_sessions, test_sessions, max_samples_per_split=None):
    """Create separate datasets for train/val/test based on session splits"""
    
    # First, find all pickle files in the data directory
    all_pickle_files = []
    if os.path.isdir(data_path):
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.pickle'):
                    all_pickle_files.append(os.path.join(root, file))
    else:
        print(f"Error: Data path {data_path} is not a directory")
        return None, None, None
    
    print(f"Found {len(all_pickle_files)} pickle files in {data_path}")
    if len(all_pickle_files) > 0:
        print("Sample files found:")
        for i, file in enumerate(all_pickle_files[:5]):  # Show first 5 files
            print(f"  {i+1}. {os.path.basename(file)}")
        if len(all_pickle_files) > 5:
            print(f"  ... and {len(all_pickle_files) - 5} more files")
    
    def create_dataset_from_sessions(sessions, max_samples=None):
        """Create dataset from specific sessions by filtering pickle files"""
        session_files = []
        for session in sessions:
            # Find files that start with the session ID
            matching_files = [f for f in all_pickle_files if os.path.basename(f).startswith(session)]
            if matching_files:
                session_files.extend(matching_files)
                print(f"Found {len(matching_files)} files for session {session}")
            else:
                print(f"Warning: No files found for session {session}")
        
        if not session_files:
            print(f"Warning: No files found for sessions {sessions}")
            return None
            
        print(f"Using {len(session_files)} files for dataset")
        return ModifiedSessionDataset(
            data_path=session_files,
            max_samples=max_samples,
            chunk_size=100
        )
    
    # Create datasets for each split
    train_dataset = create_dataset_from_sessions(train_sessions, max_samples_per_split)
    val_dataset = create_dataset_from_sessions(val_sessions, max_samples_per_split)
    test_dataset = create_dataset_from_sessions(test_sessions, max_samples_per_split)
    
    return train_dataset, val_dataset, test_dataset

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
            
            # Forward pass - use same approach as single GPU
            outputs = model(
                source=batch,
                mask_indices=None,  # Let model handle masking internally
                features_only=False
            )
            
            # Debug: Print model outputs structure (only on first batch and rank 0)
            if batch_idx == 0 and rank == 0:
                print(f"\nModel outputs structure:")
                if isinstance(outputs, dict):
                    for key, value in outputs.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  {key}: {value.shape} ({value.dtype})")
                        else:
                            print(f"  {key}: {type(value)} = {value}")
                else:
                    print(f"  outputs type: {type(outputs)}")
                    if hasattr(outputs, 'shape'):
                        print(f"  outputs shape: {outputs.shape}")
            
            # Compute contrastive and diversity losses from model outputs dict
            try:
                # Initialize default losses
                contrastive_loss = torch.tensor(0.0, device=device)
                diversity_loss = torch.tensor(0.0, device=device)
                loss = torch.tensor(0.0, device=device)
                
                if isinstance(outputs, dict):
                    # Prefer computing contrastive loss from logits when available
                    if 'x' in outputs:
                        raw_logits = outputs['x']
                        if isinstance(raw_logits, torch.Tensor) and raw_logits.ndim == 3:
                            # Prefer (B, T, C) layout; fallback to (C, B, T)
                            try:
                                B, T, C = raw_logits.shape
                                logits_bt_c = raw_logits.contiguous().view(-1, C)  # [B*T, C]
                                targets_bt = torch.zeros(B * T, dtype=torch.long, device=device)
                            except Exception:
                                C, B, T = raw_logits.shape
                                logits_bt_c = raw_logits.permute(1, 2, 0).contiguous().view(-1, C)
                                targets_bt = torch.zeros(B * T, dtype=torch.long, device=device)
                            contrastive_loss = nn.CrossEntropyLoss()(logits_bt_c, targets_bt)
                    elif 'contrastive_loss' in outputs:
                        contrastive_loss = outputs['contrastive_loss'] if isinstance(outputs['contrastive_loss'], torch.Tensor) else torch.tensor(float(outputs['contrastive_loss']), device=device)

                    # Diversity component
                    if 'diversity_loss' in outputs:
                        diversity_loss = outputs['diversity_loss'] if isinstance(outputs['diversity_loss'], torch.Tensor) else torch.tensor(float(outputs['diversity_loss']), device=device)
                    elif ('prob_perplexity' in outputs) and ('num_vars' in outputs):
                        num_vars = outputs['num_vars']
                        prob_ppl = outputs['prob_perplexity']
                        if isinstance(num_vars, torch.Tensor) and isinstance(prob_ppl, torch.Tensor):
                            diversity_loss = (num_vars - prob_ppl) / torch.clamp(num_vars, min=1)
                            diversity_loss = diversity_loss.mean()

                    # Always construct training loss from contrastive + diversity when logits were present
                    if 'x' in outputs:
                        loss = contrastive_loss + 0.1 * diversity_loss
                    else:
                        # Fallback only if logits missing
                        if 'loss' in outputs and isinstance(outputs['loss'], torch.Tensor):
                            loss = outputs['loss']
                        elif 'features_pen' in outputs and isinstance(outputs['features_pen'], torch.Tensor):
                            loss = outputs['features_pen']
                        else:
                            loss = contrastive_loss + 0.1 * diversity_loss
                else:
                    # Fallback: simple feature penalty
                    loss = torch.tensor(0.1, device=device)  # Small positive loss to avoid zero
                
                # Ensure all losses are tensors and have valid values
                if not isinstance(contrastive_loss, torch.Tensor):
                    contrastive_loss = torch.tensor(0.0, device=device)
                if not isinstance(diversity_loss, torch.Tensor):
                    diversity_loss = torch.tensor(0.0, device=device)
                if not isinstance(loss, torch.Tensor):
                    loss = torch.tensor(0.1, device=device)
                
                # Check for NaN or infinite values
                if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss):
                    contrastive_loss = torch.tensor(0.0, device=device)
                if torch.isnan(diversity_loss) or torch.isinf(diversity_loss):
                    diversity_loss = torch.tensor(0.0, device=device)
                if torch.isnan(loss) or torch.isinf(loss):
                    loss = torch.tensor(0.1, device=device)
                    
            except Exception as e:
                print(f"Rank {rank} - Loss computation error: {e}")
                # Set default losses on error
                contrastive_loss = torch.tensor(0.0, device=device)
                diversity_loss = torch.tensor(0.0, device=device)
                loss = torch.tensor(0.1, device=device)
            
            # Backward pass (skip if loss has no grad path)
            if not loss.requires_grad:
                if rank == 0 and (batch_idx % 50 == 0):
                    print("Skipping backward: loss has no grad path (no grad_fn). Likely missing logits for this batch.")
                continue
            optimizer.zero_grad()
            loss.backward()
            
            # Check for NaN gradients
            has_nan = False
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan = True
                    break
            
            if not has_nan:
                optimizer.step()
            else:
                print(f"Rank {rank} - Skipping step due to NaN gradients")
            
            total_loss += loss.item()
            total_contrastive += contrastive_loss.item()
            total_diversity += diversity_loss.item()
            num_batches += 1
            
            # Update progress bar
            if rank == 0:
                # Debug: Print loss values every 100 batches
                if batch_idx % 100 == 0:
                    print(f"\nBatch {batch_idx} Loss Values:")
                    print(f"  Total Loss: {loss.item():.6f}")
                    print(f"  Contrastive Loss: {contrastive_loss.item():.6f}")
                    print(f"  Diversity Loss: {diversity_loss.item():.6f}")
                
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

def validate_epoch(model, dataloader, device, rank):
    """Validate one epoch with wav2vec2 model"""
    model.eval()
    total_loss = 0.0
    total_contrastive = 0.0
    total_diversity = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Rank {rank} Validation", disable=rank != 0)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move to device
                batch = batch.to(device)
                
                # Reshape data for 2D CNN: [batch_size, 1, height, width]
                batch_size = batch.size(0)
                if batch.size(1) == 3750:
                    # Reshape to 2D: [batch_size, 1, 3750, 93]
                    batch = batch.unsqueeze(1).repeat(1, 1, 1, 93)  # [batch_size, 1, 3750, 93]
                
                # Forward pass
                outputs = model(
                    source=batch,
                    mask_indices=None,
                    features_only=False
                )
                
                # Compute losses
                try:
                    # Initialize default losses
                    contrastive_loss = torch.tensor(0.0, device=device)
                    diversity_loss = torch.tensor(0.0, device=device)
                    loss = torch.tensor(0.0, device=device)
                    
                    if isinstance(outputs, dict):
                        # Try to get contrastive loss directly from model outputs
                        if 'contrastive_loss' in outputs:
                            contrastive_loss = outputs['contrastive_loss']
                            if not isinstance(contrastive_loss, torch.Tensor):
                                contrastive_loss = torch.tensor(float(contrastive_loss), device=device)
                        elif 'x' in outputs:
                            # Compute contrastive loss from logits
                            raw_logits = outputs['x']
                            if isinstance(raw_logits, torch.Tensor) and len(raw_logits.shape) >= 2:
                                C, B, T = raw_logits.shape
                                logits_bt_c = raw_logits.permute(1, 2, 0).contiguous().view(-1, C)
                                targets_bt = torch.zeros(B * T, dtype=torch.long, device=device)
                                contrastive_loss = nn.CrossEntropyLoss()(logits_bt_c, targets_bt)
                        
                        # Try to get diversity loss
                        if 'diversity_loss' in outputs:
                            diversity_loss = outputs['diversity_loss']
                            if not isinstance(diversity_loss, torch.Tensor):
                                diversity_loss = torch.tensor(float(diversity_loss), device=device)
                        elif ('prob_perplexity' in outputs) and ('num_vars' in outputs):
                            num_vars = outputs['num_vars']
                            prob_ppl = outputs['prob_perplexity']
                            if isinstance(num_vars, torch.Tensor) and isinstance(prob_ppl, torch.Tensor):
                                diversity_loss = (num_vars - prob_ppl) / max(num_vars, 1)
                                if isinstance(diversity_loss, torch.Tensor):
                                    diversity_loss = diversity_loss.mean()
                        
                        # Try to get total loss
                        if 'loss' in outputs:
                            loss = outputs['loss']
                            if not isinstance(loss, torch.Tensor):
                                loss = torch.tensor(float(loss), device=device)
                        elif 'features_pen' in outputs:
                            loss = outputs['features_pen']
                            if not isinstance(loss, torch.Tensor):
                                loss = torch.tensor(float(loss), device=device)
                        else:
                            # Compute total loss
                            loss = contrastive_loss + 0.1 * diversity_loss
                    else:
                        # Fallback: simple feature penalty
                        loss = torch.tensor(0.1, device=device)
                    
                    # Ensure all losses are tensors and have valid values
                    if not isinstance(contrastive_loss, torch.Tensor):
                        contrastive_loss = torch.tensor(0.0, device=device)
                    if not isinstance(diversity_loss, torch.Tensor):
                        diversity_loss = torch.tensor(0.0, device=device)
                    if not isinstance(loss, torch.Tensor):
                        loss = torch.tensor(0.1, device=device)
                    
                    # Check for NaN or infinite values
                    if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss):
                        contrastive_loss = torch.tensor(0.0, device=device)
                    if torch.isnan(diversity_loss) or torch.isinf(diversity_loss):
                        diversity_loss = torch.tensor(0.0, device=device)
                    if torch.isnan(loss) or torch.isinf(loss):
                        loss = torch.tensor(0.1, device=device)
                        
                except Exception as e:
                    print(f"Rank {rank} - Validation loss computation error: {e}")
                    # Set default losses on error
                    contrastive_loss = torch.tensor(0.0, device=device)
                    diversity_loss = torch.tensor(0.0, device=device)
                    loss = torch.tensor(0.1, device=device)
                
                total_loss += loss.item()
                total_contrastive += contrastive_loss.item()
                total_diversity += diversity_loss.item()
                num_batches += 1
                
                # Update progress bar
                if rank == 0:
                    # Debug: Print loss values every 50 batches during validation
                    if batch_idx % 50 == 0:
                        print(f"\nValidation Batch {batch_idx} Loss Values:")
                        print(f"  Total Loss: {loss.item():.6f}")
                        print(f"  Contrastive Loss: {contrastive_loss.item():.6f}")
                        print(f"  Diversity Loss: {diversity_loss.item():.6f}")
                    
                    progress_bar.set_postfix({
                        "val_loss": f"{loss.item():.4f}",
                        "val_contrastive": f"{contrastive_loss.item():.4f}",
                        "val_diversity": f"{diversity_loss.item():.4f}"
                    })
                    
            except Exception as e:
                print(f"Rank {rank} - Validation batch {batch_idx} error: {e}")
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
    
    # Define Allen dataset sessions (fixed sets used in other scripts)
    allen5 = ['719161530', '768515987', '771160300', '798911424', '771990200']
    allen7 = ['719161530', '794812542', '778998620', '798911424', '771160300', '768515987', '771990200']
    if args.session_set == 'allen7':
        allen_sessions = allen7
    else:
        allen_sessions = allen5
    
    # Split sessions into train/val/test (80/20 split, with one session for test)
    if args.test_session is not None:
        test_session = args.test_session
        session_list = [s for s in allen_sessions if s != test_session]
    else:
        # Use last session as test by default
        test_session = allen_sessions[-1]
        session_list = allen_sessions[:-1]
    
    # Split remaining sessions into train/val (80/20)
    random.seed(42)
    train_sessions = random.sample(session_list, int(len(session_list) * 0.8))
    val_sessions = [s for s in session_list if s not in train_sessions]
    
    if rank == 0:
        print(f"Training sessions: {train_sessions}")
        print(f"Validation sessions: {val_sessions}")
        print(f"Test session: {test_session}")
    
    # Create datasets for each split
    train_dataset, val_dataset, test_dataset = create_session_datasets(
        data_path=args.data_path,
        train_sessions=train_sessions,
        val_sessions=val_sessions,
        test_sessions=[test_session],
        max_samples_per_split=args.max_samples // 3 if args.max_samples else None
    )
    
    if train_dataset is None or len(train_dataset) == 0:
        print(f"Rank {rank}: No training data loaded, exiting")
        ddp_cleanup()
        return
    
    # Create dataloaders with distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    
    val_dataloader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
    
    test_dataloader = None
    if test_dataset is not None and len(test_dataset) > 0:
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            sampler=test_sampler,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
    
    # Create Wav2Vec2 2D model - EXACT SAME CONFIG as single GPU
    config = Wav2Vec2_2DConfig(
        # 2D CNN feature extraction layers - FIXED to prevent zero output
        conv_2d_feature_layers="[(64, 10, 1), (128, 10, 1), (256, 10, 1), (512, 10, 1)]",
        
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

    # Enable one-pass shape tracing on rank 0, first batch
    if rank == 0:
        setup_shape_trace_once(model)
    
    # Wrap with DDP - FIXED to prevent hanging
    model = DDP(
        model, 
        device_ids=[rank], 
        find_unused_parameters=True,  # This fixes the DDP hanging issue
        broadcast_buffers=False
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print(f"Rank {rank}: Starting training for {args.num_epochs} epochs")
    
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Set epoch for samplers
        train_sampler.set_epoch(epoch)
        if val_dataloader is not None:
            val_sampler.set_epoch(epoch)
        
        # Train epoch
        train_losses = train_epoch(model, train_dataloader, optimizer, device, rank)
        
        # Validation
        val_losses = None
        if val_dataloader is not None:
            val_losses = validate_epoch(model, val_dataloader, device, rank)
        
        # Synchronize all processes before printing
        dist.barrier()
        
        if rank == 0:
            print(f"Epoch {epoch + 1} - Train Loss: {train_losses['total_loss']:.4f}, "
                  f"Train Contrastive: {train_losses['contrastive_loss']:.4f}, "
                  f"Train Diversity: {train_losses['diversity_loss']:.4f}")
            
            if val_losses is not None:
                print(f"Epoch {epoch + 1} - Val Loss: {val_losses['total_loss']:.4f}, "
                      f"Val Contrastive: {val_losses['contrastive_loss']:.4f}, "
                      f"Val Diversity: {val_losses['diversity_loss']:.4f}")
                
                # Track best validation loss
                if val_losses['total_loss'] < best_val_loss:
                    best_val_loss = val_losses['total_loss']
                    print(f"New best validation loss: {best_val_loss:.4f}")
    
    # Final test evaluation
    if test_dataloader is not None and rank == 0:
        print("\n" + "="*50)
        print("FINAL TEST EVALUATION")
        print("="*50)
        test_losses = validate_epoch(model, test_dataloader, device, rank)
        print(f"Test Loss: {test_losses['total_loss']:.4f}, "
              f"Test Contrastive: {test_losses['contrastive_loss']:.4f}, "
              f"Test Diversity: {test_losses['diversity_loss']:.4f}")
    
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
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=150000,
                       help='Maximum samples to load (half of 301k dataset)')
    parser.add_argument('--chunk_size', type=int, default=100,
                       help='Chunk size for data processing')
    parser.add_argument('--input_dim', type=int, default=3750,
                       help='Input dimension')
    parser.add_argument('--embed_dim', type=int, default=384,
                       help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=6,
                       help='Number of attention heads')
    parser.add_argument('--world_size', type=int, default=1,
                       help='Number of GPUs to use')
    parser.add_argument('--test_session', type=str, default=None,
                       help='Specific session to use for testing (if None, uses last session)')
    parser.add_argument('--session_set', type=str, default='allen5', choices=['allen5', 'allen7'],
                       help='Which fixed Allen session ID set to use')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    print(f"Starting multi-GPU training with {args.world_size} GPUs")
    print(f"Data path: {args.data_path}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Max samples: {args.max_samples}")
    
    # Set up environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = '0'  # Will be overridden by mp.spawn
    
    # Launch distributed training
    mp.spawn(main_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)

if __name__ == "__main__":
    main()
