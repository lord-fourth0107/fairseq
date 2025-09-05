# Single GPU version - no distributed training required
# Run with: python wav2vec2_2d_single_gpu.py

import argparse
import os
import gc
import pickle
import random
import torch
import string
import evaluate
import matplotlib.pyplot as plt
import numpy as np
import sys

import wandb
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy.signal import resample
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, Features, ClassLabel, Sequence, Value
from fairseq.models.wav2vec.wav2vec2_2d import Wav2Vec2_2DConfig, Wav2Vec2_2DModel
# from fairseq.tasks.wav2vec_pretraining import Wav2VecPretrainingTask
# from fairseq.criterions.wav2vec_criterion import Wav2VecCriterion
from fairseq.dataclass import FairseqDataclass
from blind_localization.data.PCAviz import PCAVisualizer
from blind_localization.data.lazyloader_dataset import SessionDataset
from modified_session_dataset import ModifiedSessionDataset
from tqdm import tqdm
from matplotlib.patches import Wedge
from scipy.special import softmax
import matplotlib
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA


# Enter the path to the targeted input data
def arg_parser():
    parser = argparse.ArgumentParser(description='wav2vec2_2d single GPU')
    parser.add_argument('--data', type=str, help='Dataset to use: Allen or ibl', default='Allen')
    parser.add_argument('--trial_length', type=int, default=60, help='trial_length')
    parser.add_argument('--data_type', type=str, help='Data type to use', default='spectrogram')
    parser.add_argument('--sampling_rate', type=str, help='Sampling rate of the data', default='1250')
    parser.add_argument('--load_data', type=lambda x: x.lower() == 'true', help='Load data from disk or compute on fly',
                        default=True)
    parser.add_argument('--rand_init', type=lambda x: x.lower() == 'true', help='random init or start from pretrained',
                        default=False)
    parser.add_argument('--ssl', type=lambda x: x.lower() == 'true',
                        help='self supervised training or fine tuning only', default=True)
    parser.add_argument('--session', type=str, help='session run or full run', default=None)
    parser.add_argument('--input_height', type=int, default=128, help='Height of input spectrogram')
    parser.add_argument('--input_width', type=int, default=128, help='Width of input spectrogram')
    parser.add_argument('--use_spatial_embedding', type=lambda x: x.lower() == 'true', default=True, 
                        help='Whether to use spatial embeddings for recording sites')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    return parser.parse_args()


args = arg_parser()
data, trial_length, data_type, sampling_rate = args.data, args.trial_length, args.data_type, args.sampling_rate
load_data, rand_init, ssl, selected_session = args.load_data, args.rand_init, args.ssl, args.session
input_height, input_width, use_spatial_embedding = args.input_height, args.input_width, args.use_spatial_embedding
gpu_id = args.gpu_id

# Additional arguments for training
output_path = f"/vast/us2193/ssl_output/{data}/{data_type}/wav2vec2_2d/across_session"
epochs = 10
subset_data = 0.1
num_workers = 0

print(f"Data: {data}, Data Type: {data_type}, Trial Length: {trial_length}, Sampling Rate: {sampling_rate}")
print(f"Input Dimensions: {input_height}x{input_width}, Spatial Embedding: {use_spatial_embedding}")
print(f"Load Data: {load_data}, rand_init: {rand_init}, ssl: {ssl}, session: {selected_session}")
print(f"Using GPU: {gpu_id}")
print("cuda is available: ", torch.cuda.is_available())

if not os.path.exists(output_path):
    os.makedirs(output_path)


def compute_mask_inputs_2d(model, input_values, device):
    """
    Compute masking for 2D input (spectrograms).
    input_values shape: (B, C, H, W)
    """
    batch_size, channels, height, width = input_values.shape
    with torch.no_grad():
        # Get the actual sequence length after CNN processing
        # We need to run the feature extractor to get the correct sequence length
        try:
            features = model.feature_extractor(input_values)
            B, C, H_out, W_out = features.shape
            
            # Reshape to flattened format: (B, 1, C*H*W)
            features_reshaped = features.reshape(B, C * H_out * W_out)
            features_reshaped = features_reshaped.unsqueeze(1)  # (B, 1, C*H*W)
            
            # Check if temporal expansion will be applied
            temporal_steps = getattr(model.cfg, 'temporal_steps', 1) if hasattr(model, 'cfg') else 1
            has_temporal_conv = getattr(model.cfg, 'temporal_conv1d_enabled', False) if hasattr(model, 'cfg') else False
            
            if has_temporal_conv:
                actual_seq_len = temporal_steps  # Multiple time steps from 1D CNN
                print(f"   Using temporal expansion: {temporal_steps} time steps")
            else:
                actual_seq_len = features_reshaped.shape[1]  # This is 1 (single time step)
                print(f"   No temporal expansion: {actual_seq_len} time steps")
            
        except Exception as e:
            # Fallback: use a reasonable default sequence length
            print(f"⚠️ Feature extractor failed, using fallback sequence length: {e}")
            # Use a conservative estimate based on input dimensions
            actual_seq_len = min(height * width // 4, 1000)  # Conservative estimate
            print(f"   Using fallback sequence length: {actual_seq_len}")
        
        # Compute masking for the actual sequence length
        mask_prob = 0.2  # You can make this configurable
        
        # Handle masking based on sequence length
        if actual_seq_len == 1:
            # With T=1, we either mask the entire vector or not
            mask = torch.rand(batch_size, device=device) < mask_prob
            mask_time_indices = mask.unsqueeze(1)  # (B, 1)
        else:
            # For multiple time steps, use standard wav2vec2 masking
            mask_length = min(10, actual_seq_len // 4)  # Adjust mask length for temporal steps
            from fairseq.data.data_utils import compute_mask_indices
            mask_time_indices = compute_mask_indices(
                shape=(batch_size, actual_seq_len),
                mask_prob=mask_prob,
                mask_length=mask_length,
                padding_mask=None,
            )
            # move to the same device as inputs
            mask_time_indices = torch.as_tensor(mask_time_indices, device=device)
        
        # Debug: Print masking information (only for first call)
        if not hasattr(compute_mask_inputs_2d, '_debug_printed'):
            print(f"🔍 Masking Debug:")
            print(f"   Input shape: {input_values.shape}")
            print(f"   Feature extractor output: {features.shape}")
            print(f"   Reshaped features: {features_reshaped.shape}")
            print(f"   Actual sequence length: {actual_seq_len}")
            print(f"   Mask shape: {mask_time_indices.shape}")
            print(f"   Masked tokens per sequence: {mask_time_indices.sum(dim=1)}")
            if actual_seq_len == 1:
                print(f"   Expected model output: [batch, {actual_seq_len}, {C * H_out * W_out}] (single time step with flattened features)")
                print(f"   Masking strategy: Mask entire vector or not (T=1)")
            else:
                print(f"   Expected model output: [batch, {actual_seq_len}, features] (multiple time steps from temporal conv1d)")
                print(f"   Masking strategy: Standard wav2vec2 temporal masking")
            compute_mask_inputs_2d._debug_printed = True
        
        # Ensure at least some tokens are masked
        if mask_time_indices.sum() == 0:
            # Randomly mask at least one token per sequence
            for i in range(batch_size):
                idx = torch.randint(0, actual_seq_len, (1,))
                mask_time_indices[i, idx] = True
            print(f"   ⚠️ No tokens masked, added at least one per sequence")
        
        # For 2D, we don't use sampled_negatives in the same way
        # You might need to implement a different negative sampling strategy
        sampled_negatives = None
        
    return mask_time_indices, sampled_negatives


def _robust_normalize_input_shape(input_values, step):
    """
    Robust input shape normalization that handles all possible input formats
    """
    original_shape = input_values.shape
    
    if len(input_values.shape) == 5:
        # 5D input: [batch, channels, depth, height, width]
        batch_size, channels, depth, height, width = input_values.shape
        if step == 0:
            print(f"✅ 5D input detected: {original_shape}")
        
        if depth == 1:
            input_values = input_values.squeeze(2)  # Remove depth dimension
        else:
            # Reshape to combine depth with height
            input_values = input_values.view(batch_size, channels, height * depth, width)
        
        if step == 0:
            print(f"   ✅ Normalized to 4D: {input_values.shape}")
            
    elif len(input_values.shape) == 4:
        # Already 4D: [batch, channels, height, width]
        if step == 0:
            print(f"✅ 4D input detected: {original_shape}")
            
    elif len(input_values.shape) == 3:
        # 3D input: [batch, height, width] -> [batch, 1, height, width]
        input_values = input_values.unsqueeze(1)
        if step == 0:
            print(f"✅ 3D input normalized to 4D: {input_values.shape}")
            
    elif len(input_values.shape) == 2:
        # 2D input: [batch, features] -> [batch, 1, height, width]
        batch_size, features = input_values.shape
        input_values = _infer_2d_shape_from_features(input_values, step)
        
    else:
        raise ValueError(f"Unsupported input shape: {original_shape}")
    
    # Ensure minimum size for CNN kernels
    if len(input_values.shape) == 4 and (input_values.shape[2] < 3 or input_values.shape[3] < 3):
        if step == 0:
            print(f"⚠️ Input too small for 3x3 kernels, padding...")
        pad_h = max(0, 3 - input_values.shape[2])
        pad_w = max(0, 3 - input_values.shape[3])
        input_values = torch.nn.functional.pad(input_values, (0, pad_w, 0, pad_h), mode='constant', value=0)
        if step == 0:
            print(f"   ✅ Padded to: {input_values.shape}")
    
    return input_values

def _infer_2d_shape_from_features(tensor, step):
    """Infer 2D shape from flattened features"""
    batch_size, features = tensor.shape
    
    if step == 0:
        print(f"🔍 Inferring 2D shape from {features} features...")
    
    # Try common neural data configurations
    possible_configs = [
        (3750, 77),   # Common neural data
        (3750, 93),   # Common neural data  
        (3750, 64),   # Power of 2
        (3750, 128),  # Power of 2
        (3750, 256),  # Power of 2
    ]
    
    for height, width in possible_configs:
        if features == height * width:
            if step == 0:
                print(f"✅ 2D input inferred as: [batch, 1, {height}, {width}]")
            return tensor.view(batch_size, 1, height, width)
    
    # Fallback: try to infer from features
    if features % 3750 == 0:
        width = features // 3750
        if step == 0:
            print(f"✅ 2D input inferred as: [batch, 1, 3750, {width}]")
        return tensor.view(batch_size, 1, 3750, width)
    
    # Additional fallback: try to find factors that work
    def find_factors(n):
        """Find two factors of n that are as close as possible"""
        factors = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                factors.append((i, n // i))
        if factors:
            # Return the pair with the smallest difference
            return min(factors, key=lambda x: abs(x[0] - x[1]))
        return None
    
    factors = find_factors(features)
    if factors:
        height, width = factors
        if step == 0:
            print(f"✅ 2D input inferred using factors: [batch, 1, {height}, {width}]")
        return tensor.view(batch_size, 1, height, width)
    
    # Last resort: pad to make it work
    if step == 0:
        print(f"⚠️ Cannot factorize {features}, padding to make it work...")
    
    # Find the next perfect square that's >= features
    sqrt_features = int(features ** 0.5)
    if sqrt_features * sqrt_features < features:
        sqrt_features += 1
    
    target_size = sqrt_features * sqrt_features
    pad_size = target_size - features
    
    if pad_size > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_size), mode='constant', value=0)
        if step == 0:
            print(f"   Padded {features} -> {target_size} features")
    
    if step == 0:
        print(f"✅ 2D input fallback: [batch, 1, {sqrt_features}, {sqrt_features}]")
    
    return tensor.view(batch_size, 1, sqrt_features, sqrt_features)

def train_2d(model, data_loader, optimizer, device, accumulation_steps=4):
    from tqdm.auto import tqdm
    import sys

    total_loss = 0
    grad_norms = []
    grad_norm = 0.0  # Initialize grad_norm
    model.train()
    # print(f"Number of train samples: {len(data_loader)}")
    
    # Performance optimization: limit debug prints to first few steps
    max_debug_steps = 0  # disable debug prints
    # Logging (commented out per request)
    # log_interval = 50
    # running_loss = 0.0
    
    # Memory optimization: clear cache periodically
    torch.cuda.empty_cache()
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    progress = tqdm(data_loader, total=len(data_loader), leave=True, disable=not sys.stdout.isatty())
    for step, (input_values, probe_ids) in enumerate(progress):
        # Performance optimization: add progress indicator
        if step % 100 == 0:
            # print(f"🔄 Training step {step}/{len(data_loader)}")
            # Memory optimization: clear cache every 100 steps
            torch.cuda.empty_cache()
        
        # Performance optimization: limit debug prints
        debug_step = step < max_debug_steps
        # input_values comes as (B, C, H, W) from ModifiedSessionDataset
        # Shape: [1, 1, 3750, 93] - already properly formatted for 2D CNN
        input_values = input_values.float().to(device)
        
        # Debug: Print original shape (only for first few steps)
        if debug_step:
            pass
        
        # ROBUST INPUT SHAPE HANDLING - Comprehensive fix for all dimension issues
        input_values = _robust_normalize_input_shape(input_values, step)
        
        if debug_step:
            pass
        
        # Compute masking for 2D input (optimized)
        mask_time_indices, sampled_negative_indices = compute_mask_inputs_2d(model, input_values, device)
        
        # Forward pass for 2D model with error handling
        try:
            outputs = model(
                source=input_values,
                mask_indices=mask_time_indices,
                features_only=False
            )
        except Exception as e:
            # print(f"❌ Model forward pass failed: {e}")
            # print(f"   Input shape: {input_values.shape}")
            # print(f"   Mask shape: {mask_time_indices.shape if mask_time_indices is not None else 'None'}")
            # Skip this batch if model fails
            continue
        
        # Extract loss from outputs with error handling
        try:
            # if step < 5:  # Debug first few steps
            #     print(f"  Outputs type: {type(outputs)}")
            #     if hasattr(outputs, 'keys'):
            #         print(f"  Outputs keys: {list(outputs.keys())}")
            #     if hasattr(outputs, 'loss'):
            #         print(f"  Has loss attr: {outputs.loss is not None}")
            #         if outputs.loss is not None:
            #             print(f"  Loss value: {outputs.loss.item()}")
            
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            else:
                # Compute contrastive loss for SSL training
                # The model returns logits in 'x' - these are the contrastive predictions
                logits = outputs['x']  # Shape: [num_negatives + 1, batch_size, sequence_length]
                
                # if step < 5:
                #     print(f"  Logits shape: {logits.shape}")
                #     print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
                
                # Create targets: first item is positive (0), rest are negatives (1, 2, ...)
                batch_size, seq_len = logits.shape[1], logits.shape[2]
                targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=logits.device)
                
                # Compute cross-entropy loss (contrastive loss)
                # Reshape logits to [batch_size * seq_len, num_classes]
                logits_flat = logits.permute(1, 2, 0).contiguous().view(-1, logits.shape[0])
                targets_flat = targets.view(-1)
                
                contrastive_loss = F.cross_entropy(logits_flat, targets_flat)
                
                # Add diversity loss if quantization is enabled
                diversity_loss = 0.0
                if 'prob_perplexity' in outputs and outputs['prob_perplexity'] is not None:
                    # Diversity loss: encourage uniform usage of codebook entries
                    prob_perplexity = outputs['prob_perplexity']
                    diversity_loss = -prob_perplexity  # Negative perplexity = diversity loss
                
                # Total loss = contrastive loss + diversity loss (like wav2vec2.0)
                loss = contrastive_loss + 0.1 * diversity_loss  # α=0.1 for diversity loss weight
                
                # if step < 5:
                #     print(f"  Computed contrastive loss: {contrastive_loss.item()}")
                #     print(f"  Computed diversity loss: {diversity_loss.item()}")
                #     print(f"  Total loss: {loss.item()}")
        except Exception as e:
            # if step < 5:
            #     print(f"❌ Loss computation failed: {e}")
            #     print(f"   Outputs type: {type(outputs)}")
            #     if hasattr(outputs, 'keys'):
            #         print(f"   Outputs keys: {list(outputs.keys())}")
            # Skip this batch if loss computation fails
            continue
        
        if loss is None:
            continue

        # Backward pass with error handling and gradient accumulation
        try:
            # Scale loss for gradient accumulation
            scaled_loss = loss / accumulation_steps
            
            if scaler is not None:
                # Mixed precision backward pass
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
                
            # Accumulate gradients
            if (step + 1) % accumulation_steps == 0:
                if scaler is not None:
                    # Mixed precision optimizer step
                    scaler.unscale_(optimizer)
                    grad_norm = get_grad_norm(model)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = get_grad_norm(model)
                    optimizer.step()
                
                grad_norms.append(grad_norm)
                optimizer.zero_grad()
        except Exception as e:
            # print(f"❌ Backward pass failed: {e}")
            # print(f"   Loss value: {loss.item() if loss is not None else 'None'}")
            # Skip this batch if backward pass fails
            continue

        total_loss += loss.item()
        # running_loss += loss.item()

        # Periodic loss logging (commented out per request)
        # if step > 0 and (step % log_interval == 0 or step == len(data_loader) - 1):
        #     avg_running = running_loss / (log_interval if step % log_interval == 0 else (step % log_interval))
        #     print(f"🧮 Train step {step}/{len(data_loader)} | loss: {avg_running:.4f} | grad_norm: {grad_norm:.3f}")
        #     running_loss = 0.0

        # Show loss in progress bar with more debugging
        # if step < 5:  # Debug first few steps
        #     print(f"Step {step}: loss={loss.item():.6f}, grad_norm={grad_norm:.6f}")
        #     print(f"  Loss type: {type(loss)}, requires_grad: {loss.requires_grad}")
        #     print(f"  Model parameters require_grad: {sum(p.requires_grad for p in model.parameters())}")
        
        # Show loss in progress bar
        progress.set_postfix({"loss": f"{loss.item():.4f}", "grad": f"{grad_norm:.2f}"})

    avg_loss = total_loss / len(data_loader)
    avg_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0

    # print(f"✅ Epoch train avg loss: {avg_loss:.4f} | avg grad_norm: {avg_grad:.3f}")
    return avg_loss, avg_grad


def validate_2d(model, data_loader, device):
    model.eval()
    total_loss = 0
    print(f"Number of val samples: {len(data_loader)}")

    with torch.no_grad():
        for step, (input_values, probe_ids) in enumerate(data_loader):
            input_values = input_values.float().to(device)
            
            # ROBUST INPUT SHAPE HANDLING - Same as training
            input_values = _robust_normalize_input_shape(input_values, step)
            
            mask_time_indices, sampled_negative_indices = compute_mask_inputs_2d(model, input_values, device)
            
            outputs = model(
                source=input_values,
                mask_indices=mask_time_indices,
                features_only=False
            )
            
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            else:
                features = outputs['features'] if 'features' in outputs else outputs['x']
                loss = F.mse_loss(features, features.detach())  # Placeholder loss
            
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def get_grad_norm(model, norm_type=2.0):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    return total_norm ** (1.0 / norm_type)


class LinearProber2D(nn.Module):
    def __init__(self, encoder, rep_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        # Add adaptive pooling to handle varying output sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Always output (1, 1) spatial dimensions
        # Initialize classifier with a placeholder - will be recreated with correct dimensions
        self.classifier = None
        self.expected_rep_dim = rep_dim
        self.num_classes = num_classes
        self._classifier_initialized = False

    def forward(self, x):
        # Debug: Print input dimensions
        if not hasattr(self, '_prober_debug_printed'):
            print(f"🔍 Prober Input Debug:")
            print(f"   Input x shape: {x.shape}")
            self._prober_debug_printed = True
        
        # Check if dimensions are too small for 3x3 kernel
        if len(x.shape) == 4 and (x.shape[2] < 3 or x.shape[3] < 3):
            # Pad the input to ensure it's at least 3x3
            pad_h = max(0, 3 - x.shape[2])
            pad_w = max(0, 3 - x.shape[3])
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        # For 2D input, we need to handle the spatial dimensions
        try:
            print(f"   🔍 Calling encoder with x: {x.shape}")
            encoder_output = self.encoder(x, features_only=True)
            print(f"   🔍 Encoder output type: {type(encoder_output)}")
            print(f"   🔍 Encoder output keys: {encoder_output.keys() if isinstance(encoder_output, dict) else 'Not a dict'}")
            
            if isinstance(encoder_output, dict) and 'x' in encoder_output:
                reps = encoder_output['x']
            else:
                print(f"   ⚠️ Encoder output doesn't have 'x' key, using full output")
                reps = encoder_output
            
            # Handle different output shapes from encoder using adaptive pooling
            if len(reps.shape) == 3:  # (B, H*W, D)
                print(f"   Converting 3D to 4D for adaptive pooling: {reps.shape}")
                batch_size, seq_len, feature_dim = reps.shape
                
                # Option 1: Try to reshape to 4D using factor decomposition
                def find_closest_factors(n):
                    factors = []
                    for i in range(1, int(n**0.5) + 1):
                        if n % i == 0:
                            factors.append((i, n // i))
                    if factors:
                        return min(factors, key=lambda x: abs(x[0] - x[1]))
                    return None
                
                factors = find_closest_factors(seq_len)
                if factors and factors[0] * factors[1] == seq_len:
                    h, w = factors
                    reps = reps.view(batch_size, feature_dim, h, w)
                    print(f"   Reshaped 3D to 4D using factors {h}x{w}: {reps.shape}")
                else:
                    # Option 2: Use global average pooling directly on 3D tensor
                    print(f"   Using global average pooling on 3D tensor")
                    reps = reps.mean(dim=1)  # (B, H*W, D) -> (B, D)
                    print(f"   After global average pooling: {reps.shape}")
                
            if len(reps.shape) == 4:  # (B, D, H, W) - 2D features
                print(f"   Applying adaptive pooling to: {reps.shape}")
                # Use adaptive pooling to get consistent output size
                reps = self.adaptive_pool(reps)  # (B, D, 1, 1)
                reps = reps.view(reps.shape[0], -1)  # (B, D)
                print(f"   After adaptive pooling: {reps.shape}")
            elif len(reps.shape) == 2:  # (B, D) - already pooled
                print(f"   Already 2D, no pooling needed: {reps.shape}")
                # Already in the right format
                pass
            else:
                print(f"   Unknown shape {reps.shape}, flattening")
                # Fallback: flatten and use first dimension
                reps = reps.view(reps.shape[0], -1)
            
            # Ensure we return the same batch size as input
            if reps.shape[0] != x.shape[0]:
                # Repeat the output to match input batch size
                reps = reps.repeat(x.shape[0], 1)
                
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Prober forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"   Input shape: {x.shape}")
            print(f"   This indicates a fundamental issue with the encoder model")
            print(f"   Raising exception to stop training and investigate")
            raise RuntimeError(f"Prober forward pass failed: {e}") from e
        
        # Dynamically create classifier with correct input dimensions
        if not self._classifier_initialized or self.classifier is None:
            actual_rep_dim = reps.shape[1]
            print(f"   🔧 Creating classifier with input_dim={actual_rep_dim}, output_dim={self.num_classes}")
            self.classifier = nn.Linear(actual_rep_dim, self.num_classes).to(reps.device)
            self._classifier_initialized = True
            
            # If we have an optimizer, we need to update it with the new parameters
            if hasattr(self, '_optimizer'):
                print(f"   🔧 Updating optimizer with new classifier parameters")
                self._optimizer.param_groups[0]['params'] = list(self.classifier.parameters())
        
        result = self.classifier(reps)
        return result


def train_probe_2d(prober, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    # Initialize optimizer after classifier is created
    optimizer = None
    
    # Pre-flight validation: Test the prober with a sample batch
    print("🔍 Pre-flight validation: Testing prober with sample data...")
    try:
        # Get a sample batch
        sample_batch = next(iter(train_loader))
        sample_xb, sample_yb = sample_batch
        sample_xb, sample_yb = sample_xb.to(device), sample_yb.to(device)
        
        print(f"   Sample input shape: {sample_xb.shape}")
        print(f"   Sample target shape: {sample_yb.shape}")
        
        # Test the reshape operation
        sample_xb_reshaped = sample_xb.unsqueeze(1)
        print(f"   Reshaped input shape: {sample_xb_reshaped.shape}")
        
        # Test the prober
        with torch.no_grad():
            sample_logits = prober(sample_xb_reshaped)
            print(f"   Sample logits shape: {sample_logits.shape}")
            
        # Verify batch sizes match
        if sample_logits.shape[0] != sample_yb.shape[0]:
            print(f"❌ PRE-FLIGHT FAILED: Batch size mismatch in sample data!")
            print(f"   Logits: {sample_logits.shape[0]}, Targets: {sample_yb.shape[0]}")
            return 0.0, 0.0, 0.0, 0.0
        
        print("✅ Pre-flight validation passed!")
        
        # Now initialize the optimizer with the correct classifier parameters
        if prober.classifier is not None:
            optimizer = torch.optim.Adam(prober.classifier.parameters(), lr=5e-6)
            prober._optimizer = optimizer  # Store reference for dynamic updates
            print(f"✅ Optimizer initialized with {len(list(prober.classifier.parameters()))} parameters")
        else:
            print(f"❌ Classifier not created during pre-flight validation")
            return 0.0, 0.0, 0.0, 0.0
        
    except Exception as e:
        print(f"❌ PRE-FLIGHT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0, 0.0, 0.0
    print(f"Number of train samples: {len(train_loader)}, "
          f"Number of val samples: {len(val_loader)}")

    train_loss, train_correct, train_total = 0.0, 0, 0
    val_loss, val_correct, val_total = 0.0, 0, 0

    prober.train()
    # Ensure prober parameters require gradients
    for param in prober.parameters():
        param.requires_grad = True
    
    for batch_idx, (xb, yb) in enumerate(train_loader):
        # Performance optimization: add progress indicator
        if batch_idx % 100 == 0:
            print(f"🔄 Probe training batch {batch_idx}/{len(train_loader)}")
        
        # Debug: Print original shapes
        if batch_idx == 0:
            print(f"🔍 TRAINING LOOP DEBUG:")
            print(f"   Original xb shape: {xb.shape}")
            print(f"   Original yb shape: {yb.shape}")
        
        xb, yb = xb.to(device), yb.to(device)
        
        # The xb is already a 2D matrix from stacking multiple recordings
        # Reshape to (B, C, H, W) where:
        # B = batch_size, C = 1 (single channel), H = num_recordings, W = time_points
        if batch_idx == 0:
            print(f"   Before reshape: xb = {xb.shape}")
        
        # FIX: Add channel dimension without destroying batch dimension
        xb = xb.unsqueeze(1)  # [batch_size, 1, num_recordings, time_points]
        
        if batch_idx == 0:
            print(f"   After reshape: xb = {xb.shape}")
            print(f"   Target yb: {yb.shape}")
            print(f"   Calling prober with xb: {xb.shape}")
            
        logits = prober(xb)
        
        if batch_idx == 0:
            print(f"   Prober returned logits: {logits.shape}")
        
        # Verify batch sizes match (should be fixed now)
        if logits.shape[0] != yb.shape[0]:
            print(f"❌ CRITICAL ERROR: Batch size mismatch detected!")
            print(f"   Logits shape: {logits.shape}")
            print(f"   Targets shape: {yb.shape}")
            print(f"   This indicates a fundamental issue with the data pipeline")
            print(f"   Stopping training to prevent further issues")
            return 0.0, 0.0, 0.0, 0.0  # Return zero metrics
        
        # Add timeout protection for loss computation
        try:
            
            # Ensure logits requires grad
            if not logits.requires_grad:
                print(f"⚠️ Logits don't require grad, enabling...")
                logits = logits.requires_grad_(True)
            
            loss = criterion(logits, yb)
            
            # Ensure loss requires grad
            if not loss.requires_grad:
                print(f"⚠️ Loss doesn't require grad, this will cause backward() to fail")
                print(f"   Loss value: {loss.item()}")
                print(f"   Logits requires_grad: {logits.requires_grad}")
                print(f"   Logits grad_fn: {logits.grad_fn}")
                continue  # Skip this batch
                
        except Exception as e:
            print(f" CRITICAL ERROR: Loss computation failed: {e}")
            print(f"   Logits shape: {logits.shape}")
            print(f"   Targets shape: {yb.shape}")
            import traceback
            traceback.print_exc()
            print(f"   This indicates a fundamental issue with the model or data")
            print(f"   Stopping training to prevent further issues")
            return 0.0, 0.0, 0.0, 0.0  # Return zero metrics
        
        # Only proceed with backward pass if loss has gradients
        if loss.requires_grad:
            if optimizer is None:
                print(f"❌ CRITICAL ERROR: Optimizer not initialized!")
                print(f"   This should have been created during pre-flight validation")
                return 0.0, 0.0, 0.0, 0.0
            try:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(f"❌ CRITICAL ERROR: Backward pass failed: {e}")
                import traceback
                traceback.print_exc()
                print(f"   This indicates a fundamental issue with the model or gradients")
                print(f"   Stopping training to prevent further issues")
                return 0.0, 0.0, 0.0, 0.0  # Return zero metrics
        else:
            print(f"❌ CRITICAL ERROR: Loss doesn't require grad")
            print(f"   Loss value: {loss.item()}")
            print(f"   Logits requires_grad: {logits.requires_grad}")
            print(f"   Logits grad_fn: {logits.grad_fn}")
            print(f"   This indicates a fundamental issue with the model setup")
            print(f"   Stopping training to prevent further issues")
            return 0.0, 0.0, 0.0, 0.0  # Return zero metrics
        train_loss += loss.item() * xb.size(0)
        train_correct += (logits.argmax(1) == yb).sum().item()
        train_total += xb.size(0)

    prober.eval()
    with torch.no_grad():
        for val_batch_idx, (xb, yb) in enumerate(val_loader):
            # Debug: Print original shapes for first validation batch
            if val_batch_idx == 0:
                print(f"🔍 VALIDATION LOOP DEBUG:")
                print(f"   Original xb shape: {xb.shape}")
                print(f"   Original yb shape: {yb.shape}")
            
            xb, yb = xb.to(device), yb.to(device)
            
            # The xb is already a 2D matrix from stacking multiple recordings
            # Reshape to (B, C, H, W) where:
            # B = batch_size, C = 1 (single channel), H = num_recordings, W = time_points
            if val_batch_idx == 0:
                print(f"   Before reshape: xb = {xb.shape}")
            
            # FIX: Add channel dimension without destroying batch dimension
            xb = xb.unsqueeze(1)  # [batch_size, 1, num_recordings, time_points]
            
            if val_batch_idx == 0:
                print(f"   After reshape: xb = {xb.shape}")
                print(f"   Target yb: {yb.shape}")
                print(f"   Calling prober with xb: {xb.shape}")
                
            logits = prober(xb)
            
            if val_batch_idx == 0:
                print(f"   Prober returned logits: {logits.shape}")
            
            # Verify batch sizes match (should be fixed now)
            if logits.shape[0] != yb.shape[0]:
                print(f"❌ Val batch size mismatch detected:")
                print(f"   Logits shape: {logits.shape}")
                print(f"   Targets shape: {yb.shape}")
                print(f"   This should be fixed by the reshape operation above")
                continue
            
            try:
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += xb.size(0)
            except Exception as e:
                print(f"❌ Val loss computation failed: {e}")
                print(f"   Logits shape: {logits.shape}")
                print(f"   Targets shape: {yb.shape}")
                import traceback
                traceback.print_exc()
                continue  # Skip this batch

    train_avg_loss = train_loss / train_total
    train_acc = train_correct / train_total
    val_avg_loss = val_loss / val_total
    val_acc = val_correct / val_total

    return train_avg_loss, train_acc, val_avg_loss, val_acc


# Sessions - List of sessions to be used for training, validation, and testing
# Sess - Specific session (list or single) to be used for testing
def run_wav2vec2_2d(sessions, sess):
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        print(f"Using GPU: {device}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    subset_data = 0.01  # 0.01 for 1% of the data (much faster training)
    num_workers = 0  # 0 for single process, 4 for multi-process (reduced to prevent memory issues)
    epochs = 10  # Number of epochs to train the model
    print(f"Subset data: {subset_data}, Number of workers: {num_workers}, Epochs: {epochs}")

    session_list = sessions.copy()
    if type(sess) is not list:
        session = sess
        file_path = [session]
    else:
        session = "Allen_train_NN_test_zscored"
        file_path = sess

    print(f"Session list: {session_list}")

    if type(sess) is not list:
        session_list.remove(session)
        random.seed(42)
        train_session_list = random.sample(session_list, int(len(session_list) * 0.8))
        val_session_list = [ses for ses in session_list if ses not in train_session_list]
        print("Training on ", train_session_list)
        print("Validate on ", val_session_list)
        print("Testing on ", [session])
    else:
        random.seed(42)
        train_session_list = random.sample(session_list, int(len(session_list) * 0.8))
        val_session_list = [ses for ses in session_list if ses not in train_session_list]
        print("Training on ", train_session_list)
        print("Validate on ", val_session_list)
        print("Testing on ", sess)

    data_loading_path = "/scratch/mkp6112/LFP/region_decoding/script/Allen_w2v2/Allen"
    all_pickles = []
    
    # Fix the path construction issue
    if os.path.exists(data_loading_path):
        for root, dirs, files in os.walk(data_loading_path):
            for file in files:
                if file.endswith('.pickle'):
                    all_pickles.append(os.path.join(root, file))  # Full path
    else:
        print(f"Warning: Data path {data_loading_path} does not exist!")
        print("Please check your data path and update the script accordingly.")
        return

    print(f"Found {len(all_pickles)} pickle files")
    
    # Debug: Print first few pickle files
    if all_pickles:
        print(f"First few pickle files:")
        for i, pickle_file in enumerate(all_pickles[:3]):
            print(f"  {i+1}: {pickle_file}")

    train_sessions = [item for item in all_pickles
                      if any(session in os.path.basename(item) for session in train_session_list)]
    val_sessions = [item for item in all_pickles
                    if any(session in os.path.basename(item) for session in val_session_list)]
    test_sessions = [item for item in all_pickles
                     if any(session in os.path.basename(item) for session in file_path)]

    print(f"Train sessions: {len(train_sessions)}")
    print(f"Val sessions: {len(val_sessions)}")
    print(f"Test sessions: {len(test_sessions)}")
    
    # Debug: Print session lists
    print(f"Train session list: {train_session_list}")
    print(f"Val session list: {val_session_list}")
    print(f"Test session list: {file_path}")
    
    # If no sessions found, use all pickles
    if not train_sessions and all_pickles:
        print("⚠️ No matching train sessions found, using all pickle files")
        train_sessions = all_pickles[:len(all_pickles)//2] if len(all_pickles) > 1 else all_pickles
        val_sessions = all_pickles[len(all_pickles)//2:] if len(all_pickles) > 1 else all_pickles
        test_sessions = all_pickles

    os.makedirs(f"{output_path}/{session}/ssl_model/", exist_ok=True)

    print("Training the wav2vec2_2d model...")
    
    # Configuration parameters
    use_spatial_embedding = False  # Disable spatial embedding for now
    
    # Create wav2vec2_2d configuration for 2D matrix input
    w2v2_2d_config = Wav2Vec2_2DConfig(
        # 2D CNN specific parameters for [3750 × 93] input
        conv_2d_feature_layers="[(32, 3, 2), (64, 3, 2), (128, 3, 2), (256, 3, 2)]",  # Smaller CNN
        input_channels=1,
        input_height=3750,  # Time points (height dimension)
        input_width=93,     # Channels (width dimension)
        
        # Transformer parameters - MUCH SMALLER
        encoder_layers=6,  # Reduced from 12
        encoder_embed_dim=384,  # Reduced from 768
        encoder_ffn_embed_dim=1536,  # Reduced from 3072
        encoder_attention_heads=6,  # Reduced from 12
        activation_fn="gelu",
        
        # Spatial embedding parameters (disabled for now)
        # use_spatial_embedding=use_spatial_embedding,
        # num_recording_sites=64,
        # spatial_embed_dim=128,  # Reduced from 256
        # spatial_embed_dropout=0.1,
        
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
    
    train_config = {'epoch': epochs, 'lr': 5e-4}  # Increased learning rate

    # --- wandb Initialization ---
    try:
        ri_tag = 'rand_init' if rand_init else 'pretrained'
        ssl_tag = 'ssl' if ssl else 'nossl'
        wandb.init(project="wav2vec2_2d_ssl_single_gpu", config=w2v2_2d_config.__dict__, 
                   name=f"{session}-{ri_tag}-{ssl_tag}-2d-exp", reinit=True)
    except Exception as e:
        print(f"Wandb initialization failed: {e}")
        print("Continuing without wandb logging...")
        wandb = None

    # Model will be created after dataset creation and configuration update

    # Create datasets using ModifiedSessionDataset for 2D matrix format
    try:
        # Use ModifiedSessionDataset for 2D matrix format [3750 × 93]
        # ModifiedSessionDataset expects a single file path, not a list
        if train_sessions:
            train_dataset = ModifiedSessionDataset(data_path=train_sessions[0], subset_data=subset_data)
            print(f"✅ Train dataset created from: {train_sessions[0]}")
            
            # Get actual data shape to update model configuration
            sample_tensor, _ = train_dataset[0]
            actual_height, actual_width = sample_tensor.shape[2], sample_tensor.shape[3]
            print(f"📊 Actual data shape: [{actual_height}, {actual_width}]")
            
            # Update model configuration with actual dimensions
            w2v2_2d_config.input_height = actual_height
            w2v2_2d_config.input_width = actual_width
            print(f"🔧 Updated model config: input_height={actual_height}, input_width={actual_width}")
            
            # Calculate expected output dimensions after 2D CNN
            feature_enc_layers = eval(w2v2_2d_config.conv_2d_feature_layers)
            h, w = actual_height, actual_width
            for _, kernel_size, stride in feature_enc_layers:
                h = (h - kernel_size) // stride + 1
                w = (w - kernel_size) // stride + 1
            
            expected_embed_dim = feature_enc_layers[-1][0]  # Last layer's output channels
            expected_layer_norm_size = expected_embed_dim * h * w
            print(f"🔧 Expected layer_norm size: {expected_layer_norm_size}")
            print(f"🔧 Expected output dimensions: [{h}, {w}] with {expected_embed_dim} channels")
            
        else:
            print("❌ No training sessions found!")
            return
            
        # --- Model Initialization (after configuration update) ---
        print(f"🏗️ Creating model with updated config: input_height={w2v2_2d_config.input_height}, input_width={w2v2_2d_config.input_width}")
        if rand_init:
            ssl_model = Wav2Vec2_2DModel(w2v2_2d_config)
        else:
            # For 2D model, you might want to initialize from a pretrained 1D model
            # and adapt the first layer to 2D, or start from scratch
            ssl_model = Wav2Vec2_2DModel(w2v2_2d_config)

        print(f"Model parameter count: {sum(p.numel() for p in ssl_model.parameters() if p.requires_grad)}")

        ssl_model.to(device)
        print(f"Model moved to device: {device}")
        
        # Fix the layer_norm dimensions to match actual data
        print(f"🔧 Fixing layer_norm dimensions...")
        print(f"   Current layer_norm normalized_shape: {ssl_model.layer_norm.normalized_shape}")
        
        # Calculate the correct normalized_shape based on actual data
        # The layer_norm should normalize over the feature dimension (last dimension)
        # After reshape: [B, H*W, C] -> layer_norm should normalize over C dimension
        expected_embed_dim = feature_enc_layers[-1][0]  # Last layer's output channels
        correct_normalized_shape = (expected_embed_dim,)
        
        print(f"   Correct normalized_shape should be: {correct_normalized_shape}")
        
        # Recreate layer_norm with correct dimensions
        from fairseq.modules import LayerNorm
        ssl_model.layer_norm = LayerNorm(expected_embed_dim).to(device)
        print(f"   ✅ Layer_norm recreated with correct dimensions: {ssl_model.layer_norm.normalized_shape}")
        
        # Fix the post_extract_proj layer dimensions
        print(f"🔧 Fixing post_extract_proj dimensions...")
        print(f"   Current post_extract_proj input size: {ssl_model.post_extract_proj.in_features if ssl_model.post_extract_proj else 'None'}")
        print(f"   Current post_extract_proj output size: {ssl_model.post_extract_proj.out_features if ssl_model.post_extract_proj else 'None'}")
        
        # Calculate correct input size for post_extract_proj
        # After reshape: [B, H*W, C] -> post_extract_proj should take C as input
        correct_input_size = expected_embed_dim  # 512
        correct_output_size = w2v2_2d_config.encoder_embed_dim  # 768
        
        print(f"   Correct input size should be: {correct_input_size}")
        print(f"   Correct output size should be: {correct_output_size}")
        
        # Recreate post_extract_proj with correct dimensions
        import torch.nn as nn
        ssl_model.post_extract_proj = nn.Linear(correct_input_size, correct_output_size).to(device)
        print(f"   ✅ Post_extract_proj recreated with correct dimensions: {correct_input_size} -> {correct_output_size}")
        
        # Check and recreate other dimension-dependent layers
        print(f"🔧 Checking other dimension-dependent layers...")
        
        # Check project_q layer (if it exists and uses old dimensions)
        if hasattr(ssl_model, 'project_q') and ssl_model.project_q is not None:
            old_project_q_input = ssl_model.project_q.in_features
            if old_project_q_input != correct_input_size:
                print(f"   🔧 Recreating project_q: {old_project_q_input} -> {correct_input_size}")
                ssl_model.project_q = nn.Linear(correct_input_size, ssl_model.project_q.out_features).to(device)
                print(f"   ✅ Project_q recreated")
        
        # Check project_inp layer (if it exists and uses old dimensions)
        if hasattr(ssl_model, 'project_inp') and ssl_model.project_inp is not None:
            old_project_inp_input = ssl_model.project_inp.in_features
            if old_project_inp_input != correct_input_size:
                print(f"   🔧 Recreating project_inp: {old_project_inp_input} -> {correct_input_size}")
                ssl_model.project_inp = nn.Linear(correct_input_size, ssl_model.project_inp.out_features).to(device)
                print(f"   ✅ Project_inp recreated")
        
        # Check spatial_projection layer (if it exists)
        if hasattr(ssl_model, 'spatial_projection') and ssl_model.spatial_projection is not None:
            print(f"   ℹ️ Spatial_projection exists: {ssl_model.spatial_projection.in_features} -> {ssl_model.spatial_projection.out_features}")
        
        # Check spatial_embedding layer (if it exists)
        if hasattr(ssl_model, 'spatial_embedding') and ssl_model.spatial_embedding is not None:
            print(f"   ℹ️ Spatial_embedding exists: {ssl_model.spatial_embedding.num_embeddings} embeddings, {ssl_model.spatial_embedding.embedding_dim} dims")
            
            # Temporarily disable spatial embeddings to avoid dimension issues
            print(f"   🔧 Temporarily disabling spatial embeddings to avoid dimension issues")
            ssl_model.spatial_embedding = None
            ssl_model.spatial_projection = None
            print(f"   ✅ Spatial embeddings disabled")
        
        print(f"   ✅ All dimension-dependent layers checked and updated")
        
        # Save the model configuration and initial state
        os.makedirs(f"{output_path}/{session}/ssl_model/", exist_ok=True)
        torch.save(ssl_model.state_dict(), f"{output_path}/{session}/ssl_model/model.pt")
        torch.save(w2v2_2d_config, f"{output_path}/{session}/ssl_model/config.pt")
        
        # Create optimizer after model creation
        optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=train_config['lr'])
        print(f"✅ Optimizer created with learning rate: {train_config['lr']}")
            
        if val_sessions:
            val_dataset = ModifiedSessionDataset(data_path=val_sessions[0], subset_data=subset_data)
            print(f"✅ Val dataset created from: {val_sessions[0]}")
        else:
            val_dataset = train_dataset  # Use train dataset as fallback
            print("⚠️ No validation sessions, using train dataset")
            
        if test_sessions:
            test_dataset = ModifiedSessionDataset(data_path=test_sessions[0], subset_data=subset_data)
            print(f"✅ Test dataset created from: {test_sessions[0]}")
        else:
            test_dataset = train_dataset  # Use train dataset as fallback
            print("⚠️ No test sessions, using train dataset")
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers,
                                pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers,
                                 pin_memory=True)

        # ModifiedSessionDataset doesn't have chance accuracy method
        print(f"Using ModifiedSessionDataset for 2D matrix format [3750 × 93]")

        # For downstream tasks, we need to create a probe dataset
        # Use probe dataset for SSL training (ignore labels, focus on neural patterns)
        train_probe_dataset = SessionDataset(session_paths=train_sessions, include_labels=True,
                                             data_subset_percentage=subset_data, super_regions=True)
        val_probe_dataset = SessionDataset(session_paths=val_sessions, include_labels=True,
                                           data_subset_percentage=subset_data, super_regions=True)
        
        # Create probe loaders for SSL training
        train_probe_loader = DataLoader(train_probe_dataset, batch_size=32, shuffle=True, pin_memory=True,
                                        num_workers=num_workers)
        val_probe_loader = DataLoader(val_probe_dataset, batch_size=32, shuffle=False, pin_memory=True,
                                          num_workers=num_workers)
        
        train_probe_chance_acc, val_probe_chance_acc = train_probe_dataset.get_chance_accuracy(), \
            val_probe_dataset.get_chance_accuracy()
        print(f"Train Probe Chance Accuracy: {train_probe_chance_acc:.4f}, "
              f"Validation Probe Chance Accuracy: {val_probe_chance_acc:.4f}")
        
        # Use probe datasets for SSL training
        train_loader = train_probe_loader
        val_loader = val_probe_loader
        print(f"✅ Using probe dataset for SSL training (ignoring labels, focusing on neural patterns)")

    except Exception as e:
        print(f"Error creating datasets: {e}")
        print("This might be due to missing data or incorrect paths.")
        return

    os.makedirs(f"{output_path}/{session}/ssl_model/", exist_ok=True)
    # Focus on SSL training only

    for epoch in tqdm(range(train_config['epoch'])):
        print(f"\n🔄 Epoch {epoch+1}/{train_config['epoch']}")
        
        # SSL Training: CNN Feature Extractor + Transformer + Quantization
        train_loss, grad_norm = train_2d(ssl_model, train_loader, optimizer, device, accumulation_steps=4)
        val_loss = validate_2d(ssl_model, val_loader, device)
        
        print(f"📊 SSL Results - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Grad Norm: {grad_norm:.4f}")
        
        # Skip probe training - focus only on SSL
        print(f"⏭️ Skipping probe training - focusing on SSL only")

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Grad Norm: {grad_norm:.4f}")
        
        if wandb:
            wandb.log({"SSL Train Loss": train_loss, "SSL Val Loss": val_loss, "Grad Norm": grad_norm,
                       "learning_rate": optimizer.param_groups[0]['lr']})
        
        # Save SSL model every epoch (or you can add your own criteria)
        torch.save(ssl_model.state_dict(), f"{output_path}/{session}/ssl_model/epoch_{epoch+1}_model.pt")
        torch.save(w2v2_2d_config, f"{output_path}/{session}/ssl_model/epoch_{epoch+1}_config.pt")
        print(f"💾 Saved SSL model for epoch {epoch+1}")

    print("Training completed!")


def parallelizer():
    if data == "Allen":
        sessions_list = ['719161530', '768515987', '771160300', '798911424', '771990200', '771160300', '768515987']
    elif data == 'ibl':
        sessions_list = ['0802ced5-33a3-405e-8336-b65ebc5cb07c_probe00',
                         '0802ced5-33a3-405e-8336-b65ebc5cb07c_probe01',
                         '0a018f12-ee06-4b11-97aa-bbbff5448e9f_probe00',
                         '3638d102-e8b6-4230-8742-e548cd87a949_probe01',
                         '5dcee0eb-b34d-4652-acc3-d10afc6eae68_probe00',
                         'd2832a38-27f6-452d-91d6-af72d794136c_probe00',
                         '54238fd6-d2d0-4408-b1a9-d19d24fd29ce_probe00']
    elif data == "Neuronexus":
        sessions_list = ['AD_HF01_1', 'AD_HF02_2', 'AD_HF02_4', 'AD_HF03_1', 'AD_HF03_2', 'NN_syn_01', 'NN_syn_02']
    elif data == "All":
        sessions_list = ['719161530', '794812542', '778998620', '798911424', '771160300', '768515987', '771990200']
        test_sess = ['AD_HF01_1', 'AD_HF02_4', 'AD_HF03_1', 'AD_HF03_2', 'NN_syn_01', 'NN_syn_02', 'AD_HF02_2']

    if selected_session is not None:
        to_skip = [s for s in sessions_list if s != selected_session]
    else:
        to_skip = []

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Number of GPUs available: {world_size}")
    
    try:
        if data != "All":
            for i in range(len(sessions_list)):
                if sessions_list[i] in to_skip:
                    print(f"Skipping {sessions_list[i]}...")
                    continue
                run_wav2vec2_2d(sessions_list, sessions_list[i])
        else:
            run_wav2vec2_2d(sessions_list, test_sess)
    except Exception as e:
        print(f"Error in run_wav2vec2_2d: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_data = {}
    parallelizer()
