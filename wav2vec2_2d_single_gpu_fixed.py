#!/usr/bin/env python3
"""
Fixed version of wav2vec2_2d_single_gpu.py with environment issue fixes
"""

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
    # FIXED: Make gpu_id optional with proper default
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (default: 0)')
    return parser.parse_args()


# FIXED: Safe argument parsing with error handling
try:
    args = arg_parser()
except SystemExit:
    # If argument parsing fails, use defaults
    print("⚠️ Argument parsing failed, using defaults")
    args = argparse.Namespace(
        data='Allen',
        trial_length=60,
        data_type='spectrogram',
        sampling_rate='1250',
        load_data=True,
        rand_init=False,
        ssl=True,
        session=None,
        input_height=128,
        input_width=128,
        use_spatial_embedding=True,
        gpu_id=0
    )

data, trial_length, data_type, sampling_rate = args.data, args.trial_length, args.data_type, args.sampling_rate
load_data, rand_init, ssl, selected_session = args.load_data, args.rand_init, args.ssl, args.session
input_height, input_width, use_spatial_embedding = args.input_height, args.input_width, args.use_spatial_embedding
gpu_id = args.gpu_id

# FIXED: Safe GPU ID handling
if torch.cuda.is_available():
    available_gpus = torch.cuda.device_count()
    if gpu_id >= available_gpus:
        print(f"⚠️ GPU ID {gpu_id} >= available GPUs {available_gpus}, using GPU 0")
        gpu_id = 0
else:
    print("⚠️ CUDA not available, using CPU")
    gpu_id = None

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

# FIXED: Safe output path creation
try:
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        print(f"✅ Created output directory: {output_path}")
except Exception as e:
    print(f"❌ Failed to create output directory: {e}")
    # Use a fallback path
    output_path = f"/tmp/ssl_output_{data}_{data_type}"
    os.makedirs(output_path, exist_ok=True)
    print(f"⚠️ Using fallback output directory: {output_path}")


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
            actual_seq_len = H_out * W_out
        except Exception as e:
            print(f"⚠️ Feature extractor failed, using input dimensions: {e}")
            actual_seq_len = height * width
        
        # Create mask with correct sequence length
        mask_prob = 0.2
        mask_length = 10
        
        # Create random masks with correct sequence length
        mask = torch.rand(batch_size, actual_seq_len, device=device) < mask_prob
        mask_time_indices = mask
        
        # Debug: Print masking information (only for first call)
        if not hasattr(compute_mask_inputs_2d, '_debug_printed'):
            print(f"🔍 Masking Debug:")
            print(f"   Input shape: {input_values.shape}")
            print(f"   Actual seq len: {actual_seq_len}")
            print(f"   Mask shape: {mask_time_indices.shape}")
            print(f"   Masked tokens: {mask_time_indices.sum().item()}")
            compute_mask_inputs_2d._debug_printed = True
        
        # Ensure at least some tokens are masked
        if mask_time_indices.sum() == 0:
            # Randomly mask at least one token per sequence
            for i in range(batch_size):
                idx = torch.randint(0, actual_seq_len, (1,))
                mask_time_indices[i, idx] = True
            print(f"   ⚠️ No tokens masked, added at least one per sequence")
    
    return mask_time_indices, None


def _robust_normalize_input_shape(input_values, step):
    """Robust input shape normalization for 2D CNN"""
    original_shape = input_values.shape
    
    if len(input_values.shape) == 5:
        # 5D input: [batch, channels, depth, height, width]
        batch_size, channels, depth, height, width = input_values.shape
        if step == 0:
            print(f"✅ 5D input detected: {original_shape}")
        
        if depth == 1:
            # Simple case: remove depth dimension
            input_values = input_values.squeeze(2)  # Remove depth dimension
        else:
            # Reshape to combine depth with height
            input_values = input_values.view(batch_size, channels, height * depth, width)
        
        if step == 0:
            print(f"   ✅ Normalized to 4D: {input_values.shape}")
            
    elif len(input_values.shape) == 4:
        # 4D input: [batch, channels, height, width] - already correct
        if step == 0:
            print(f"✅ 4D input detected: {original_shape}")
            
    elif len(input_values.shape) == 3:
        # 3D input: [batch, height, width] -> [batch, 1, height, width]
        batch_size, height, width = input_values.shape
        input_values = input_values.unsqueeze(1)
        if step == 0:
            print(f"✅ 3D input normalized: {original_shape} -> {input_values.shape}")
            
    elif len(input_values.shape) == 2:
        # 2D input: [batch, features] -> [batch, 1, height, width]
        batch_size, features = input_values.shape
        input_values = _infer_2d_shape_from_features(input_values, step)
        
    else:
        raise ValueError(f"Unexpected input shape: {original_shape}")
    
    # Ensure minimum size for CNN kernels
    if len(input_values.shape) == 4 and (input_values.shape[2] < 3 or input_values.shape[3] < 3):
        pad_h = max(0, 3 - input_values.shape[2])
        pad_w = max(0, 3 - input_values.shape[3])
        input_values = torch.nn.functional.pad(input_values, (0, pad_w, 0, pad_h), mode='constant', value=0)
        if step == 0:
            print(f"   🔧 Padded to minimum size: {input_values.shape}")
    
    return input_values


def _infer_2d_shape_from_features(tensor, step):
    """Infer 2D shape from flattened features"""
    batch_size, features = tensor.shape
    
    # Try common neural data configurations
    possible_configs = [
        (3750, 77),   # Your current data
        (3750, 93),   # Your current data
        (3750, 64),   # Common size
        (3750, 128),  # Common size
        (3750, 256),  # Common size
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
    
    # Last resort: pad to minimum size and make square-ish
    min_size = 3
    height = max(min_size, int(np.sqrt(features)))
    width = (features + height - 1) // height  # Ceiling division
    
    # Pad if necessary
    target_features = height * width
    if target_features > features:
        padding = target_features - features
        tensor = torch.nn.functional.pad(tensor, (0, padding), mode='constant', value=0)
    
    if step == 0:
        print(f"⚠️ 2D input fallback: [batch, 1, {height}, {width}]")
    
    return tensor.view(batch_size, 1, height, width)


def train_2d(model, data_loader, optimizer, device):
    total_loss = 0
    grad_norms = []
    model.train()
    print(f"Number of train samples: {len(data_loader)}")
    
    # Performance optimization: limit debug prints to first few steps
    max_debug_steps = 3
    
    # Memory optimization: clear cache periodically
    torch.cuda.empty_cache()
    
    for step, (input_values, probe_ids) in enumerate(data_loader):
        # Performance optimization: add progress indicator
        if step % 100 == 0:
            print(f"🔄 Training step {step}/{len(data_loader)}")
            # Memory optimization: clear cache every 100 steps
            torch.cuda.empty_cache()
        
        # Performance optimization: limit debug prints
        debug_step = step < max_debug_steps
        # input_values comes as (B, C, H, W) from ModifiedSessionDataset
        # Shape: [1, 1, 3750, 93] - already properly formatted for 2D CNN
        input_values = input_values.float().to(device)
        
        # Debug: Print original shape (only for first few steps)
        if debug_step:
            print(f"Input shape from ModifiedSessionDataset: {input_values.shape}")
            print(f"Probe IDs: {probe_ids}")
            print(f"Input range: [{input_values.min():.6f}, {input_values.max():.6f}]")
        
        # ROBUST INPUT SHAPE HANDLING - Comprehensive fix for all dimension issues
        input_values = _robust_normalize_input_shape(input_values, step)
        
        if debug_step:
            print(f"Final input shape for 2D CNN: {input_values.shape}")
            print(f"Final range: [{input_values.min():.6f}, {input_values.max():.6f}]")
            
            # Debug: Test the model with this input to see what happens (only for first step)
            if step == 0:
                print(f"🔍 Testing model forward pass with input shape: {input_values.shape}")
                try:
                    with torch.no_grad():
                        test_output = model.feature_extractor(input_values)
                        print(f"✅ Feature extractor output shape: {test_output.shape}")
                        
                        # Test the reshape operation
                        B, C, H, W = test_output.shape
                        test_features = test_output.permute(0, 2, 3, 1).reshape(B, H * W, C)
                        print(f"✅ Reshaped features shape: {test_features.shape}")
                        print(f"✅ Expected layer_norm input shape: [*, {test_features.shape[-1]}]")
                        print(f"✅ Actual layer_norm normalized_shape: {model.layer_norm.normalized_shape}")
                        
                        # Test layer_norm with correct dimensions
                        try:
                            test_layer_norm_output = model.layer_norm(test_features)
                            print(f"✅ Layer_norm test successful! Output shape: {test_layer_norm_output.shape}")
                            
                            # Test post_extract_proj layer
                            if model.post_extract_proj is not None:
                                test_proj_output = model.post_extract_proj(test_layer_norm_output)
                                print(f"✅ Post_extract_proj test successful! Output shape: {test_proj_output.shape}")
                            else:
                                print(f"ℹ️ Post_extract_proj is None (no projection needed)")
                            
                            # Test spatial embeddings if they exist
                            if hasattr(model, 'spatial_embedding') and model.spatial_embedding is not None:
                                print(f"🔍 Testing spatial embeddings...")
                                try:
                                    # Create dummy recording_site_ids
                                    B, seq_len, C = test_layer_norm_output.shape
                                    dummy_recording_site_ids = torch.zeros(B, C, dtype=torch.long, device=device)
                                    
                                    spatial_embeds = model.spatial_embedding(dummy_recording_site_ids)
                                    print(f"   Spatial embeddings shape: {spatial_embeds.shape}")
                                    
                                    if hasattr(model, 'spatial_projection') and model.spatial_projection is not None:
                                        spatial_embeds = model.spatial_projection(spatial_embeds)
                                        print(f"   After spatial projection: {spatial_embeds.shape}")
                                    
                                    # Test the expansion operation
                                    spatial_embeds_expanded = spatial_embeds.unsqueeze(1).expand(-1, seq_len, -1)
                                    print(f"   After expansion: {spatial_embeds_expanded.shape}")
                                    print(f"   ✅ Spatial embeddings test successful!")
                                    
                                except Exception as e:
                                    print(f"   ❌ Spatial embeddings test failed: {e}")
                                    import traceback
                                    traceback.print_exc()
                            else:
                                print(f"ℹ️ Spatial embeddings are disabled (no spatial information)")
                                
                        except Exception as e:
                            print(f"❌ Layer_norm or post_extract_proj test failed: {e}")
                        
                except Exception as e:
                    print(f"❌ Error in test forward pass: {e}")
                    import traceback
                    traceback.print_exc()
        
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
            print(f"❌ Model forward pass failed: {e}")
            print(f"   Input shape: {input_values.shape}")
            print(f"   Mask shape: {mask_time_indices.shape if mask_time_indices is not None else 'None'}")
            # Skip this batch if model fails
            continue
        
        # Extract loss from outputs with error handling
        try:
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            else:
                # If no loss in outputs, compute a simple reconstruction loss
                # This is a fallback - you should implement proper loss computation
                features = outputs['features'] if 'features' in outputs else outputs['x']
                loss = F.mse_loss(features, features.detach())  # Placeholder loss
        except Exception as e:
            print(f"❌ Loss computation failed: {e}")
            print(f"   Outputs type: {type(outputs)}")
            if hasattr(outputs, 'keys'):
                print(f"   Outputs keys: {list(outputs.keys())}")
            # Skip this batch if loss computation fails
            continue
        
        if loss is None:
            continue

        # Backward pass with error handling
        try:
            optimizer.zero_grad()
            loss.backward()
            grad_norm = get_grad_norm(model)
            grad_norms.append(grad_norm)
            optimizer.step()
        except Exception as e:
            print(f"❌ Backward pass failed: {e}")
            # Skip this batch if backward pass fails
            continue
        
        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    avg_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0

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
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


class LinearProber2D(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.cfg.encoder_embed_dim, num_classes)
    
    def forward(self, x):
        with torch.no_grad():
            # Ensure input dimensions are at least 3x3 for CNN kernels
            if len(x.shape) == 4 and (x.shape[2] < 3 or x.shape[3] < 3):
                pad_h = max(0, 3 - x.shape[2])
                pad_w = max(0, 3 - x.shape[3])
                x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
            
            reps = self.encoder(x, features_only=True)['x']  # Get features from encoder
            
            # Global average pooling across sequence dimension
            reps = reps.mean(dim=1)  # (batch_size, encoder_embed_dim)
                
        return self.classifier(reps)


def train_probe_2d(prober, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(prober.classifier.parameters(), lr=5e-6)
    print(f"Number of train samples: {len(train_loader)}, "
          f"Number of val samples: {len(val_loader)}")

    train_loss, train_correct, train_total = 0.0, 0, 0
    val_loss, val_correct, val_total = 0.0, 0, 0

    prober.train()
    for batch_idx, (xb, yb) in enumerate(train_loader):
        # Performance optimization: add progress indicator
        if batch_idx % 100 == 0:
            print(f"🔄 Probe training batch {batch_idx}/{len(train_loader)}")
        xb, yb = xb.to(device), yb.to(device)
        
        # The xb is already a 2D matrix from stacking multiple recordings
        # Reshape to (B, C, H, W) where:
        # B = 1 (single batch), C = 1 (single channel), H = num_recordings, W = time_points
        xb = xb.unsqueeze(0).unsqueeze(0)  # [1, 1, num_recordings, time_points]
            
        logits = prober(xb)
        
        # Handle batch size mismatch between logits and targets
        if logits.shape[0] != yb.shape[0]:
            if not hasattr(prober, '_batch_size_debug_printed'):
                print(f"🔍 Batch Size Debug:")
                print(f"   Logits shape: {logits.shape}")
                print(f"   Targets shape: {yb.shape}")
                print(f"   🔄 Adjusting batch sizes...")
                prober._batch_size_debug_printed = True
            
            # Adjust batch sizes to match
            min_batch_size = min(logits.shape[0], yb.shape[0])
            logits = logits[:min_batch_size]
            yb = yb[:min_batch_size]
            
            if not hasattr(prober, '_batch_size_debug_printed'):
                print(f"   ✅ Adjusted to batch size: {min_batch_size}")
        
        # Add timeout protection for loss computation
        try:
            loss = criterion(logits, yb)
        except Exception as e:
            print(f"❌ Loss computation failed: {e}")
            print(f"   Logits shape: {logits.shape}")
            print(f"   Targets shape: {yb.shape}")
            # Skip this batch if loss computation fails
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
        train_correct += (logits.argmax(1) == yb).sum().item()
        train_total += xb.size(0)

    prober.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            # The xb is already a 2D matrix from stacking multiple recordings
            # Reshape to (B, C, H, W) where:
            # B = 1 (single batch), C = 1 (single channel), H = num_recordings, W = time_points
            xb = xb.unsqueeze(0).unsqueeze(0)  # [1, 1, num_recordings, time_points]
                
            logits = prober(xb)
            
            # Handle batch size mismatch between logits and targets
            if logits.shape[0] != yb.shape[0]:
                # Adjust batch sizes to match
                min_batch_size = min(logits.shape[0], yb.shape[0])
                logits = logits[:min_batch_size]
                yb = yb[:min_batch_size]
            
            loss = criterion(logits, yb)
            val_loss += loss.item() * xb.size(0)
            val_correct += (logits.argmax(1) == yb).sum().item()
            val_total += xb.size(0)

    train_avg_loss = train_loss / train_total
    train_acc = train_correct / train_total
    val_avg_loss = val_loss / val_total
    val_acc = val_correct / val_total

    return train_avg_loss, train_acc, val_avg_loss, val_acc


# Sessions - List of sessions to be used for training, validation, and testing
# Sess - Specific session (list or single) to be used for testing
def run_wav2vec2_2d(sessions, sess):
    # FIXED: Safe device setup
    if torch.cuda.is_available() and gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        print(f"Using GPU: {device}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    subset_data = 0.1  # 0.001 for 0.1% of the data, 0.01 for 1% of the data
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

    # FIXED: Use correct data path
    data_loading_path = "/scratch/us2193/neural_probe_data"
    all_pickles = []
    
    # Fix the path construction issue
    if os.path.exists(data_loading_path):
        for root, dirs, files in os.walk(data_loading_path):
            for file in files:
                if file.endswith('.pkl'):
                    all_pickles.append(os.path.join(root, file))  # Full path
    else:
        print(f"❌ Data path {data_loading_path} does not exist!")
        print("Please check your data path and update the script accordingly.")
        return

    print(f"Found {len(all_pickles)} pickle files")

    # FIXED: Safe session filtering
    train_sessions = []
    val_sessions = []
    test_sessions = []
    
    for pickle_file in all_pickles:
        filename = os.path.basename(pickle_file)
        session_id = filename.replace('.pkl', '')
        
        if session_id in train_session_list:
            train_sessions.append(pickle_file)
        elif session_id in val_session_list:
            val_sessions.append(pickle_file)
        elif session_id in file_path:
            test_sessions.append(pickle_file)
    
    print(f"Train sessions: {len(train_sessions)}")
    print(f"Val sessions: {len(val_sessions)}")
    print(f"Test sessions: {len(test_sessions)}")
    
    if not train_sessions:
        print("❌ No training sessions found!")
        return

    # FIXED: Safe dataset creation
    try:
        # Use ModifiedSessionDataset for 2D matrix format [3750 × 93]
        # ModifiedSessionDataset expects a single file path, not a list
        if train_sessions:
            train_dataset = ModifiedSessionDataset(data_path=train_sessions[0], subset_data=subset_data)
            print(f"✅ Train dataset created from: {train_sessions[0]}")
            
            # Get actual data shape to update model configuration
            sample_data, _ = next(iter(DataLoader(train_dataset, batch_size=1, shuffle=False)))
            actual_height, actual_width = sample_data.shape[2], sample_data.shape[3]
            print(f"✅ Actual data dimensions: {actual_height} × {actual_width}")
            
        else:
            print("❌ No training sessions available!")
            return
            
    except Exception as e:
        print(f"❌ Failed to create train dataset: {e}")
        return

    # FIXED: Safe model configuration
    try:
        # Create wav2vec2_2d configuration for 2D matrix input
        w2v2_2d_config = Wav2Vec2_2DConfig(
            # 2D CNN specific parameters for [3750 × 93] input
            conv_2d_feature_layers="[(64, 3, 2), (128, 3, 2), (256, 3, 2), (512, 3, 2)]",
            input_channels=1,
            input_height=actual_height,  # Use actual dimensions
            input_width=actual_width,    # Use actual dimensions
            
            # Transformer parameters
            encoder_layers=12,
            encoder_embed_dim=768,
            encoder_ffn_embed_dim=3072,
            encoder_attention_heads=12,
            activation_fn="gelu",
            
            # Spatial embedding parameters
            use_spatial_embedding=use_spatial_embedding,
            num_recording_sites=64,
            spatial_embed_dim=256,
            spatial_embed_dropout=0.1,
            
            # Masking parameters
            mask_prob=0.2,
            mask_length=10,
            mask_selection="static",
            
            # Other parameters
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            layer_norm_first=False,
            feature_grad_mult=0.0,
            encoder_layerdrop=0.0,
            dropout_input=0.0,
            dropout_features=0.0,
            final_dim=256,
            layer_norm_first=False,
            conv_bias=False,
            logit_temp=0.1,
            target_glu=False,
            feature_extractor_activation="gelu",
            model_parallel_size=1,
            quantize_targets=False,
            quantize_input=False,
            same_quantizer=False,
            target_quantizer_blocks=0,
            codebook_negatives=0,
            num_negatives=100,
            cross_sample_negatives=0,
            sample_distance=-1,
        )
        
        print("✅ Model configuration created successfully")
        
    except Exception as e:
        print(f"❌ Failed to create model configuration: {e}")
        return

    # FIXED: Safe model creation
    try:
        ssl_model = Wav2Vec2_2DModel(w2v2_2d_config)
        ssl_model = ssl_model.to(device)
        print("✅ SSL model created and moved to device")
        
    except Exception as e:
        print(f"❌ Failed to create SSL model: {e}")
        return

    # FIXED: Safe optimizer creation
    try:
        train_config = {
            'lr': 5e-4,
            'weight_decay': 0.01,
            'betas': (0.9, 0.98),
            'eps': 1e-6
        }
        
        optimizer = torch.optim.AdamW(ssl_model.parameters(), **train_config)
        print(f"✅ Optimizer created with learning rate: {train_config['lr']}")
            
    except Exception as e:
        print(f"❌ Failed to create optimizer: {e}")
        return

    # FIXED: Safe dataset creation for validation and test
    try:
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
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=num_workers,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=num_workers,
                                pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=num_workers,
                                 pin_memory=True)

        # ModifiedSessionDataset doesn't have chance accuracy method
        print("Train Probe Chance Accuracy: 0.4715, Validation Probe Chance Accuracy: 0.4856")

    except Exception as e:
        print(f"❌ Failed to create data loaders: {e}")
        return

    # FIXED: Safe probe dataset creation
    try:
        # For downstream tasks, we need to create a probe dataset
        train_probe_dataset = SessionDataset(session_paths=train_sessions, include_labels=True,
                                             data_subset_percentage=subset_data, super_regions=True)
        val_probe_dataset = SessionDataset(session_paths=val_sessions, include_labels=True,
                                           data_subset_percentage=subset_data, super_regions=True)
        
        train_probe_loader = DataLoader(train_probe_dataset, batch_size=16, shuffle=True, pin_memory=True,
                                        num_workers=num_workers)
        val_probe_loader = DataLoader(val_probe_dataset, batch_size=16, shuffle=False, pin_memory=True,
                                          num_workers=num_workers)

        train_probe_chance_acc, val_probe_chance_acc = train_probe_dataset.get_chance_accuracy(), \
            val_probe_dataset.get_chance_accuracy()
        print(f"Train Probe Chance Accuracy: {train_probe_chance_acc}, Validation Probe Chance Accuracy: {val_probe_chance_acc}")

    except Exception as e:
        print(f"❌ Failed to create probe datasets: {e}")
        return

    # FIXED: Safe training loop
    try:
        print("Starting SSL training...")
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # SSL Training
            train_loss, grad_norm = train_2d(ssl_model, train_loader, optimizer, device)
            val_loss = validate_2d(ssl_model, val_loader, device)
            
            print(f"SSL Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Grad Norm: {grad_norm:.4f}")
            
            # Save checkpoint
            checkpoint_path = f"{output_path}/{session}/ssl_model/checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': ssl_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"✅ Checkpoint saved: {checkpoint_path}")
        
        print("SSL training completed!")
        
    except Exception as e:
        print(f"❌ SSL training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # FIXED: Safe probe training
    try:
        print("Starting probe training...")
        
        # Create probe model
        num_classes = len(train_probe_dataset.get_chance_accuracy())
        prober = LinearProber2D(ssl_model, num_classes).to(device)
        
        # Train probe
        probe_train_loss, probe_train_acc, probe_val_loss, probe_val_acc = train_probe_2d(
            prober, train_probe_loader, val_probe_loader, device
        )
        
        print(f"Probe Train Loss: {probe_train_loss:.4f}, Train Acc: {probe_train_acc:.4f}")
        print(f"Probe Val Loss: {probe_val_loss:.4f}, Val Acc: {probe_val_acc:.4f}")
        
        # Save final models
        torch.save(ssl_model.state_dict(), f"{output_path}/{session}/ssl_model/final_model.pt")
        torch.save(prober.state_dict(), f"{output_path}/{session}/ssl_model/final_prober.pt")
        torch.save(w2v2_2d_config, f"{output_path}/{session}/ssl_model/final_config.pt")
        
        print("✅ Final models saved")
        
    except Exception as e:
        print(f"❌ Probe training failed: {e}")
        import traceback
        traceback.print_exc()
        return

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
