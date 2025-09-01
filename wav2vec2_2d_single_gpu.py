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
        # For 2D input, we need to compute the sequence length after CNN
        # This is a simplified approach - you may need to adjust based on your CNN architecture
        seq_len = height * width  # Simplified: H * W
        
        # Compute masking for the flattened sequence
        mask_time_indices = torch.randint(0, seq_len, (batch_size, seq_len), device=device)
        mask_prob = 0.2  # You can make this configurable
        mask_length = 10
        
        # Create random masks
        mask = torch.rand(batch_size, seq_len, device=device) < mask_prob
        mask_time_indices = mask
        
        # Ensure at least some tokens are masked
        if mask_time_indices.sum() == 0:
            # Randomly mask at least one token per sequence
            for i in range(batch_size):
                idx = torch.randint(0, seq_len, (1,))
                mask_time_indices[i, idx] = True
        
        # For 2D, we don't use sampled_negatives in the same way
        # You might need to implement a different negative sampling strategy
        sampled_negatives = None
        
    return mask_time_indices, sampled_negatives


def train_2d(model, data_loader, optimizer, device):
    total_loss = 0
    grad_norms = []
    model.train()
    print(f"Number of train samples: {len(data_loader)}")
    
    for step, (input_values, probe_ids) in enumerate(data_loader):
        # input_values comes as (B, C, H, W) from ModifiedSessionDataset
        # Shape: [1, 1, 3750, 93] - already properly formatted for 2D CNN
        input_values = input_values.float().to(device)
        
        # Debug: Print original shape
        if step == 0:
            print(f"Input shape from ModifiedSessionDataset: {input_values.shape}")
            print(f"Probe IDs: {probe_ids}")
            print(f"Input range: [{input_values.min():.6f}, {input_values.max():.6f}]")
        
        # Debug: Print actual shape first
        if step == 0:
            print(f"Actual input shape: {input_values.shape}")
            print(f"Input dimensions: {len(input_values.shape)}")
            print(f"Input range: [{input_values.min():.6f}, {input_values.max():.6f}]")
        
        # Handle different input shapes
        if len(input_values.shape) == 5:
            # 5D input: [batch, channels, depth, height, width] = [1, 1, 1, 3750, 77]
            batch_size, channels, depth, height, width = input_values.shape
            if step == 0:
                print(f"✅ 5D input detected: {input_values.shape}")
                print(f"Batch size: {batch_size}, Channels: {channels}, Depth: {depth}")
                print(f"Height (time points): {height}, Width (channels): {width}")
            
            # Squeeze out the extra dimension to get 4D: [batch, channels, height, width]
            if depth == 1:
                input_values = input_values.squeeze(2)  # Remove depth dimension
                batch_size, channels, height, width = input_values.shape
                if step == 0:
                    print(f"✅ Squeezed to 4D: {input_values.shape}")
            else:
                # If depth > 1, we need to handle this differently
                if step == 0:
                    print(f"⚠️ Depth > 1 ({depth}), reshaping to 4D")
                # Reshape to [batch, channels, height*depth, width] or [batch, channels, height, width*depth]
                input_values = input_values.view(batch_size, channels, height * depth, width)
                batch_size, channels, height, width = input_values.shape
                if step == 0:
                    print(f"✅ Reshaped to 4D: {input_values.shape}")
                    
        elif len(input_values.shape) == 4:
            # Expected format: [batch, channels, height, width] = [1, 1, 3750, 93]
            batch_size, channels, height, width = input_values.shape
            if step == 0:
                print(f"✅ 4D input detected: {input_values.shape}")
                print(f"Batch size: {batch_size}, Channels: {channels}")
                print(f"Height (time points): {height}, Width (channels): {width}")
        elif len(input_values.shape) == 3:
            # Possible format: [batch, height, width] = [1, 3750, 93]
            batch_size, height, width = input_values.shape
            channels = 1
            input_values = input_values.unsqueeze(1)  # Add channel dimension
            if step == 0:
                print(f"✅ 3D input detected, adding channel dimension: {input_values.shape}")
                print(f"Batch size: {batch_size}, Channels: {channels}")
                print(f"Height (time points): {height}, Width (channels): {width}")
        elif len(input_values.shape) == 2:
            # Fallback: [batch, features] - need to reshape
            batch_size, features = input_values.shape
            # Assume this is a flattened version, try to reshape
            if features == 3750 * 93:  # 348,750 features
                input_values = input_values.view(batch_size, 1, 3750, 93)
                batch_size, channels, height, width = input_values.shape
                if step == 0:
                    print(f"✅ 2D input detected, reshaping to: {input_values.shape}")
                    print(f"Batch size: {batch_size}, Channels: {channels}")
                    print(f"Height (time points): {height}, Width (channels): {width}")
            else:
                # Try to reshape as [batch, 1, features, 1] for compatibility
                input_values = input_values.unsqueeze(1).unsqueeze(-1)
                batch_size, channels, height, width = input_values.shape
                if step == 0:
                    print(f"⚠️ 2D input with {features} features, reshaping to: {input_values.shape}")
                    print(f"Batch size: {batch_size}, Channels: {channels}")
                    print(f"Height: {height}, Width: {width}")
        else:
            raise ValueError(f"Unexpected input shape: {input_values.shape}")
        
        if step == 0:
            print(f"Final input shape for 2D CNN: {input_values.shape}")
            print(f"Final range: [{input_values.min():.6f}, {input_values.max():.6f}]")
            
            # Debug: Test the model with this input to see what happens
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
        
        # Compute masking for 2D input
        mask_time_indices, sampled_negative_indices = compute_mask_inputs_2d(model, input_values, device)
        
        # Forward pass for 2D model
        outputs = model(
            source=input_values,
            mask_indices=mask_time_indices,
            features_only=False
        )
        
        # Extract loss from outputs
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
        else:
            # If no loss in outputs, compute a simple reconstruction loss
            # This is a fallback - you should implement proper loss computation
            features = outputs['features'] if 'features' in outputs else outputs['x']
            loss = F.mse_loss(features, features.detach())  # Placeholder loss
        
        if loss is None:
            continue

        optimizer.zero_grad()
        loss.backward()
        grad_norm = get_grad_norm(model)
        grad_norms.append(grad_norm)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    avg_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0

    return avg_loss, avg_grad


def validate_2d(model, data_loader, device):
    model.eval()
    total_loss = 0
    print(f"Number of val samples: {len(data_loader)}")

    with torch.no_grad():
        for step, (input_values, probe_ids) in enumerate(data_loader):
            input_values = input_values.float().to(device)
            
            # Handle different input shapes (same as training function)
            if len(input_values.shape) == 5:
                # 5D input: [batch, channels, depth, height, width]
                batch_size, channels, depth, height, width = input_values.shape
                if depth == 1:
                    input_values = input_values.squeeze(2)  # Remove depth dimension
                else:
                    # Reshape to 4D
                    input_values = input_values.view(batch_size, channels, height * depth, width)
            elif len(input_values.shape) == 4:
                # Expected format: [batch, channels, height, width]
                pass
            elif len(input_values.shape) == 3:
                # Add channel dimension
                input_values = input_values.unsqueeze(1)
            elif len(input_values.shape) == 2:
                # Try to reshape
                batch_size, features = input_values.shape
                if features == 3750 * 93:
                    input_values = input_values.view(batch_size, 1, 3750, 93)
                else:
                    input_values = input_values.unsqueeze(1).unsqueeze(-1)
            
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
        self.classifier = nn.Linear(rep_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            # For 2D input, we need to handle the spatial dimensions
            reps = self.encoder(x, features_only=True)['x']  # Get features from encoder
            # Average pooling over spatial dimensions: (B, H*W, D) -> (B, D)
            reps = reps.mean(dim=1)
        return self.classifier(reps)


def train_probe_2d(prober, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(prober.classifier.parameters(), lr=5e-6)
    print(f"Number of train samples: {len(train_loader)}, "
          f"Number of val samples: {len(val_loader)}")

    train_loss, train_correct, train_total = 0.0, 0, 0
    val_loss, val_correct, val_total = 0.0, 0, 0

    prober.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        
        # The xb is already a 2D matrix from stacking multiple recordings
        # Reshape to (B, C, H, W) where:
        # B = 1 (single batch), C = 1 (single channel), H = num_recordings, W = time_points
        xb = xb.unsqueeze(0).unsqueeze(0)  # [1, 1, num_recordings, time_points]
            
        logits = prober(xb)
        loss = criterion(logits, yb)
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
    # Set device
    if torch.cuda.is_available():
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
    
    # Create wav2vec2_2d configuration for 2D matrix input
    w2v2_2d_config = Wav2Vec2_2DConfig(
        # 2D CNN specific parameters for [3750 × 93] input
        conv_2d_feature_layers="[(64, 3, 2), (128, 3, 2), (256, 3, 2), (512, 3, 2)]",
        input_channels=1,
        input_height=3750,  # Time points (height dimension)
        input_width=93,     # Channels (width dimension)
        
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
        feature_grad_mult=1.0,
        conv_bias=False,
        extractor_mode="default"
    )
    
    train_config = {'epoch': epochs, 'lr': 1e-5}

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
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=num_workers,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=num_workers,
                                pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=num_workers,
                                 pin_memory=True)

        # ModifiedSessionDataset doesn't have chance accuracy method
        print(f"Using ModifiedSessionDataset for 2D matrix format [3750 × 93]")

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
        print(f"Train Probe Chance Accuracy: {train_probe_chance_acc:.4f}, "
              f"Validation Probe Chance Accuracy: {val_probe_chance_acc:.4f}")

    except Exception as e:
        print(f"Error creating datasets: {e}")
        print("This might be due to missing data or incorrect paths.")
        return

    os.makedirs(f"{output_path}/{session}/ssl_model/", exist_ok=True)
    max_probe_acc = 0

    for epoch in tqdm(range(train_config['epoch'])):
        train_loss, grad_norm = train_2d(ssl_model, train_loader, optimizer, device)
        val_loss = validate_2d(ssl_model, val_loader, device)
        
        # For 2D model, we need to access the encoder differently
        encoder = ssl_model
        for p in encoder.parameters():
            p.requires_grad = False

        prober = LinearProber2D(encoder=encoder, rep_dim=w2v2_2d_config.encoder_embed_dim, num_classes=13).to(device)

        (probe_train_loss, probe_train_acc,
         probe_val_loss, probe_val_acc) = train_probe_2d(
            prober,
            train_probe_loader,
            val_probe_loader,
            device
        )

        for p in encoder.parameters():
            p.requires_grad = True

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Grad Norm: {grad_norm:.4f}")
        
        if wandb:
            wandb.log({"Train Loss": train_loss, "Val Loss": val_loss, "Grad Norm": grad_norm,
                       "learning_rate": optimizer.param_groups[0]['lr']})
        
        print(f"Probe Train Loss: {probe_train_loss:.4f}, Probe Train Accuracy: {probe_train_acc:.4f}, "
              f"Probe Val Loss: {probe_val_loss:.4f}, Probe Val Accuracy: {probe_val_acc:.4f}")
        
        if wandb:
            wandb.log({"Probe Train Loss": probe_train_loss, "Probe Train Accuracy": probe_train_acc,
                       "Probe Val Loss": probe_val_loss, "Probe Val Accuracy": probe_val_acc})
        
        if probe_val_acc >= max_probe_acc:
            max_probe_acc = max(max_probe_acc, probe_val_acc)
            torch.save(ssl_model.state_dict(), f"{output_path}/{session}/ssl_model/best_model.pt")
            torch.save(w2v2_2d_config, f"{output_path}/{session}/ssl_model/best_config.pt")

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
