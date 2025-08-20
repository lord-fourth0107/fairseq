# To run, use the command:
# torchrun --nproc_per_node=4 wav2vec2_2d_multi_gpu.py

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

from torch.distributed.elastic.multiprocessing.errors import get_error_handler

import wandb
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy.signal import resample
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, Features, ClassLabel, Sequence, Value
from fairseq.models.wav2vec.wav2vec2_2d import Wav2Vec2_2DConfig, Wav2Vec2_2DModel
from fairseq.tasks.wav2vec_pretraining import Wav2VecPretrainingTask
from fairseq.criterions.wav2vec_criterion import Wav2VecCriterion
from fairseq.dataclass import FairseqDataclass
from blind_localization.data.PCAviz import PCAVisualizer
from blind_localization.data.lazyloader_dataset import SessionDataset
from tqdm import tqdm
from matplotlib.patches import Wedge
from scipy.special import softmax
import matplotlib
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp


def ddp_setup():
    """
    Initializes the distributed process group.
    """
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    return rank, world_size, local_rank

def ddp_cleanup():
    """
    Cleans up the distributed process group.
    """
    dist.destroy_process_group()

# Enter the path to the targeted input data
def arg_parser():
    parser = argparse.ArgumentParser(description='wav2vec2_2d')
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
    return parser.parse_args()


args = arg_parser()
data, trial_length, data_type, sampling_rate = args.data, args.trial_length, args.data_type, args.sampling_rate
load_data, rand_init, ssl, selected_session = args.load_data, args.rand_init, args.ssl, args.session
input_height, input_width, use_spatial_embedding = args.input_height, args.input_width, args.use_spatial_embedding
print(f"Data: {data}, Data Type: {data_type}, Trial Length: {trial_length}, Sampling Rate: {sampling_rate}")
print(f"Input Dimensions: {input_height}x{input_width}, Spatial Embedding: {use_spatial_embedding}")
print(f"Load Data: {load_data}, rand_init: {rand_init}, ssl: {ssl}, session: {selected_session}")
print("cuda is available: ", torch.cuda.is_available())

output_path = f"../results/{data}/{data_type}/wav2vec2_2d/across_session"
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


def gather_metric(metric, device):
    """Gathers a metric tensor from all processes and averages it."""
    metric_tensor = torch.tensor(metric).to(device)
    dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
    return metric_tensor.item() / int(os.environ["WORLD_SIZE"])


def reduce_tensor(tensor, world_size):
    rt = tensor.clone().float()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def train_2d(model, data_loader, optimizer, device):
    total_loss = 0
    grad_norms = []
    model.train()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[Rank {rank}] Number of train samples: {len(data_loader.sampler)}")
    
    for step, (input_values, _) in enumerate(data_loader):
        # input_values should be (B, C, H, W) for 2D
        input_values = input_values.float().to(device)
        
        # Ensure input is 4D: (B, C, H, W)
        if input_values.dim() == 3:
            input_values = input_values.unsqueeze(1)  # Add channel dimension
        
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
    avg_loss = torch.tensor(avg_loss, device=device)
    avg_loss = reduce_tensor(avg_loss, world_size).item()

    avg_grad = sum(grad_norms) / len(grad_norms)
    avg_grad = torch.tensor(avg_grad, device=device)
    avg_grad = reduce_tensor(avg_grad, world_size).item()

    return avg_loss, avg_grad


def validate_2d(model, data_loader, device):
    model.eval()
    total_loss = 0
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[Rank {rank}] Number of val samples: {len(data_loader.sampler)}")

    with torch.no_grad():
        for step, (input_values, _) in enumerate(data_loader):
            input_values = input_values.float().to(device)
            
            # Ensure input is 4D: (B, C, H, W)
            if input_values.dim() == 3:
                input_values = input_values.unsqueeze(1)  # Add channel dimension
            
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
    avg_loss = torch.tensor(avg_loss, device=device)
    avg_loss = reduce_tensor(avg_loss, world_size).item()
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
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(prober.module.classifier.parameters(), lr=5e-6)
    print(f"[Rank {rank}] Number of train samples: {len(train_loader.sampler)}, "
          f"Number of val samples: {len(val_loader.sampler)}")

    train_loss, train_correct, train_total = 0.0, 0, 0
    val_loss, val_correct, val_total = 0.0, 0, 0

    prober.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        
        # Ensure input is 4D: (B, C, H, W)
        if xb.dim() == 3:
            xb = xb.unsqueeze(1)
            
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
            
            # Ensure input is 4D: (B, C, H, W)
            if xb.dim() == 3:
                xb = xb.unsqueeze(1)
                
            logits = prober(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * xb.size(0)
            val_correct += (logits.argmax(1) == yb).sum().item()
            val_total += xb.size(0)

    train_loss = reduce_tensor(torch.tensor(train_loss, device=device), world_size).item()
    train_correct = reduce_tensor(torch.tensor(train_correct, device=device), world_size).item()
    train_total = reduce_tensor(torch.tensor(train_total, device=device), world_size).item()

    val_loss = reduce_tensor(torch.tensor(val_loss, device=device), world_size).item()
    val_correct = reduce_tensor(torch.tensor(val_correct, device=device), world_size).item()
    val_total = reduce_tensor(torch.tensor(val_total, device=device), world_size).item()

    train_avg_loss = train_loss / train_total
    train_acc = train_correct / train_total
    val_avg_loss = val_loss / val_total
    val_acc = val_correct / val_total

    return train_avg_loss, train_acc, val_avg_loss, val_acc


# Sessions - List of sessions to be used for training, validation, and testing
# Sess - Specific session (list or single) to be used for testing
def run_wav2vec2_2d(sessions, sess):
    rank, world_size, local_rank = ddp_setup()  # Initialize DDP
    device = torch.device(f"cuda:{local_rank}")  # Set the device for this process
    subset_data = 0.5  # 0.001 for 0.1% of the data, 0.01 for 1% of the data
    num_workers = 4  # 0 for single process, 4 for multi-process
    epochs = 10  # Number of epochs to train the model
    find_unused_parameters = True  # Whether to find unused parameters in the model
    print(f"Subset data: {subset_data}, Number of workers: {num_workers}, Epochs: {epochs}, "
          f"Find unused parameters: {find_unused_parameters}")

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

    data_loading_path = "Allen_w2v2/Allen"
    all_pickles = []
    for root, dirs, files in os.walk(data_loading_path):
        for file in files:
            if file.endswith('.pickle'):
                all_pickles.append(os.path.join(file))

    train_sessions = [os.path.join("Allen_w2v2/Allen", item) for item in all_pickles
                      if any(item.startswith(session) for session in train_session_list)]
    val_sessions = [os.path.join("Allen_w2v2/Allen", item) for item in all_pickles
                    if any(item.startswith(session) for session in val_session_list)]
    test_sessions = [os.path.join("Allen_w2v2/Allen", item) for item in all_pickles
                     if any(item.startswith(session) for session in file_path)]

    os.makedirs(f"{output_path}/{session}/ssl_model/", exist_ok=True)

    print("Training the wav2vec2_2d model...")
    
    # Create wav2vec2_2d configuration
    w2v2_2d_config = Wav2Vec2_2DConfig(
        # 2D CNN specific parameters
        conv_2d_feature_layers="[(64, 3, 2), (128, 3, 2), (256, 3, 2), (512, 3, 2)]",
        input_channels=1,
        input_height=input_height,
        input_width=input_width,
        
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

    # --- wandb Initialization (only on main process) ---
    if rank == 0:
        ri_tag = 'rand_init' if rand_init else 'pretrained'
        ssl_tag = 'ssl' if ssl else 'nossl'
        wandb.init(project="wav2vec2_2d_ssl_multi_gpu", config=w2v2_2d_config.__dict__, 
                   name=f"{session}-{ri_tag}-{ssl_tag}-2d-exp", reinit=True)

    # --- Model Initialization (on each process) ---
    if rand_init:
        ssl_model = Wav2Vec2_2DModel(w2v2_2d_config)
    else:
        # For 2D model, you might want to initialize from a pretrained 1D model
        # and adapt the first layer to 2D, or start from scratch
        ssl_model = Wav2Vec2_2DModel(w2v2_2d_config)

    print(f"Model parameter count: {sum(p.numel() for p in ssl_model.parameters() if p.requires_grad)}")

    ssl_model.to(device)
    # Wrap the model with DDP
    ssl_model = DDP(ssl_model, device_ids=[local_rank], output_device=local_rank, 
                    find_unused_parameters=find_unused_parameters)
    print(f"Rank {rank}: Model wrapped in DDP and moved to device: {device}")
    
    # Save the model configuration
    if rank == 0:
        ssl_model.module.save_pretrained(f"{output_path}/{session}/ssl_model/")

    # self supervised pretraining
    optimizer = torch.optim.AdamW(ssl_model.module.parameters(), lr=train_config['lr'])

    train_dataset = SessionDataset(session_paths=train_sessions, include_labels=False,
                                   data_subset_percentage=subset_data)
    val_dataset = SessionDataset(session_paths=val_sessions, include_labels=False,
                                 data_subset_percentage=subset_data)
    test_dataset = SessionDataset(session_paths=test_sessions, include_labels=False,
                                  data_subset_percentage=subset_data)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=num_workers,
                              pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers,
                            pin_memory=True, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers,
                             pin_memory=True, sampler=test_sampler)

    train_chance_acc, val_chance_acc, test_chance_acc = train_dataset.get_chance_accuracy(), \
        val_dataset.get_chance_accuracy(), test_dataset.get_chance_accuracy()
    print(f"Train Chance Accuracy: {train_chance_acc:.4f}, "
          f"Validation Chance Accuracy: {val_chance_acc:.4f}, "
          f"Test Chance Accuracy: {test_chance_acc:.4f}")

    # For downstream tasks, we need to create a probe dataset
    train_probe_dataset = SessionDataset(session_paths=train_sessions, include_labels=True,
                                         data_subset_percentage=subset_data, super_regions=True)
    val_probe_dataset = SessionDataset(session_paths=val_sessions, include_labels=True,
                                       data_subset_percentage=subset_data, super_regions=True)
    train_probe_sampler = DistributedSampler(train_probe_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_probe_sampler = DistributedSampler(val_probe_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_probe_loader = DataLoader(train_probe_dataset, batch_size=32, shuffle=False, pin_memory=True,
                                    num_workers=num_workers, sampler=train_probe_sampler)
    val_probe_loader = DataLoader(val_probe_dataset, batch_size=32, shuffle=False, pin_memory=True,
                                  num_workers=num_workers, sampler=val_probe_sampler)

    train_probe_chance_acc, val_probe_chance_acc = train_probe_dataset.get_chance_accuracy(), \
        val_probe_dataset.get_chance_accuracy()
    print(f"Train Probe Chance Accuracy: {train_probe_chance_acc:.4f}, "
          f"Validation Probe Chance Accuracy: {val_probe_chance_acc:.4f}")

    os.makedirs(f"{output_path}/{session}/ssl_model/", exist_ok=True)
    max_probe_acc = 0

    for epoch in tqdm(range(train_config['epoch'])):
        train_loader.sampler.set_epoch(epoch)
        train_loss, grad_norm = train_2d(ssl_model, train_loader, optimizer, device)
        dist.barrier()  # Ensure all processes complete
        val_loss = validate_2d(ssl_model, val_loader, device)

        dist.barrier()  # Ensure all processes complete
        
        # For 2D model, we need to access the encoder differently
        encoder = ssl_model.module
        for p in encoder.parameters():
            p.requires_grad = False

        prober = LinearProber2D(encoder=encoder, rep_dim=w2v2_2d_config.encoder_embed_dim, num_classes=13).to(device)
        prober = DDP(prober, device_ids=[local_rank], output_device=local_rank, 
                     find_unused_parameters=find_unused_parameters)

        (probe_train_loss, probe_train_acc,
         probe_val_loss, probe_val_acc) = train_probe_2d(
            prober,
            train_probe_loader,
            val_probe_loader,
            device
        )

        for p in encoder.parameters():
            p.requires_grad = True

        dist.barrier()  # Ensure all processes complete

        if rank == 0:
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Grad Norm: {grad_norm:.4f}")
            wandb.log({"Train Loss": train_loss, "Val Loss": val_loss, "Grad Norm": grad_norm,
                       "learning_rate": optimizer.param_groups[0]['lr']})
            print(f"Probe Train Loss: {probe_train_loss:.4f}, Probe Train Accuracy: {probe_train_acc:.4f}, "
                  f"Probe Val Loss: {probe_val_loss:.4f}, Probe Val Accuracy: {probe_val_acc:.4f}")
            wandb.log({"Probe Train Loss": probe_train_loss, "Probe Train Accuracy": probe_train_acc,
                       "Probe Val Loss": probe_val_loss, "Probe Val Accuracy": probe_val_acc})
            if probe_val_acc >= max_probe_acc:
                max_probe_acc = max(max_probe_acc, probe_val_acc)
                ssl_model.module.save_pretrained(f"{output_path}/{session}/ssl_model/")

        dist.barrier()  # Ensure all processes complete

    ddp_cleanup()


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

    error_handler = get_error_handler()
    world_size = torch.cuda.device_count()
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
        error_handler.record_exception(e)
        print(f"Error in run_wav2vec2_2d: {e}")


if __name__ == "__main__":
    debug_data = {}
    parallelizer()
