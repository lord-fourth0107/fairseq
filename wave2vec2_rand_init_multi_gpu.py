# To run, use the command:
# torchrun --nproc_per_node=4 <> wav2vec_random_init_multi_gpu.py

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
from transformers import AutoConfig, Wav2Vec2Config, AutoFeatureExtractor, Wav2Vec2ForPreTraining, \
    AutoModelForAudioClassification, TrainingArguments, Trainer, pipeline, Wav2Vec2ForSequenceClassification
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
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
    parser = argparse.ArgumentParser(description='wave2vec2')
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
    return parser.parse_args()


args = arg_parser()
data, trial_length, data_type, sampling_rate = args.data, args.trial_length, args.data_type, args.sampling_rate,
load_data, rand_init, ssl, selected_session = args.load_data, args.rand_init, args.ssl, args.session
print(f"Data: {data}, Data Type: {data_type}, Trial Length: {trial_length}, Sampling Rate: {sampling_rate}")
print(f"Load Data: {load_data}, rand_init: {rand_init}, ssl: {ssl}, session: {selected_session}")
print("cuda is available: ", torch.cuda.is_available())

output_path = f"../results/{data}/{data_type}/wave2vec2/across_session"
if not os.path.exists(output_path):
    os.makedirs(output_path)


# data_loading_path = f'/scratch/mkp6112/LFP/region_decoding/results/ibl/spectrogram/wave2vec2/across_session'

def compute_mask_inputs(model, input_values, device):
    batch_size, raw_seq_len = input_values.shape
    with torch.no_grad():
        # Compute the feature extractor output length
        seq_len = model.module._get_feat_extract_output_lengths(raw_seq_len).item()
        # Compute masking
        mask_time_indices = _compute_mask_indices(
            (batch_size, seq_len),
            mask_prob=model.module.config.mask_time_prob,
            mask_length=model.module.config.mask_time_length
        )
        # print(mask_time_indices.shape, mask_time_indices.sum(), model.config.mask_time_prob, model.config.mask_time_length)
        assert (mask_time_indices.sum() > 0)
        sampled_negatives = _sample_negative_indices(
            (batch_size, seq_len),
            num_negatives=model.module.config.num_negatives,
            mask_time_indices=mask_time_indices
        )
        mask_time_indices = torch.tensor(mask_time_indices).to(device)
        sampled_negatives = torch.tensor(sampled_negatives).to(device)
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

def train(model, data_loader, optimizer, device):
    total_loss = 0
    grad_norms = []
    model.train()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[Rank {rank}] Number of train samples: {len(data_loader.sampler)}")
    for step, (input_values, _) in enumerate(data_loader):
        input_values = input_values.float().to(device)

        mask_time_indices, sampled_negative_indices = compute_mask_inputs(model, input_values, device)
        sampled_negative_indices = sampled_negative_indices.to(device)

        # print(f"[Rank {rank} | Step {step}] Forward pass...", flush=True)
        outputs = model(
            input_values=input_values,
            mask_time_indices=mask_time_indices,
            sampled_negative_indices=sampled_negative_indices
        )

        loss = outputs.loss
        if loss is None:
            # print(f"[Rank {rank} | Step {step}] Warning: Loss is None!", flush=True)
            continue

        # print(f"[Rank {rank} | Step {step}] Loss: {loss.item()}", flush=True)

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


def validate(model, data_loader, device):
    model.eval()
    total_loss = 0
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[Rank {rank}] Number of val samples: {len(data_loader.sampler)}")

    with torch.no_grad():
        for step, (input_values, _) in enumerate(data_loader):
            input_values = input_values.float().to(device)
            mask_time_indices, sampled_negative_indices = compute_mask_inputs(model, input_values, device)
            sampled_negative_indices = sampled_negative_indices.to(device)

            # print(f"[Rank {rank} | Step {step}] Validation pass...", flush=True)
            outputs = model(
                input_values=input_values,
                mask_time_indices=mask_time_indices,
                sampled_negative_indices=sampled_negative_indices
            )

            loss = outputs.loss
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


class LinearProber(nn.Module):
    def __init__(self, encoder, rep_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(rep_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            reps = self.encoder(x).last_hidden_state.detach()  # e.g., output shape: (B, T, D)
            reps = reps.mean(dim=1)  # first token pooling, also consider mean pooling
        return self.classifier(reps)


def train_probe(prober, train_loader, val_loader, device):
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
def run_wave2vec2(sessions, sess):
    rank, world_size, local_rank = ddp_setup()  # Initialize DDP
    device = torch.device(f"cuda:{local_rank}")  # Set the device for this process
    subset_data = 0.5  # 0.001 for 0.1% of the data, 0.01 for 1% of the data
    num_workers = 4  # 0 for single process, 4 for multi-process
    epochs = 10  # Number of epochs to train the model
    find_unused_parameters = True  # Whether to find unused parameters in the model
    print(f"Subset data: {subset_data}, Number of workers: {num_workers}, Epochs: {epochs}, "
          f"Find unused parameters: {find_unused_parameters}")
    # acronyms_arr = ['CA1', 'CA2', 'CA3', 'DG', 'Cortex']
    # label_json = {str(i): region for i, region in enumerate(acronyms_arr)}
    # print(label_json)

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

    print("Training the model...")
    w2v2_config = {"vocab_size": 32, "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
                   "intermediate_size": 3072, "hidden_act": "gelu", "hidden_dropout": 0.5,
                   "attention_dropout": 0.5, "final_dropout": 0.5, "initializer_range": 0.02,
                   "layer_norm_eps": 1e-12, "feat_extract_norm": "group", "feat_proj_dropout": 0.0,
                   "feat_extract_activation": "gelu", "feat_quantizer_dropout": 0.0,
                   "conv_dim": (512, 512, 512, 512, 512, 512, 512),
                   "conv_stride": (5, 2, 2, 2, 2, 2, 2), "conv_kernel": (10, 3, 3, 3, 3, 2, 2), "conv_bias": False,
                   "num_conv_pos_embeddings": 128, "num_conv_pos_embeddings_groups": 16, "do_stable_layer_norm": False,
                   "apply_spec_augment": True, "mask_time_prob": 0.05, "mask_time_length": 10, "mask_feature_prob": 0.0,
                   "mask_feature_length": 10, "num_codevectors_per_group": 320, "num_codevector_groups": 2,
                   "contrastive_logits_temperature": 0.1, "num_negatives": 100, "codevector_dim": 256,
                   "proj_codevector_dim": 256, "diversity_loss_weight": 0.1, "ctc_loss_reduction": "sum",
                   "ctc_zero_infinity": False, "use_weighted_layer_sum": False, "classifier_proj_size": 256}
    config = Wav2Vec2Config(**w2v2_config)
    config.update({
        'mask_time_min_masks': 2, 'mask_time_prob': 0.2,
        'random_init': rand_init, 'self_supervised': ssl})
    train_config = {'epoch': epochs, 'lr': 1e-5}

    # --- wandb Initialization (only on main process) ---
    if rank == 0:
        ri_tag = 'rand_init' if rand_init else 'pretrained'
        ssl_tag = 'ssl' if ssl else 'nossl'
        wandb.init(project="wav2vec_ssl_multi_gpu", config=w2v2_config, name=f"{session}-{ri_tag}-{ssl_tag}-exp", reinit=True)

    # --- Model Initialization (on each process) ---
    if rand_init:
        ssl_model = Wav2Vec2ForPreTraining(config=config)
    else:
        ssl_model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base", config=config)

    print(f"Model parameter count: {sum(p.numel() for p in ssl_model.parameters() if p.requires_grad)}")

    ssl_model.to(device)
    # Wrap the model with DDP
    ssl_model = DDP(ssl_model, device_ids=[local_rank], output_device=local_rank, 
                    find_unused_parameters=find_unused_parameters)
    print(f"Rank {rank}: Model wrapped in DDP and moved to device: {device}")
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
        train_loss, grad_norm = train(ssl_model, train_loader, optimizer, device)
        dist.barrier()  # Ensure all processes complete
        val_loss = validate(ssl_model, val_loader, device)

        dist.barrier()  # Ensure all processes complete
        encoder = ssl_model.module.wav2vec2
        for p in encoder.parameters():
            p.requires_grad = False

        prober = LinearProber(encoder=encoder, rep_dim=w2v2_config["hidden_size"], num_classes=13).to(device)
        prober = DDP(prober, device_ids=[local_rank], output_device=local_rank, 
                     find_unused_parameters=find_unused_parameters)

        (probe_train_loss, probe_train_acc,
         probe_val_loss, probe_val_acc) = train_probe(
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
        sessions_list = ['719161530', '768515987', '771160300', '798911424', '771990200', '771160300', '768515987']  # , '798911424', '771990200', '771160300', '768515987']
        # pickle_path = "/scratch/mkp6112/LFP/region_decoding/script/spectrogram/Allen"
    elif data == 'ibl':
        sessions_list = ['0802ced5-33a3-405e-8336-b65ebc5cb07c_probe00',
                         '0802ced5-33a3-405e-8336-b65ebc5cb07c_probe01',
                         '0a018f12-ee06-4b11-97aa-bbbff5448e9f_probe00',
                         '3638d102-e8b6-4230-8742-e548cd87a949_probe01',
                         '5dcee0eb-b34d-4652-acc3-d10afc6eae68_probe00',
                         'd2832a38-27f6-452d-91d6-af72d794136c_probe00',
                         '54238fd6-d2d0-4408-b1a9-d19d24fd29ce_probe00']
        # pickle_path = f'/vast/th3129/data/ibl_new/spectrogram_preprocessed'
    elif data == "Neuronexus":
        sessions_list = ['AD_HF01_1', 'AD_HF02_2', 'AD_HF02_4', 'AD_HF03_1', 'AD_HF03_2', 'NN_syn_01', 'NN_syn_02']
        # pickle_path = f'/scratch/th3129/region_decoding/data/Neuronexus/lfp'
    elif data == "All":
        sessions_list = ['719161530', '794812542', '778998620', '798911424', '771160300', '768515987', '771990200']
        test_sess = ['AD_HF01_1', 'AD_HF02_4', 'AD_HF03_1', 'AD_HF03_2', 'NN_syn_01', 'NN_syn_02', 'AD_HF02_2']
        # pickle_path = "spectrogram/Allen"

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
                run_wave2vec2(sessions_list, sessions_list[i])
        else:
            run_wave2vec2(sessions_list, test_sess)
    except Exception as e:
        error_handler.record_exception(e)
        print(f"Error in run_wave2vec2: {e}")


if __name__ == "__main__":
    debug_data = {}
    parallelizer()