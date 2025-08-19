import argparse
import os
import sys
import time
import pickle
import gc
from torch.distributed.elastic.multiprocessing import ElasticDataLoader, ElasticDistributedSampler
from torch.distributed.elastic.multiprocessing.errors import get_error_handler
import torch
import wandb
import tempfile



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spicy.signal import resample
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class RawDataLoader:
    def __init__(self, data_dir, sample_rate=16000, num_samples=16000):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.num_samples = num_samples

    def load_data(self):
        data = []


    def add_gaussian_noise(self, raw_data, signal_mean,gaussia_noise_level=0.0001):
        signal = raw_data.to(device)
        gaussian_noise = torch.normal(mean = signal_mean,
                                       std = gaussia_noise_level,
                                       shape = raw_data.shape,
                                       device = device)
        
        noisy_signal = signal + gaussian_noise
        return noisy_signal.to(device)
    
    
        return noisy_signal.to(device)
    
    def get_all_brain_regions(self, raw_data):
        brain_regions = []
        for i in range(raw_data.shape[0]):
            brain_regions.append(raw_data[i])
        return brain_regions
    
        
