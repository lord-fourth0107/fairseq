# Make sure to add comments to all files
# This makes Tianxiao's life easier when he tries to understand your code

from collections import OrderedDict

import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from scipy.signal import resample


class SessionDataset(Dataset):
    def __init__(self, session_paths, include_labels=False, target_sampling_rate=16000,
                 cache_size=2, data_subset_percentage=1.0, super_regions=False, dataset="Allen"):
        super().__init__()
        self.session_paths = session_paths
        self.include_labels = include_labels
        self.target_sampling_rate = target_sampling_rate
        self.signal_sampling_rate = 1250 if dataset != "Monkey" else 2500 # Sampling rate
        self.data_subset_percentage = data_subset_percentage
        self._recording_map = []

        if dataset != "Monkey":
            self._fine_label_map = {'APN': 0, 'BMAa': 1, 'CA1': 2, 'CA2': 3, 'CA3': 4, 'COAa': 5, 'COApm': 6, 'CP': 7,
                                    'DG': 8, 'Eth': 9, 'HPF': 10, 'IGL': 11, 'IntG': 12, 'LD': 13, 'LGd': 14, 'LGv': 15,
                                    'LP': 16, 'LT': 17, 'MB': 18, 'MGd': 19, 'MGm': 20, 'MGv': 21, 'MRN': 22, 'NOT': 23,
                                    'OLF': 24, 'OP': 25, 'PF': 26, 'PIL': 27, 'PO': 28, 'POL': 29, 'POST': 30, 'PP': 31,
                                    'PPT': 32, 'PRE': 33, 'PoT': 34, 'ProS': 35, 'RPF': 36, 'RT': 37, 'SCig': 38,
                                    'SCiw': 39, 'SCop': 40, 'SCsg': 41, 'SCzo': 42, 'SGN': 43, 'SUB': 44, 'TH': 45,
                                    'VIS': 46, 'VISal': 47, 'VISam': 48, 'VISl': 49, 'VISli': 50, 'VISmma': 51,
                                    'VISmmp': 52, 'VISp': 53, 'VISpm': 54, 'VISrl': 55, 'VL': 56, 'VPL': 57, 'VPM': 58,
                                    'ZI': 59, 'grey': 60, 'na': 61, 'nan': 62}
            self._coarse_label_map = {'VIS': 0, 'HPF': 1, 'OLF': 2, 'AMY': 3, 'CTX-STR': 4, 'TH-GEN': 5, 'VIS-TH': 6,
                                      'AUD-TH': 7, 'SM-TH': 8, 'RT-ZI': 9, 'MB-Tect': 10, 'MB-Teg': 11, 'Misc': 12}
            self._fine_to_coarse_label_map = {
                # Visual Cortex (VIS)
                'VIS': 'VIS', 'VISal': 'VIS', 'VISam': 'VIS', 'VISl': 'VIS', 'VISli': 'VIS',
                'VISp': 'VIS', 'VISpm': 'VIS', 'VISrl': 'VIS', 'VISmma': 'VIS', 'VISmmp': 'VIS',

                # Hippocampal Formation (HPF)
                'CA1': 'HPF', 'CA2': 'HPF', 'CA3': 'HPF', 'DG': 'HPF', 'HPF': 'HPF',
                'ProS': 'HPF', 'SUB': 'HPF',

                # Olfactory Cortex (OLF)
                'OLF': 'OLF', 'Eth': 'OLF',

                # Amygdala Complex (AMY)
                'BMAa': 'AMY', 'COAa': 'AMY', 'COApm': 'AMY',

                # Cortical Association & Striatum (CTX-STR)
                'CP': 'CTX-STR', 'POST': 'CTX-STR', 'PRE': 'CTX-STR', 'PP': 'CTX-STR',
                'RPF': 'CTX-STR', 'OP': 'CTX-STR', 'POL': 'CTX-STR',

                # General & Intralaminar Thalamus (TH-GEN)
                'TH': 'TH-GEN', 'PF': 'TH-GEN', 'PIL': 'TH-GEN', 'PoT': 'TH-GEN',

                # Visual Thalamus (VIS-TH)
                'IGL': 'VIS-TH', 'LD': 'VIS-TH', 'LGd': 'VIS-TH', 'LGv': 'VIS-TH', 'LP': 'VIS-TH',

                # Auditory Thalamus (AUD-TH)
                'MGd': 'AUD-TH', 'MGm': 'AUD-TH', 'MGv': 'AUD-TH',

                # Somatosensory & Motor Thalamus (SM-TH)
                'VPL': 'SM-TH', 'VPM': 'SM-TH', 'PO': 'SM-TH', 'VL': 'SM-TH',

                # Thalamic & Subthalamic Reticular Nuclei (RT-ZI)
                'RT': 'RT-ZI', 'ZI': 'RT-ZI',

                # Midbrain - Tectum & Pretectum (MB-Tect)
                'APN': 'MB-Tect', 'SCig': 'MB-Tect', 'SCiw': 'MB-Tect', 'SCop': 'MB-Tect',
                'SCsg': 'MB-Tect', 'SCzo': 'MB-Tect', 'NOT': 'MB-Tect',

                # Midbrain & Pontine Tegmentum (MB-Teg)
                'MB': 'MB-Teg', 'MRN': 'MB-Teg', 'PPT': 'MB-Teg', 'SGN': 'MB-Teg', 'LT': 'MB-Teg',

                # Miscellaneous (Misc)
                'IntG': 'Misc'
            }
        else:
            self._fine_label_map = {'imec0': 0, 'imec1': 1, 'imec2': 2}
            self._coarse_label_map = {'imec0': 0, 'imec1': 1, 'imec2': 2}
            self._fine_to_coarse_label_map = {'imec0': 'imec0', 'imec1': 'imec1', 'imec2': 'imec2'}

        self.super_regions = super_regions
        self.region_dict, self.trials = self._build_index()

        total = sum(self.region_dict.values())
        most_common = max(self.region_dict.values())
        self.chance_accuracy = most_common / total

        self.cache_size = cache_size
        self.cache = OrderedDict()

    def _build_index(self):
        """
        Builds a map of all recordings without loading the actual signal data.
        For caching...
        """
        region_count_dict = {}
        trials = []
        np.random.seed(42)
        print("Building recording index...")
        for session_path in self.session_paths:
            with open(session_path, 'rb') as f:
                # We load the session just to see how many recordings are inside
                # and to get the labels if needed. The actual signal data is not stored.
                session_data = pickle.load(f)
                for i, recording_tuple in enumerate(session_data):
                    random_number = np.random.rand()
                    if random_number > self.data_subset_percentage:
                        continue
                    label_str = recording_tuple[1].split('_')[-1]
                    # print(recording_tuple[1])

                    # If we are using super regions, we only keep the recordings that have a coarse label
                    if self.super_regions:
                        if label_str not in self._fine_to_coarse_label_map:
                            continue
                        label_str = self._fine_to_coarse_label_map[recording_tuple[1].split('_')[-1]]

                    # Each entry in our map is a reference to the session file and the
                    # index of the recording within that file's list.
                    self._recording_map.append((session_path, i))
                    trials.append(recording_tuple[1].split('_')[1])
                    region_count_dict[label_str] = region_count_dict.get(label_str, 0) + 1
        print(f"Index built. Found {len(self._recording_map)} total recordings.")
        return region_count_dict, trials

    def get_chance_accuracy(self):
        """
        Returns the chance accuracy of the dataset.
        This is the accuracy you would get if you randomly guessed the labels.
        """
        return self.chance_accuracy

    def get_label_counts(self):
        """
        Returns a dictionary with the counts of each label in the dataset.
        Useful for understanding the distribution of labels.
        """
        return self.region_dict

    def get_trials(self):
        """
        Returns a list of trials in the dataset.
        Useful for understanding the distribution of trials.
        """
        return self.trials

    def __len__(self):
        """
        Returns the total number of recordings across all sessions.
        """
        return len(self._recording_map)

    def __getitem__(self, idx):
        """
        Loads, preprocesses, and returns a single recording.

        Crucial for lazy loading and caching.
        """
        session_path, recording_index_in_session = self._recording_map[idx]

        # 1. Check if the session is in the cache (cache hit)
        if session_path in self.cache:
            # Move the accessed item to the end to mark it as most recently used
            self.cache.move_to_end(session_path)
            session_data = self.cache[session_path]

        # 2. If not in the cache (cache miss)
        else:
            # print(f"[Process {os.getpid()}] Cache miss. Loading {os.path.basename(session_path)}")
            # Load the new session from disk
            with open(session_path, 'rb') as f:
                session_data = pickle.load(f)

            # Check if the cache is full
            if len(self.cache) >= self.cache_size:
                # Evict the least recently used item (the first item in the OrderedDict)
                lru_path, _ = self.cache.popitem(last=False)
                # print(f"[Process {os.getpid()}] Cache full. Evicting {os.path.basename(lru_path)}")

            # Add the newly loaded session to the cache
            self.cache[session_path] = session_data

        recording_tuple = session_data[recording_index_in_session]
        signal = recording_tuple[0].astype(np.float32)

        num_target_samples = int(len(signal) * self.target_sampling_rate / self.signal_sampling_rate)
        upsampled_signal = resample(signal, num_target_samples)
        signal_tensor = torch.tensor(upsampled_signal, dtype=torch.float32)

        if not self.include_labels:
            return signal_tensor, ""
        else:
            label_str = recording_tuple[1].split('_')[-1]
            label_int = self._coarse_label_map[self._fine_to_coarse_label_map[label_str]]
            label_tensor = torch.tensor(label_int, dtype=torch.long)
            return signal_tensor, label_tensor
