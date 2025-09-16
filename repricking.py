#!/usr/bin/env python3
"""
Pickle File Structure Analyzer
Analyzes the structure and contents of a pickle file to understand its data organization.
"""

import pickle
import os
import numpy as np
import torch
from collections import defaultdict
import sys

def analyze_pickle_structure(file_path):
    """
    Analyze the structure of a pickle file and display detailed information.
    """
    print(f"Analyzing pickle file: {file_path}")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return
    
    try:
        # Load the pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        print(f"Data type: {type(data)}")
        print()
        
        if isinstance(data, dict):
            print("Dictionary structure:")
            print("-" * 30)
            for key, value in data.items():
                print(f"Key: {key}")
                print(f"  Type: {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"  Shape: {value.shape}")
                elif hasattr(value, '__len__'):
                    print(f"  Length: {len(value)}")
                print()
        
        elif isinstance(data, list):
            print("List structure:")
            print("-" * 30)
            print(f"Length: {len(data)}")
            if len(data) > 0:
                print(f"First element type: {type(data[0])}")
                if hasattr(data[0], 'shape'):
                    print(f"First element shape: {data[0].shape}")
                print(f"First few elements: {data[:3]}")
                
                # Analyze string labels if it's a list of tuples
                if isinstance(data[0], tuple) and len(data[0]) >= 2:
                    print("\nString label analysis:")
                    print("-" * 30)
                    
                    # Extract all labels
                    labels = [item[1] for item in data if isinstance(item, tuple) and len(item) >= 2]
                    print(f"Total labels: {len(labels)}")
                    
                    if labels:
                        # Analyze label structure
                        print(f"First 10 labels: {labels[:10]}")
                        print(f"Last 10 labels: {labels[-10:]}")
                        
                        # Parse labels to understand structure
                        parsed_labels = []
                        for label in labels[:100]:  # Sample first 100
                            if isinstance(label, str) and '_' in label:
                                parts = label.split('_')
                                if len(parts) >= 5:
                                    parsed_labels.append({
                                        'session': parts[0],
                                        'count': parts[1], 
                                        'probe': parts[2],
                                        'channel_id': parts[3],
                                        'brain_region': parts[4]
                                    })
                        
                        if parsed_labels:
                            print(f"\nParsed label structure (first 10):")
                            for i, parsed in enumerate(parsed_labels[:10]):
                                print(f"  {i+1}: {parsed}")
                            
                            # Count unique values
                            sessions = set(p['session'] for p in parsed_labels)
                            counts = set(p['count'] for p in parsed_labels)
                            probes = set(p['probe'] for p in parsed_labels)
                            channels = set(p['channel_id'] for p in parsed_labels)
                            regions = set(p['brain_region'] for p in parsed_labels)
                            
                            print(f"\nUnique counts:")
                            print(f"  Sessions: {len(sessions)} - {sorted(sessions)}")
                            print(f"  Counts: {len(counts)} - {sorted(counts)}")
                            print(f"  Probes: {len(probes)} - {sorted(probes)}")
                            print(f"  Channels: {len(channels)} - {sorted(channels)}")
                            print(f"  Brain regions: {len(regions)} - {sorted(regions)}")
        
        elif isinstance(data, np.ndarray):
            print("NumPy array structure:")
            print("-" * 30)
            print(f"Shape: {data.shape}")
            print(f"Dtype: {data.dtype}")
            print(f"Min value: {np.min(data)}")
            print(f"Max value: {np.max(data)}")
            print(f"Mean value: {np.mean(data):.4f}")
        
        elif isinstance(data, torch.Tensor):
            print("PyTorch tensor structure:")
            print("-" * 30)
            print(f"Shape: {data.shape}")
            print(f"Dtype: {data.dtype}")
            print(f"Device: {data.device}")
            print(f"Min value: {torch.min(data).item()}")
            print(f"Max value: {torch.max(data).item()}")
            print(f"Mean value: {torch.mean(data).item():.4f}")
        
        else:
            print("Other data type:")
            print("-" * 30)
            print(f"Type: {type(data)}")
            print(f"String representation: {str(data)[:200]}...")
        
        # If it's a dictionary, try to find common patterns
        if isinstance(data, dict):
            print("\nDetailed analysis of dictionary contents:")
            print("=" * 50)
            
            for key, value in data.items():
                print(f"\nKey: '{key}'")
                print(f"Type: {type(value)}")
                
                if isinstance(value, (list, tuple)):
                    print(f"Length: {len(value)}")
                    if len(value) > 0:
                        print(f"First element type: {type(value[0])}")
                        if hasattr(value[0], 'shape'):
                            print(f"First element shape: {value[0].shape}")
                
                elif isinstance(value, np.ndarray):
                    print(f"Shape: {value.shape}")
                    print(f"Dtype: {value.dtype}")
                    print(f"Min/Max: {np.min(value):.4f} / {np.max(value):.4f}")
                
                elif isinstance(value, torch.Tensor):
                    print(f"Shape: {value.shape}")
                    print(f"Dtype: {value.dtype}")
                    print(f"Min/Max: {torch.min(value).item():.4f} / {torch.max(value).item():.4f}")
                
                elif isinstance(value, str):
                    print(f"String length: {len(value)}")
                    print(f"Content preview: {value[:100]}...")
                
                else:
                    print(f"Value: {str(value)[:100]}...")
    
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        print("This might be due to:")
        print("1. File corruption")
        print("2. Incompatible Python version")
        print("3. Missing dependencies")
        print("4. File is not actually a pickle file")

def main():
    # Default path to the pickle file in Downloads
    downloads_path = os.path.expanduser("~/Downloads")
    pickle_file = "715093703_810755797.pickle"
    file_path = os.path.join(downloads_path, pickle_file)
    
    # Check if file exists in Downloads
    if not os.path.exists(file_path):
        print(f"File not found in Downloads: {file_path}")
        print("Please provide the correct path to your pickle file.")
        
        # Try to find the file in current directory
        if os.path.exists(pickle_file):
            file_path = pickle_file
            print(f"Found file in current directory: {file_path}")
        else:
            print("File not found. Please check the filename and path.")
            return
    
    analyze_pickle_structure(file_path)

if __name__ == "__main__":
    main()
