#!/usr/bin/env python3
"""
Enrich Pickle Files with CCF Coordinates
Combines information from joined.csv and channels.csv to add proper coordinates to pickle file labels.
Enhanced version that can process all pickle files in a folder.
"""

import pickle
import os
import pandas as pd
import shutil
import argparse
import glob
import multiprocessing as mp
from collections import defaultdict
from functools import partial
from tqdm import tqdm

def load_coordinate_data(input_path):
    """Load and merge coordinate data from both CSV files."""
    print("Loading coordinate data...")
    
    # Determine the directory to look for CSV files
    if os.path.isfile(input_path):
        # If input is a file, look in its directory
        csv_dir = os.path.dirname(input_path)
    else:
        # If input is a directory, look in that directory
        csv_dir = input_path
    
    # Look for CSV files in the input directory
    joined_path = os.path.join(csv_dir, 'joined.csv')
    channels_path = os.path.join(csv_dir, 'channels.csv')
    
    # Check if files exist
    if not os.path.exists(joined_path):
        print(f"Error: joined.csv not found in {csv_dir}")
        print("Please ensure joined.csv is in the same directory as your pickle files")
        return None
    
    if not os.path.exists(channels_path):
        print(f"Error: channels.csv not found in {csv_dir}")
        print("Please ensure channels.csv is in the same directory as your pickle files")
        return None
    
    print(f"Found joined.csv at: {joined_path}")
    print(f"Found channels.csv at: {channels_path}")
    
    joined_df = pd.read_csv(joined_path, dtype=str)
    channels_df = pd.read_csv(channels_path, dtype=str)
    
    print(f"Loaded joined.csv: {joined_df.shape}")
    print(f"Loaded channels.csv: {channels_df.shape}")
    
    # Rename columns to avoid conflicts before merging
    joined_renamed = joined_df[['session_id', 'probe_id']].copy()
    channels_renamed = channels_df[['ecephys_probe_id', 'local_index', 'probe_horizontal_position', 
                                   'probe_vertical_position', 'anterior_posterior_ccf_coordinate', 
                                   'dorsal_ventral_ccf_coordinate', 'left_right_ccf_coordinate']].copy()
    
    # Merge on probe_id to get session info
    merged_df = pd.merge(
        joined_renamed,
        channels_renamed,
        left_on='probe_id', 
        right_on='ecephys_probe_id', 
        how='inner'
    )
    
    print(f"Merged data shape: {merged_df.shape}")
    
    # Create a lookup dictionary: (session_id, probe_id, local_index) -> coordinates
    coord_lookup = {}
    for _, row in merged_df.iterrows():
        key = (row['session_id'], row['probe_id'], str(row['local_index']))
        coord_lookup[key] = {
            'ap': row['anterior_posterior_ccf_coordinate'],
            'dv': row['dorsal_ventral_ccf_coordinate'],
            'lr': row['left_right_ccf_coordinate'],
            'probe_h': row['probe_horizontal_position'],
            'probe_v': row['probe_vertical_position']
        }
    
    print(f"Created coordinate lookup with {len(coord_lookup)} entries")
    return coord_lookup

def enrich_pickle_file(pickle_path, coord_lookup):
    """Enrich a pickle file with coordinate information."""
    print(f"\nProcessing pickle file: {pickle_path}")
    
    # Create backup
    backup_path = pickle_path + '.backup'
    if not os.path.exists(backup_path):
        shutil.copy2(pickle_path, backup_path)
        print(f"Created backup: {backup_path}")
    
    # Load the pickle file
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} entries")
    
    # Process each entry
    enriched_data = []
    stats = {
        'total': 0,
        'enriched': 0,
        'not_found': 0,
        'parse_error': 0
    }
    
    for entry in data:
        stats['total'] += 1
        
        if isinstance(entry, tuple) and len(entry) >= 2:
            signal, label = entry[0], entry[1]
            
            # Parse the label to extract session, count, probe, channel_id
            if isinstance(label, str) and '_' in label:
                parts = label.split('_')
                if len(parts) >= 4:
                    try:
                        session_id = parts[0]
                        count = parts[1]
                        probe_id = parts[2]
                        channel_id = parts[3]
                        
                        # Look up coordinates
                        lookup_key = (session_id, probe_id, channel_id)
                        if lookup_key in coord_lookup:
                            coords = coord_lookup[lookup_key]
                            # Append coordinates to the label
                            enriched_label = f"{label}_{coords['ap']}_{coords['dv']}_{coords['lr']}_{coords['probe_h']}_{coords['probe_v']}"
                            enriched_data.append((signal, enriched_label))
                            stats['enriched'] += 1
                        else:
                            # If not found, try with local_index instead of channel_id
                            # Get all available local_index values for this (session_id, probe_id)
                            available_indices = [k[2] for k in coord_lookup.keys() 
                                              if k[0] == session_id and k[1] == probe_id]
                            
                            if available_indices:
                                # Use modulo to cycle through available indices
                                try:
                                    channel_idx = int(channel_id)
                                    coord_idx = channel_idx % len(available_indices)
                                    selected_index = available_indices[coord_idx]
                                    lookup_key = (session_id, probe_id, selected_index)
                                    
                                    if lookup_key in coord_lookup:
                                        coords = coord_lookup[lookup_key]
                                        enriched_label = f"{label}_{coords['ap']}_{coords['dv']}_{coords['lr']}_{coords['probe_h']}_{coords['probe_v']}"
                                        enriched_data.append((signal, enriched_label))
                                        stats['enriched'] += 1
                                    else:
                                        enriched_data.append(entry)  # Keep original
                                        stats['not_found'] += 1
                                except ValueError:
                                    enriched_data.append(entry)  # Keep original
                                    stats['parse_error'] += 1
                            else:
                                enriched_data.append(entry)  # Keep original
                                stats['not_found'] += 1
                    except Exception as e:
                        enriched_data.append(entry)  # Keep original
                        stats['parse_error'] += 1
                else:
                    enriched_data.append(entry)  # Keep original
                    stats['parse_error'] += 1
            else:
                enriched_data.append(entry)  # Keep original
                stats['parse_error'] += 1
        else:
            enriched_data.append(entry)  # Keep original
    
    # Print statistics
    print(f"Enrichment statistics:")
    print(f"  Total entries: {stats['total']}")
    print(f"  Successfully enriched: {stats['enriched']}")
    print(f"  Not found in lookup: {stats['not_found']}")
    print(f"  Parse errors: {stats['parse_error']}")
    print(f"  Success rate: {stats['enriched']/stats['total']*100:.1f}%")
    
    # Save the enriched data
    with open(pickle_path, 'wb') as f:
        pickle.dump(enriched_data, f)
    
    print(f"Saved enriched data to: {pickle_path}")
    
    return stats

def _looks_like_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def is_label_enriched(label: str) -> bool:
    """Heuristically determine if a label already has appended coordinates.
    Expected enriched label suffix: _{ap}_{dv}_{lr}_{probe_h}_{probe_v}
    We check that the label has at least 9 underscore-separated parts and the
    last 5 parts are numeric-like.
    """
    if not isinstance(label, str):
        return False
    parts = label.split('_')
    if len(parts) < 9:
        return False
    tail = parts[-5:]
    return all(_looks_like_number(x) for x in tail)


def should_skip_pickle(pickle_path: str, sample_size: int = 50, threshold: float = 0.9) -> bool:
    """Load a sample of entries and decide if the pickle is already enriched.
    - sample_size: number of entries to sample from the start (or all if fewer)
    - threshold: fraction of sampled labels that must be enriched to skip
    """
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, (list, tuple)) or len(data) == 0:
            return False
        n = min(len(data), sample_size)
        enriched_flags = 0
        checked = 0
        for i in range(n):
            entry = data[i]
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                label = entry[1]
                checked += 1
                if is_label_enriched(label):
                    enriched_flags += 1
        if checked == 0:
            return False
        frac = enriched_flags / checked
        return frac >= threshold
    except Exception:
        return False

def process_single_pickle_file(args):
    """Worker function to process a single pickle file."""
    pickle_path, coord_lookup = args
    
    try:
        # Process all files without checking if already enriched
        stats = enrich_pickle_file(pickle_path, coord_lookup)
        stats['skipped'] = False
        return {
            'file': pickle_path,
            'success': True,
            'stats': stats,
            'error': None
        }
    except Exception as e:
        return {
            'file': pickle_path,
            'success': False,
            'stats': None,
            'error': str(e)
        }

def find_pickle_files(input_path):
    """Find all pickle files in the given path."""
    if os.path.isfile(input_path):
        # Single file
        if input_path.endswith('.pickle'):
            return [input_path]
        else:
            print(f"Error: {input_path} is not a pickle file")
            return []
    elif os.path.isdir(input_path):
        # Directory - find all pickle files
        pattern = os.path.join(input_path, "*.pickle")
        pickle_files = glob.glob(pattern)
        if not pickle_files:
            print(f"No pickle files found in {input_path}")
            return []
        return sorted(pickle_files)
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return []

def process_pickle_files(input_path, coord_lookup, num_workers=None):
    """Process all pickle files in the given path using multiprocessing."""
    pickle_files = find_pickle_files(input_path)
    
    if not pickle_files:
        return
    
    if coord_lookup is None:
        print("Error: Could not load coordinate data. Exiting.")
        return
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(pickle_files))
    
    total_cores = mp.cpu_count()
    print(f"Found {len(pickle_files)} pickle file(s) to process")
    print(f"System CPU cores available: {total_cores}")
    print(f"Using {num_workers} worker process(es)")
    print("=" * 60)
    
    # Prepare arguments for worker processes
    worker_args = [(pickle_path, coord_lookup) for pickle_path in pickle_files]
    
    # Overall statistics
    overall_stats = {
        'files_processed': 0,
        'files_failed': 0,
        'total_entries': 0,
        'total_enriched': 0,
        'total_not_found': 0,
        'total_parse_errors': 0
    }
    
    # Process files using multiprocessing
    if len(pickle_files) == 1 or num_workers == 1:
        # Single file or single worker - process sequentially
        print("Processing files sequentially...")
        for i, pickle_path in enumerate(tqdm(pickle_files, desc="Sequential", unit="file"), 1):
            result = process_single_pickle_file((pickle_path, coord_lookup))
            
            if result['success']:
                stats = result['stats']
                overall_stats['files_processed'] += 1
                overall_stats['total_entries'] += stats['total']
                overall_stats['total_enriched'] += stats['enriched']
                overall_stats['total_not_found'] += stats['not_found']
                overall_stats['total_parse_errors'] += stats['parse_error']
            else:
                print(f"Error processing {pickle_path}: {result['error']}")
                overall_stats['files_failed'] += 1
    else:
        # Multiple files with multiple workers - process in parallel
        print("Processing files in parallel...")
        
        with mp.Pool(processes=num_workers) as pool:
            results = pool.imap(process_single_pickle_file, worker_args)
            with tqdm(total=len(pickle_files), desc=f"Parallel x{num_workers}", unit="file") as pbar:
                for result in results:
                    pbar.update(1)
                    if result['success']:
                        stats = result['stats']
                        overall_stats['files_processed'] += 1
                        overall_stats['total_entries'] += stats['total']
                        overall_stats['total_enriched'] += stats['enriched']
                        overall_stats['total_not_found'] += stats['not_found']
                        overall_stats['total_parse_errors'] += stats['parse_error']
                    else:
                        overall_stats['files_failed'] += 1
    
    # Print overall summary
    print("\n" + "=" * 60)
    print("OVERALL PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Files processed successfully: {overall_stats['files_processed']}")
    print(f"Files failed: {overall_stats['files_failed']}")
    print(f"Total entries processed: {overall_stats['total_entries']}")
    print(f"Total entries enriched: {overall_stats['total_enriched']}")
    print(f"Total entries not found: {overall_stats['total_not_found']}")
    print(f"Total parse errors: {overall_stats['total_parse_errors']}")
    
    if overall_stats['total_entries'] > 0:
        success_rate = overall_stats['total_enriched'] / overall_stats['total_entries'] * 100
        print(f"Overall success rate: {success_rate:.1f}%")
    
    print(f"\nProcessing completed!")

def main():
    """Main function to enrich pickle files with coordinates."""
    parser = argparse.ArgumentParser(
        description="Enrich Pickle Files with CCF Coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all pickle files in a directory (auto-detect CPU cores)
  # Note: joined.csv and channels.csv should be in the same directory
  python enrich_pickle_with_coordinates.py /path/to/pickle/files/
  
  # Process with specific number of workers
  python enrich_pickle_with_coordinates.py /path/to/pickle/files/ --workers 4
  
  # Process a single pickle file
  python enrich_pickle_with_coordinates.py /path/to/file.pickle
  
  # Process files in Downloads folder (default)
  python enrich_pickle_with_coordinates.py
  
  # Disable parallel processing (sequential)
  python enrich_pickle_with_coordinates.py /path/to/pickle/files/ --no-parallel
        """
    )
    
    parser.add_argument(
        'input_path',
        nargs='?',
        default=os.path.expanduser("~/Downloads"),
        help='Path to pickle file or directory containing pickle files. joined.csv and channels.csv should be in the same directory (default: ~/Downloads)'
    )
    
    parser.add_argument(
        '--workers',
        '-w',
        type=int,
        default=None,
        help=f'Number of worker processes to use (default: auto-detect, max: {mp.cpu_count()})'
    )
    
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing and use single worker'
    )
    
    args = parser.parse_args()
    
    # Determine number of workers
    if args.no_parallel:
        num_workers = 1
    elif args.workers is not None:
        num_workers = min(max(1, args.workers), mp.cpu_count())
    else:
        num_workers = None  # Auto-detect
    
    print("Pickle File Coordinate Enrichment Tool")
    print("=" * 50)
    print(f"Input path: {args.input_path}")
    if num_workers == 1:
        print("Processing mode: Sequential")
    else:
        print(f"Processing mode: Parallel (up to {num_workers if num_workers else mp.cpu_count()} workers)")
    
    # Load coordinate data from the input path
    coord_lookup = load_coordinate_data(args.input_path)
    
    # Process pickle files
    process_pickle_files(args.input_path, coord_lookup, num_workers)

if __name__ == "__main__":
    # Required for multiprocessing on Windows
    mp.freeze_support()
    main()
