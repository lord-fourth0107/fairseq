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
    import time
    start_time = time.time()
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
    
    # Load CSV files with timing
    csv_start = time.time()
    joined_df = pd.read_csv(joined_path)
    print(f"Loaded joined.csv: {time.time() - csv_start:.2f}s")
    
    csv_start = time.time()
    channels_df = pd.read_csv(channels_path)
    print(f"Loaded channels.csv: {time.time() - csv_start:.2f}s")
    
    print(f"Loaded joined.csv: {joined_df.shape}")
    print(f"Loaded channels.csv: {channels_df.shape}")
    
    # Merge joined.csv and channels.csv on probe_id
    # joined.csv has: session_id, probe_id, coordinates
    # channels.csv has: id (channel_id), ecephys_probe_id, probe positions
    merged_df = pd.merge(
        joined_df,
        channels_df,
        left_on='probe_id',
        right_on='ecephys_probe_id',
        how='inner'
    )
    
    print(f"Merged data shape: {merged_df.shape}")
    
    # Create a lookup dictionary: (session_id, probe_id, channel_id) -> coordinates
    coord_lookup = {}
    for _, row in merged_df.iterrows():
        # Use channel_id from channels.csv
        key = (str(row['session_id']), str(row['probe_id']), str(row['id']))
        coord_lookup[key] = {
            'ap': row['anterior_posterior_ccf_coordinate_x'],  # From joined.csv
            'dv': row['dorsal_ventral_ccf_coordinate_x'],      # From joined.csv
            'lr': row['left_right_ccf_coordinate_x'],          # From joined.csv
            'probe_h': row['probe_horizontal_position'],       # From channels.csv
            'probe_v': row['probe_vertical_position']          # From channels.csv
        }
    
    print(f"Created coordinate lookup with {len(coord_lookup)} entries")
    print(f"Total CSV loading time: {time.time() - start_time:.2f}s")
    return coord_lookup

def process_entry_batch(entry_batch, coord_lookup):
    """Process a batch of entries for multiprocessing."""
    enriched_batch = []
    stats = {
        'total': 0,
        'enriched': 0,
        'not_found': 0,
        'parse_error': 0
    }
    
    for entry in entry_batch:
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
                        
                        # Clean up the label by removing everything after the brain region
                        # Find the brain region (usually the last meaningful part before coordinates)
                        # Look for common brain region patterns
                        brain_region = None
                        for i in range(4, len(parts)):
                            if parts[i] in ['APN', 'CA1', 'CA2', 'CA3', 'DG', 'SUB', 'VIS', 'AUD', 'SOM', 'GUST', 'OLF', 'TH', 'HY', 'MB', 'P', 'MY']:
                                brain_region = parts[i]
                                break
                        
                        if brain_region:
                            # Create clean base label: session_count_probe_channel_brain_region
                            clean_label = f"{session_id}_{count}_{probe_id}_{channel_id}_{brain_region}"
                        else:
                            # Fallback: use first 5 parts if no brain region found
                            clean_label = '_'.join(parts[:5])
                        
                        # Look up coordinates using channel_id from the label
                        # The channel_id should match the 'id' field in channels.csv
                        lookup_key = (session_id, probe_id, channel_id)
                        if lookup_key in coord_lookup:
                            coords = coord_lookup[lookup_key]
                            # Append fresh coordinates to the clean label
                            enriched_label = f"{clean_label}_{coords['ap']}_{coords['dv']}_{coords['lr']}_{coords['probe_h']}_{coords['probe_v']}"
                            enriched_batch.append((signal, enriched_label))
                            stats['enriched'] += 1
                        else:
                            # If not found, keep original entry
                            enriched_batch.append(entry)  # Keep original
                            stats['not_found'] += 1
                    except Exception as e:
                        enriched_batch.append(entry)  # Keep original
                        stats['parse_error'] += 1
                else:
                    enriched_batch.append(entry)  # Keep original
                    stats['parse_error'] += 1
            else:
                enriched_batch.append(entry)  # Keep original
                stats['parse_error'] += 1
        else:
            enriched_batch.append(entry)  # Keep original
    
    return enriched_batch, stats

def enrich_pickle_file(pickle_path, coord_lookup, batch_size=10000, num_workers=None):
    """Enrich a pickle file with coordinate information using multiprocessing for large files."""
    import time
    start_time = time.time()
    
    print(f"\nProcessing pickle file: {pickle_path}")
    
    # Create backup
    backup_start = time.time()
    backup_path = pickle_path + '.backup'
    if not os.path.exists(backup_path):
        shutil.copy2(pickle_path, backup_path)
        print(f"Created backup: {backup_path}")
    print(f"Backup time: {time.time() - backup_start:.2f}s")
    
    # Load the pickle file
    load_start = time.time()
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Load time: {time.time() - load_start:.2f}s")
    print(f"Loaded {len(data)} entries")
    
    # Determine if we should use multiprocessing
    use_multiprocessing = len(data) > batch_size and num_workers and num_workers > 1
    
    if use_multiprocessing:
        print(f"Using multiprocessing with {num_workers} workers, batch size {batch_size}")
        
        # Split data into batches
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        print(f"Split into {len(batches)} batches")
        
        # Process batches in parallel
        process_start = time.time()
        enriched_data = []
        overall_stats = {
            'total': 0,
            'enriched': 0,
            'not_found': 0,
            'parse_error': 0
        }
        
        # Create partial function for multiprocessing
        process_func = partial(process_entry_batch, coord_lookup=coord_lookup)
        
        with mp.Pool(processes=num_workers) as pool:
            results = pool.imap(process_func, batches)
            
            with tqdm(total=len(batches), desc="Processing batches", unit="batch") as pbar:
                for enriched_batch, stats in results:
                    enriched_data.extend(enriched_batch)
                    overall_stats['total'] += stats['total']
                    overall_stats['enriched'] += stats['enriched']
                    overall_stats['not_found'] += stats['not_found']
                    overall_stats['parse_error'] += stats['parse_error']
                    pbar.update(1)
        
        stats = overall_stats
    else:
        # Process sequentially for small files or single worker
        print("Processing sequentially")
        process_start = time.time()
        enriched_data, stats = process_entry_batch(data, coord_lookup)
    
    # Print statistics
    process_time = time.time() - process_start
    print(f"Processing time: {process_time:.2f}s")
    print(f"Enrichment statistics:")
    print(f"  Total entries: {stats['total']}")
    print(f"  Successfully enriched: {stats['enriched']}")
    print(f"  Not found in lookup: {stats['not_found']}")
    print(f"  Parse errors: {stats['parse_error']}")
    print(f"  Success rate: {stats['enriched']/stats['total']*100:.1f}%")
    
    # Save the enriched data
    save_start = time.time()
    with open(pickle_path, 'wb') as f:
        pickle.dump(enriched_data, f)
    print(f"Save time: {time.time() - save_start:.2f}s")
    print(f"Total time: {time.time() - start_time:.2f}s")
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
    pickle_path, coord_lookup, batch_size, num_workers = args
    
    try:
        # Process all files without checking if already enriched
        stats = enrich_pickle_file(pickle_path, coord_lookup, batch_size, num_workers)
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

def process_pickle_files(input_path, coord_lookup, num_workers=None, batch_size=10000):
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
    print(f"Batch size for large files: {batch_size}")
    print("=" * 60)
    
    # Prepare arguments for worker processes
    worker_args = [(pickle_path, coord_lookup, batch_size, num_workers) for pickle_path in pickle_files]
    
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
            result = process_single_pickle_file((pickle_path, coord_lookup, batch_size, num_workers))
            
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
  
  # Process with specific number of workers and batch size
  python enrich_pickle_with_coordinates.py /path/to/pickle/files/ --workers 8 --batch-size 5000
  
  # Process a single pickle file
  python enrich_pickle_with_coordinates.py /path/to/file.pickle
  
  # Process files in Downloads folder (default)
  python enrich_pickle_with_coordinates.py
  
  # Disable parallel processing (sequential)
  python enrich_pickle_with_coordinates.py /path/to/pickle/files/ --no-parallel
  
  # HPC-friendly: Use many workers with smaller batches
  python enrich_pickle_with_coordinates.py /path/to/pickle/files/ --workers 32 --batch-size 2000
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
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='Batch size for processing large pickle files (default: 10000)'
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
    process_pickle_files(args.input_path, coord_lookup, num_workers, args.batch_size)

if __name__ == "__main__":
    # Required for multiprocessing on Windows
    mp.freeze_support()
    main()
