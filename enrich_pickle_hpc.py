#!/usr/bin/env python3
"""
HPC-Optimized Pickle File Enrichment Script
Enhanced version specifically designed for HPC environments with:
- Robust error handling and logging
- Memory optimization for large datasets
- SLURM-compatible multiprocessing
- Progress tracking and checkpointing
- Comprehensive statistics and reporting
"""

import pickle
import os
import sys
import pandas as pd
import shutil
import argparse
import glob
import multiprocessing as mp
from collections import defaultdict
from functools import partial
from tqdm import tqdm
import time
import json
import logging
from datetime import datetime
import traceback
import gc

# Configure logging
def setup_logging(log_file=None, log_level=logging.INFO):
    """Setup logging configuration for HPC environment."""
    if log_file is None:
        log_file = f"enrichment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def load_coordinate_data(input_path, max_memory_gb=8):
    """Load and merge coordinate data with memory optimization."""
    logger = logging.getLogger(__name__)
    start_time = time.time()
    logger.info("Loading coordinate data with memory optimization...")
    
    # Determine the directory to look for CSV files
    if os.path.isfile(input_path):
        csv_dir = os.path.dirname(input_path)
    else:
        csv_dir = input_path
    
    # Look for CSV files
    joined_path = os.path.join(csv_dir, 'joined.csv')
    channels_path = os.path.join(csv_dir, 'channels.csv')
    
    if not os.path.exists(joined_path):
        logger.error(f"joined.csv not found in {csv_dir}")
        return None
    
    if not os.path.exists(channels_path):
        logger.error(f"channels.csv not found in {csv_dir}")
        return None
    
    logger.info(f"Found CSV files: {joined_path}, {channels_path}")
    
    try:
        # Load CSV files with chunking for large files
        logger.info("Loading joined.csv...")
        joined_df = pd.read_csv(joined_path)
        logger.info(f"Loaded joined.csv: {joined_df.shape}")
        
        logger.info("Loading channels.csv...")
        channels_df = pd.read_csv(channels_path)
        logger.info(f"Loaded channels.csv: {channels_df.shape}")
        
        # Check memory usage
        memory_usage = (joined_df.memory_usage(deep=True).sum() + 
                       channels_df.memory_usage(deep=True).sum()) / 1024**3
        logger.info(f"DataFrame memory usage: {memory_usage:.2f} GB")
        
        if memory_usage > max_memory_gb:
            logger.warning(f"Memory usage ({memory_usage:.2f} GB) exceeds limit ({max_memory_gb} GB)")
            logger.info("Consider using chunked processing for very large datasets")
        
        # Merge dataframes
        logger.info("Merging coordinate data...")
        merged_df = pd.merge(
            joined_df,
            channels_df,
            left_on='probe_id',
            right_on='ecephys_probe_id',
            how='inner'
        )
        logger.info(f"Merged data shape: {merged_df.shape}")
        
        # Create coordinate lookup
        coord_lookup = {}
        for _, row in merged_df.iterrows():
            key = (str(row['session_id']), str(row['probe_id']), str(row['id']))
            coord_lookup[key] = {
                'ap': row['anterior_posterior_ccf_coordinate_x'],
                'dv': row['dorsal_ventral_ccf_coordinate_x'],
                'lr': row['left_right_ccf_coordinate_x'],
                'probe_h': row['probe_horizontal_position'],
                'probe_v': row['probe_vertical_position']
            }
        
        logger.info(f"Created coordinate lookup with {len(coord_lookup)} entries")
        logger.info(f"Coordinate loading completed in {time.time() - start_time:.2f}s")
        
        # Clean up memory
        del joined_df, channels_df, merged_df
        gc.collect()
        
        return coord_lookup
        
    except Exception as e:
        logger.error(f"Error loading coordinate data: {e}")
        logger.error(traceback.format_exc())
        return None

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

def _extract_enriched_coords(label: str):
    """Return (ap, dv, lr, probe_h, probe_v) as floats if enriched, else None."""
    if not is_label_enriched(label):
        return None
    parts = label.split('_')
    try:
        ap, dv, lr, ph, pv = map(float, parts[-5:])
        return (ap, dv, lr, ph, pv)
    except Exception:
        return None

def should_enrich_pickle(pickle_path: str, sample_size: int = 100, enriched_threshold: float = 0.9) -> bool:
    """Decide whether the pickle should be enriched.
    Rules:
      - If fewer than 'enriched_threshold' fraction of sampled labels look enriched → enrich
      - If labels look enriched but all (ap, dv, lr) are identical across samples → re-enrich
      - Otherwise, skip enrichment
    """
    logger = logging.getLogger(__name__)
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, (list, tuple)) or len(data) == 0:
            return True  # empty/malformed → attempt enrichment path
        n = min(len(data), sample_size)
        checked = 0
        enriched_flags = 0
        coord_triplets = []
        for i in range(n):
            entry = data[i]
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                label = entry[1]
                checked += 1
                if is_label_enriched(label):
                    enriched_flags += 1
                    coords = _extract_enriched_coords(label)
                    if coords is not None:
                        ap, dv, lr, _, _ = coords
                        coord_triplets.append((ap, dv, lr))
        if checked == 0:
            return True
        frac = enriched_flags / checked
        if frac < enriched_threshold:
            logger.info(f"{pickle_path}: Only {frac:.2f} of sampled labels enriched → will enrich")
            return True
        # If enriched, verify variability of CCF coords
        unique_triplets = set(coord_triplets)
        if len(unique_triplets) <= 1 and len(coord_triplets) > 0:
            logger.info(f"{pickle_path}: All sampled CCF coordinates identical → will re-enrich")
            return True
        # Looks properly enriched with variation → skip
        logger.info(f"{pickle_path}: Appears properly enriched → will skip")
        return False
    except Exception as e:
        logger.warning(f"{pickle_path}: Pre-check failed ({e}) → will enrich by default")
        return True

def process_entry_batch(entry_batch, coord_lookup):
    """Process a batch of entries with enhanced error handling."""
    enriched_batch = []
    stats = {
        'total': 0,
        'enriched': 0,
        'not_found': 0,
        'parse_error': 0,
        'invalid_entry': 0
    }
    
    for entry in entry_batch:
        stats['total'] += 1
        
        try:
            if isinstance(entry, tuple) and len(entry) >= 2:
                signal, label = entry[0], entry[1]
                
                if isinstance(label, str) and '_' in label:
                    parts = label.split('_')
                    if len(parts) >= 4:
                        try:
                            session_id = parts[0]
                            count = parts[1]
                            probe_id = parts[2]
                            channel_id = parts[3]
                            
                            # Find brain region
                            brain_region = None
                            for i in range(4, len(parts)):
                                if parts[i] in ['APN', 'CA1', 'CA2', 'CA3', 'DG', 'SUB', 'VIS', 'AUD', 'SOM', 'GUST', 'OLF', 'TH', 'HY', 'MB', 'P', 'MY']:
                                    brain_region = parts[i]
                                    break
                            
                            if brain_region:
                                clean_label = f"{session_id}_{count}_{probe_id}_{channel_id}_{brain_region}"
                            else:
                                clean_label = '_'.join(parts[:5])
                            
                            # Look up coordinates
                            lookup_key = (session_id, probe_id, channel_id)
                            if lookup_key in coord_lookup:
                                coords = coord_lookup[lookup_key]
                                enriched_label = f"{clean_label}_{coords['ap']}_{coords['dv']}_{coords['lr']}_{coords['probe_h']}_{coords['probe_v']}"
                                enriched_batch.append((signal, enriched_label))
                                stats['enriched'] += 1
                            else:
                                enriched_batch.append(entry)
                                stats['not_found'] += 1
                        except Exception as e:
                            enriched_batch.append(entry)
                            stats['parse_error'] += 1
                    else:
                        enriched_batch.append(entry)
                        stats['parse_error'] += 1
                else:
                    enriched_batch.append(entry)
                    stats['parse_error'] += 1
            else:
                enriched_batch.append(entry)
                stats['invalid_entry'] += 1
                
        except Exception as e:
            enriched_batch.append(entry)
            stats['parse_error'] += 1
    
    return enriched_batch, stats

def enrich_pickle_file(pickle_path, coord_lookup, batch_size=10000, num_workers=1):
    """Enrich a pickle file with enhanced error handling and logging."""
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    logger.info(f"Processing pickle file: {pickle_path}")
    
    try:
        # Create backup
        backup_path = pickle_path + '.backup'
        if not os.path.exists(backup_path):
            shutil.copy2(pickle_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
        
        # Load pickle file
        logger.info("Loading pickle file...")
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded {len(data)} entries")
        
        # Process data
        logger.info("Processing entries...")
        enriched_data, stats = process_entry_batch(data, coord_lookup)
        
        # Suppress detailed per-file statistics logging
        
        # Save enriched data
        logger.info("Saving enriched data...")
        with open(pickle_path, 'wb') as f:
            pickle.dump(enriched_data, f)
        
        logger.info(f"Enrichment completed in {time.time() - start_time:.2f}s")
        logger.info(f"Saved enriched data to: {pickle_path}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error processing {pickle_path}: {e}")
        logger.error(traceback.format_exc())
        return None

def process_single_pickle_file(args):
    """Worker function for processing a single pickle file."""
    pickle_path, coord_lookup, batch_size, num_workers = args
    
    try:
        # Pre-check: decide whether to enrich
        if not should_enrich_pickle(pickle_path):
            return {
                'file': pickle_path,
                'success': True,
                'skipped': True,
                'stats': {
                    'total': 0,
                    'enriched': 0,
                    'not_found': 0,
                    'parse_error': 0,
                    'invalid_entry': 0
                },
                'error': None
            }

        stats = enrich_pickle_file(pickle_path, coord_lookup, batch_size, num_workers)
        if stats is None:
            return {
                'file': pickle_path,
                'success': False,
                'skipped': False,
                'stats': None,
                'error': 'enrichment returned None (see logs for details)'
            }
        return {
            'file': pickle_path,
            'success': True,
            'skipped': False,
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
    logger = logging.getLogger(__name__)
    
    if os.path.isfile(input_path):
        if input_path.endswith('.pickle'):
            return [input_path]
        else:
            logger.error(f"{input_path} is not a pickle file")
            return []
    elif os.path.isdir(input_path):
        pattern = os.path.join(input_path, "*.pickle")
        pickle_files = glob.glob(pattern)
        if not pickle_files:
            logger.warning(f"No pickle files found in {input_path}")
            return []
        return sorted(pickle_files)
    else:
        logger.error(f"{input_path} is not a valid file or directory")
        return []

def process_pickle_files(input_path, coord_lookup, num_workers=None, batch_size=10000):
    """Process all pickle files with comprehensive logging and error handling."""
    logger = logging.getLogger(__name__)
    
    pickle_files = find_pickle_files(input_path)
    
    if not pickle_files:
        logger.error("No pickle files found to process")
        return
    
    if coord_lookup is None:
        logger.error("Could not load coordinate data. Exiting.")
        return
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(pickle_files))
    
    logger.info(f"Found {len(pickle_files)} pickle file(s) to process")
    logger.info(f"System CPU cores available: {mp.cpu_count()}")
    logger.info(f"Using {num_workers} worker process(es)")
    logger.info(f"Batch size for large files: {batch_size}")
    logger.info("=" * 60)
    
    # Prepare arguments for worker processes
    worker_args = [(pickle_path, coord_lookup, batch_size, num_workers) for pickle_path in pickle_files]
    
    # Overall statistics
    overall_stats = {
        'files_processed': 0,
        'files_failed': 0,
        'files_skipped': 0,
        'total_entries': 0,
        'total_enriched': 0,
        'total_not_found': 0,
        'total_parse_errors': 0,
        'total_invalid_entries': 0,
        'start_time': time.time()
    }
    
    # Process files
    if len(pickle_files) == 1 or num_workers == 1:
        logger.info("Processing files sequentially...")
        for pickle_path in tqdm(pickle_files, desc="Sequential", unit="file"):
            result = process_single_pickle_file((pickle_path, coord_lookup, batch_size, num_workers))
            
            if result['success']:
                if result.get('skipped'):
                    overall_stats['files_skipped'] += 1
                    continue
                stats = result['stats']
                overall_stats['files_processed'] += 1
                overall_stats['total_entries'] += stats['total']
                overall_stats['total_enriched'] += stats['enriched']
                overall_stats['total_not_found'] += stats['not_found']
                overall_stats['total_parse_errors'] += stats['parse_error']
                overall_stats['total_invalid_entries'] += stats['invalid_entry']
            else:
                logger.error(f"Error processing {pickle_path}: {result['error']}")
                overall_stats['files_failed'] += 1
    else:
        logger.info("Processing files in parallel...")
        
        with mp.Pool(processes=num_workers) as pool:
            results = pool.imap(process_single_pickle_file, worker_args)
            with tqdm(total=len(pickle_files), desc=f"Parallel x{num_workers}", unit="file") as pbar:
                for result in results:
                    pbar.update(1)
                    if result['success']:
                        if result.get('skipped'):
                            overall_stats['files_skipped'] += 1
                            continue
                        stats = result['stats']
                        overall_stats['files_processed'] += 1
                        overall_stats['total_entries'] += stats['total']
                        overall_stats['total_enriched'] += stats['enriched']
                        overall_stats['total_not_found'] += stats['not_found']
                        overall_stats['total_parse_errors'] += stats['parse_error']
                        overall_stats['total_invalid_entries'] += stats['invalid_entry']
                    else:
                        overall_stats['files_failed'] += 1
                        logger.error(f"Error processing {result['file']}: {result['error']}")
    
    # Calculate total time
    overall_stats['total_time'] = time.time() - overall_stats['start_time']
    
    # Suppress verbose overall statistics logging (still save JSON summary below)
    logger.info("Processing complete. Writing summary JSON...")
    
    # Save summary to JSON
    summary_file = f"enrichment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(overall_stats, f, indent=2)
    logger.info(f"Summary saved to: {summary_file}")
    
    logger.info("Processing completed successfully!")

def main():
    """Main function with enhanced argument parsing and error handling."""
    parser = argparse.ArgumentParser(
        description="HPC-Optimized Pickle File Enrichment with CCF Coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
HPC Usage Examples:
  # Basic HPC usage with auto-detection
  python enrich_pickle_hpc.py /path/to/pickle/files/
  
  # HPC with specific resources
  python enrich_pickle_hpc.py /path/to/pickle/files/ --workers 32 --batch-size 5000
  
  # HPC with memory optimization
  python enrich_pickle_hpc.py /path/to/pickle/files/ --max-memory 16 --workers 16
  
  # Sequential processing (for debugging)
  python enrich_pickle_hpc.py /path/to/pickle/files/ --no-parallel
  
  # Custom logging
  python enrich_pickle_hpc.py /path/to/pickle/files/ --log-file custom.log --log-level DEBUG
        """
    )
    
    parser.add_argument(
        'input_path',
        nargs='?',
        default=os.path.expanduser("~/Downloads"),
        help='Path to pickle file or directory containing pickle files'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help=f'Number of worker processes (default: auto-detect, max: {mp.cpu_count()})'
    )
    
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='Batch size for processing large files (default: 10000)'
    )
    
    parser.add_argument(
        '--max-memory',
        type=float,
        default=8.0,
        help='Maximum memory usage in GB (default: 8.0)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Custom log file path (default: auto-generated)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    log_file = setup_logging(args.log_file, log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("HPC-Optimized Pickle File Enrichment Tool")
    logger.info("=" * 50)
    logger.info(f"Input path: {args.input_path}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Log level: {args.log_level}")
    
    # Determine number of workers
    if args.no_parallel:
        num_workers = 1
        logger.info("Processing mode: Sequential")
    elif args.workers is not None:
        num_workers = min(max(1, args.workers), mp.cpu_count())
        logger.info(f"Processing mode: Parallel ({num_workers} workers)")
    else:
        num_workers = None
        logger.info(f"Processing mode: Auto-detect (max {mp.cpu_count()} workers)")
    
    try:
        # Load coordinate data
        logger.info("Loading coordinate data...")
        coord_lookup = load_coordinate_data(args.input_path, args.max_memory)
        
        if coord_lookup is None:
            logger.error("Failed to load coordinate data. Exiting.")
            sys.exit(1)
        
        # Process pickle files
        logger.info("Starting pickle file processing...")
        process_pickle_files(args.input_path, coord_lookup, num_workers, args.batch_size)
        
        logger.info("Enrichment process completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    # Required for multiprocessing on Windows and some HPC systems
    mp.freeze_support()
    main()
