#!/usr/bin/env python3
"""
Serial Pickle File Enrichment Script
Processes pickle files one at a time, reading from input directory and writing to output directory.
- Serial processing (one pickle at a time)
- Input/output directory separation
- No modification of input files
- Skips backup files
- Robust error handling and logging
"""

import pickle
import os
import sys
import pandas as pd
import argparse
import glob
from collections import defaultdict
from tqdm import tqdm
import time
import json
import logging
from datetime import datetime
import traceback
import gc

# Configure logging
def setup_logging(log_file=None, log_level=logging.INFO):
    """Setup logging configuration."""
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
        # Load CSV files
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
        
        # Create coordinate lookup using channels.csv coordinates (more accurate)
        coord_lookup = {}
        for _, row in merged_df.iterrows():
            key = (str(row['session_id']), str(row['probe_id']), str(row['id']))
            coord_lookup[key] = {
                'ap': row['anterior_posterior_ccf_coordinate_y'],  # Use channels.csv coordinates
                'dv': row['dorsal_ventral_ccf_coordinate_y'],      # Use channels.csv coordinates
                'lr': row['left_right_ccf_coordinate_y'],          # Use channels.csv coordinates
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
    """Heuristically determine if a label already has appended coordinates."""
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
    """Decide whether the pickle should be enriched."""
    logger = logging.getLogger(__name__)
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, (list, tuple)) or len(data) == 0:
            return True  # empty/malformed → attempt enrichment
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
            logger.info(f"{os.path.basename(pickle_path)}: Only {frac:.2f} of sampled labels enriched → will enrich")
            return True
        # If enriched, verify variability of CCF coords
        unique_triplets = set(coord_triplets)
        if len(unique_triplets) <= 1 and len(coord_triplets) > 0:
            logger.info(f"{os.path.basename(pickle_path)}: All sampled CCF coordinates identical → will re-enrich")
            return True
        # Looks properly enriched with variation → skip
        logger.info(f"{os.path.basename(pickle_path)}: Appears properly enriched → will skip")
        return False
    except Exception as e:
        logger.warning(f"{os.path.basename(pickle_path)}: Pre-check failed ({e}) → will enrich by default")
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

def enrich_pickle_file(input_path, output_path, coord_lookup, batch_size=10000):
    """Enrich a pickle file and save to output directory."""
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    logger.info(f"Processing: {os.path.basename(input_path)}")
    
    try:
        # Load pickle file
        logger.info("Loading pickle file...")
        try:
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
        except EOFError as e:
            logger.error("EOFError while reading pickle (likely empty/corrupted)")
            raise
        
        logger.info(f"Loaded {len(data)} entries")
        
        # Process data
        logger.info("Processing entries...")
        enriched_data, stats = process_entry_batch(data, coord_lookup)
        
        # Save enriched data to output directory
        logger.info("Saving enriched data...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(enriched_data, f)
        
        # Post-enrichment verification: ensure channels do not all share the same CCF
        try:
            with open(output_path, 'rb') as f:
                verify_data = pickle.load(f)
            channel_to_coord = {}
            for entry in verify_data[:min(10000, len(verify_data))]:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    label = entry[1]
                    parts = str(label).split('_')
                    if len(parts) >= 9:
                        chan = parts[3]
                        try:
                            ap, dv, lr = float(parts[-5]), float(parts[-4]), float(parts[-3])
                            if chan not in channel_to_coord:
                                channel_to_coord[chan] = (ap, dv, lr)
                        except Exception:
                            continue
            unique_coords = len(set(channel_to_coord.values())) if channel_to_coord else 0
            stats['channels_seen'] = len(channel_to_coord)
            stats['unique_channel_coords'] = unique_coords
            stats['ccf_identical'] = stats['channels_seen'] > 1 and unique_coords <= 1
            if stats['ccf_identical']:
                logger.warning("Post-check: all channels share identical CCF coordinates")
        except Exception:
            # Do not fail the run if verification has issues
            pass

        logger.info(f"Enrichment completed in {time.time() - start_time:.2f}s")
        logger.info(f"Saved enriched data to: {output_path}")

        return stats
        
    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        logger.error(traceback.format_exc())
        return None

def find_pickle_files(input_path):
    """Find all pickle files in the given path, excluding backup files."""
    logger = logging.getLogger(__name__)
    
    if os.path.isfile(input_path):
        if input_path.endswith('.pickle') and not input_path.endswith('.backup'):
            return [input_path]
        else:
            logger.error(f"{input_path} is not a pickle file")
            return []
    elif os.path.isdir(input_path):
        pattern = os.path.join(input_path, "*.pickle")
        pickle_files = glob.glob(pattern)
        # Filter out backup files
        pickle_files = [f for f in pickle_files if not f.endswith('.backup')]
        if not pickle_files:
            logger.warning(f"No pickle files found in {input_path}")
            return []
        return sorted(pickle_files)
    else:
        logger.error(f"{input_path} is not a valid file or directory")
        return []

def process_pickle_files(input_dir, output_dir, coord_lookup, batch_size=10000):
    """Process all pickle files serially with comprehensive logging and error handling."""
    logger = logging.getLogger(__name__)
    
    pickle_files = find_pickle_files(input_dir)
    
    if not pickle_files:
        logger.error("No pickle files found to process")
        return
    
    if coord_lookup is None:
        logger.error("Could not load coordinate data. Exiting.")
        return
    
    logger.info(f"Found {len(pickle_files)} pickle file(s) to process")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Batch size for large files: {batch_size}")
    logger.info("=" * 60)
    
    # Overall statistics
    overall_stats = {
        'files_processed': 0,
        'files_failed': 0,
        'files_skipped': 0,
        'bad_files': [],
        'total_entries': 0,
        'total_enriched': 0,
        'total_not_found': 0,
        'total_parse_errors': 0,
        'total_invalid_entries': 0,
        'start_time': time.time()
    }
    
    # Process files serially
    logger.info("Processing files serially...")
    for pickle_path in tqdm(pickle_files, desc="Processing", unit="file"):
        # Create output path
        filename = os.path.basename(pickle_path)
        output_path = os.path.join(output_dir, filename)
        
        # Pre-check: decide whether to enrich
        if not should_enrich_pickle(pickle_path):
            overall_stats['files_skipped'] += 1
            # Copy original file to output directory if skipping
            os.makedirs(output_dir, exist_ok=True)
            import shutil
            shutil.copy2(pickle_path, output_path)
            logger.info(f"Skipped enrichment, copied original to: {output_path}")
            continue
        
        # Process the file
        try:
            stats = enrich_pickle_file(pickle_path, output_path, coord_lookup, batch_size)
            if stats is None:
                logger.error(f"Error processing {pickle_path}: enrichment returned None")
                overall_stats['files_failed'] += 1
                continue
            
            overall_stats['files_processed'] += 1
            overall_stats['total_entries'] += stats['total']
            overall_stats['total_enriched'] += stats['enriched']
            overall_stats['total_not_found'] += stats['not_found']
            overall_stats['total_parse_errors'] += stats['parse_error']
            overall_stats['total_invalid_entries'] += stats['invalid_entry']
            
            # Track files whose channels remained identical after enrichment
            if stats.get('ccf_identical'):
                overall_stats.setdefault('postcheck_issues', []).append(pickle_path)
                
        except EOFError as e:
            logger.error(f"Error processing {pickle_path}: Ran out of input")
            overall_stats['files_failed'] += 1
            overall_stats['bad_files'].append({'file': pickle_path, 'reason': 'EOFError: Ran out of input'})
        except Exception as e:
            logger.error(f"Error processing {pickle_path}: {e}")
            overall_stats['files_failed'] += 1
    
    # Calculate total time
    overall_stats['total_time'] = time.time() - overall_stats['start_time']
    
    # Log summary
    logger.info("Processing complete. Writing summary JSON...")
    
    # Save summary to JSON
    summary_file = f"enrichment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(overall_stats, f, indent=2)
    logger.info(f"Summary saved to: {summary_file}")

    # Save bad files (EOF/corrupt) to a separate JSON for quick triage
    if overall_stats['bad_files']:
        bad_file_path = f"bad_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(bad_file_path, 'w') as bf:
            json.dump(overall_stats['bad_files'], bf, indent=2)
        logger.info(f"Bad files list saved to: {bad_file_path}")

    # Report any files that still have identical channel CCFs after enrichment
    if overall_stats.get('postcheck_issues'):
        logger.warning("Some files appear to have identical CCF across channels after enrichment")
        logger.warning("Files: " + ", ".join(overall_stats['postcheck_issues']))
    
    logger.info("Processing completed successfully!")

def main():
    """Main function with enhanced argument parsing and error handling."""
    parser = argparse.ArgumentParser(
        description="Serial Pickle File Enrichment with CCF Coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # Basic usage
  python enrich_pickle_serial.py /path/to/input/ /path/to/output/
  
  # With custom batch size
  python enrich_pickle_serial.py /path/to/input/ /path/to/output/ --batch-size 5000
  
  # With memory optimization
  python enrich_pickle_serial.py /path/to/input/ /path/to/output/ --max-memory 16
  
  # Custom logging
  python enrich_pickle_serial.py /path/to/input/ /path/to/output/ --log-file custom.log --log-level DEBUG
        """
    )
    
    parser.add_argument(
        'input_dir',
        help='Path to input directory containing pickle files'
    )
    
    parser.add_argument(
        'output_dir',
        help='Path to output directory for enriched pickle files'
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
    logger.info("Serial Pickle File Enrichment Tool")
    logger.info("=" * 50)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Log level: {args.log_level}")
    logger.info(f"Processing mode: Serial (one file at a time)")
    
    try:
        # Load coordinate data
        logger.info("Loading coordinate data...")
        coord_lookup = load_coordinate_data(args.input_dir, args.max_memory)
        
        if coord_lookup is None:
            logger.error("Failed to load coordinate data. Exiting.")
            sys.exit(1)
        
        # Process pickle files
        logger.info("Starting pickle file processing...")
        process_pickle_files(args.input_dir, args.output_dir, coord_lookup, args.batch_size)
        
        logger.info("Enrichment process completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
