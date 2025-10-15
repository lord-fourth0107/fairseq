#!/usr/bin/env python3
"""
Serial Pickle Enrichment Validation Script
Checks if pickle files are correctly enriched by verifying that all channels have distinct CCF coordinates.
Based on chat history: enrichment is successful if all channels have distinct (ap, dv, lr) coordinates.
"""

import pickle
import os
import sys
import argparse
import glob
from collections import defaultdict
from tqdm import tqdm
import logging
from datetime import datetime
import json

def setup_logging():
    """Setup simple logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def _looks_like_number(s: str) -> bool:
    """Check if a string looks like a number."""
    try:
        float(s)
        return True
    except Exception:
        return False

def is_label_enriched(label: str) -> bool:
    """Check if a label has the enriched format with CCF coordinates.
    Expected format: session_id_count_probe_id_channel_id_brain_region_ap_dv_lr_probe_h_probe_v
    """
    if not isinstance(label, str):
        return False
    parts = label.split('_')
    if len(parts) < 9:  # Need at least 9 parts for enrichment
        return False
    # Last 5 parts should be numeric (ap, dv, lr, probe_h, probe_v)
    tail = parts[-5:]
    return all(_looks_like_number(x) for x in tail)

def extract_ccf_coordinates(label: str):
    """Extract CCF coordinates (ap, dv, lr) from an enriched label."""
    if not is_label_enriched(label):
        return None
    parts = label.split('_')
    try:
        ap, dv, lr = map(float, parts[-5:-2])  # Last 5 parts: ap, dv, lr, probe_h, probe_v
        return (ap, dv, lr)
    except Exception:
        return None

def extract_channel_id(label: str):
    """Extract channel ID from label."""
    if not isinstance(label, str) or '_' not in label:
        return None
    parts = label.split('_')
    if len(parts) < 4:
        return None
    return parts[3]  # channel_id is the 4th part

def validate_pickle_file_serial(pickle_path: str):
    """Validate a single pickle file for correct enrichment - serial processing."""
    logger = logging.getLogger(__name__)
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, (list, tuple)) or len(data) == 0:
            return {
                'file': os.path.basename(pickle_path),
                'status': 'ERROR',
                'message': 'Empty or malformed data',
                'total_entries': 0,
                'enriched_entries': 0,
                'unique_channels': 0,
                'unique_ccf_coords': 0,
                'duplicate_coords': [],
                'issues': ['Empty or malformed data']
            }
        
        total_entries = len(data)
        enriched_entries = 0
        channel_to_coords = {}
        coord_to_channels = defaultdict(list)
        issues = []
        
        # Process each entry serially
        for i, entry in enumerate(data):
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                signal, label = entry[0], entry[1]
                
                if is_label_enriched(label):
                    enriched_entries += 1
                    
                    # Extract channel ID
                    channel_id = extract_channel_id(label)
                    if channel_id:
                        # Extract CCF coordinates
                        ccf_coords = extract_ccf_coordinates(label)
                        if ccf_coords:
                            channel_to_coords[channel_id] = ccf_coords
                            coord_to_channels[ccf_coords].append(channel_id)
                        else:
                            issues.append(f"Channel {channel_id}: Could not extract CCF coordinates")
                    else:
                        issues.append(f"Entry {i}: Could not extract channel ID from label")
                else:
                    issues.append(f"Entry {i}: Label not enriched - {label}")
        
        # Check for duplicate coordinates
        duplicate_coords = []
        for coords, channels in coord_to_channels.items():
            if len(channels) > 1:
                duplicate_coords.append({
                    'coordinates': coords,
                    'channels': channels,
                    'count': len(channels)
                })
        
        # Determine overall status based on chat history criteria
        enrichment_ratio = enriched_entries / total_entries if total_entries > 0 else 0
        
        if enrichment_ratio < 0.9:
            status = 'INCOMPLETE'
            issues.append(f"Only {enrichment_ratio:.2%} of entries are enriched")
        elif duplicate_coords:
            status = 'FAILED'
            issues.append(f"Found {len(duplicate_coords)} sets of duplicate CCF coordinates")
        else:
            status = 'SUCCESS'
        
        return {
            'file': os.path.basename(pickle_path),
            'status': status,
            'message': f"{status}: {enriched_entries}/{total_entries} enriched, {len(channel_to_coords)} channels, {len(set(channel_to_coords.values()))} unique CCF coords",
            'total_entries': total_entries,
            'enriched_entries': enriched_entries,
            'enrichment_ratio': enrichment_ratio,
            'unique_channels': len(channel_to_coords),
            'unique_ccf_coords': len(set(channel_to_coords.values())),
            'duplicate_coords': duplicate_coords,
            'issues': issues
        }
        
    except Exception as e:
        return {
            'file': os.path.basename(pickle_path),
            'status': 'ERROR',
            'message': f'Error reading file: {str(e)}',
            'total_entries': 0,
            'enriched_entries': 0,
            'unique_channels': 0,
            'unique_ccf_coords': 0,
            'duplicate_coords': [],
            'issues': [f'File read error: {str(e)}']
        }

def find_pickle_files(input_path):
    """Find all pickle files in the given path."""
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

def validate_pickle_files_serial(input_path):
    """Validate all pickle files serially."""
    logger = logging.getLogger(__name__)
    
    pickle_files = find_pickle_files(input_path)
    
    if not pickle_files:
        logger.error("No pickle files found to validate")
        return
    
    logger.info(f"Found {len(pickle_files)} pickle file(s) to validate")
    logger.info("Processing files serially (one at a time)...")
    logger.info("=" * 80)
    
    # Overall statistics
    overall_stats = {
        'total_files': len(pickle_files),
        'successful_files': 0,
        'failed_files': 0,
        'incomplete_files': 0,
        'error_files': 0,
        'total_entries': 0,
        'total_enriched_entries': 0,
        'total_unique_channels': 0,
        'total_unique_ccf_coords': 0,
        'files_with_duplicates': 0,
        'start_time': datetime.now().isoformat(),
        'results': []
    }
    
    # Process files serially
    for pickle_path in tqdm(pickle_files, desc="Validating", unit="file"):
        result = validate_pickle_file_serial(pickle_path)
        overall_stats['results'].append(result)
        
        # Update counters
        if result['status'] == 'SUCCESS':
            overall_stats['successful_files'] += 1
        elif result['status'] == 'FAILED':
            overall_stats['failed_files'] += 1
        elif result['status'] == 'INCOMPLETE':
            overall_stats['incomplete_files'] += 1
        else:
            overall_stats['error_files'] += 1
        
        overall_stats['total_entries'] += result['total_entries']
        overall_stats['total_enriched_entries'] += result['enriched_entries']
        overall_stats['total_unique_channels'] += result['unique_channels']
        overall_stats['total_unique_ccf_coords'] += result['unique_ccf_coords']
        
        if result['duplicate_coords']:
            overall_stats['files_with_duplicates'] += 1
        
        # Print result immediately (serial processing)
        status_symbol = {
            'SUCCESS': '✓',
            'FAILED': '✗',
            'INCOMPLETE': '⚠',
            'ERROR': '!'
        }.get(result['status'], '?')
        
        print(f"{status_symbol} {result['file']:<50} {result['status']:<12} {result['message']}")
        
        # Print issues if any
        if result['issues']:
            for issue in result['issues'][:3]:  # Show first 3 issues
                print(f"    - {issue}")
            if len(result['issues']) > 3:
                print(f"    - ... and {len(result['issues']) - 3} more issues")
    
    # Calculate final statistics
    overall_stats['end_time'] = datetime.now().isoformat()
    overall_stats['total_enrichment_ratio'] = (
        overall_stats['total_enriched_entries'] / overall_stats['total_entries'] 
        if overall_stats['total_entries'] > 0 else 0
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("SERIAL VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {overall_stats['total_files']}")
    print(f"Successful enrichment: {overall_stats['successful_files']}")
    print(f"Failed enrichment: {overall_stats['failed_files']}")
    print(f"Incomplete enrichment: {overall_stats['incomplete_files']}")
    print(f"Error files: {overall_stats['error_files']}")
    print(f"Files with duplicate CCF coordinates: {overall_stats['files_with_duplicates']}")
    print(f"Overall enrichment ratio: {overall_stats['total_enrichment_ratio']:.2%}")
    print(f"Total entries: {overall_stats['total_entries']:,}")
    print(f"Total enriched entries: {overall_stats['total_enriched_entries']:,}")
    print(f"Total unique channels: {overall_stats['total_unique_channels']:,}")
    print(f"Total unique CCF coordinates: {overall_stats['total_unique_ccf_coords']:,}")
    
    # Save detailed results to JSON
    results_file = f"enrichment_validation_serial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(overall_stats, f, indent=2)
    logger.info(f"Detailed results saved to: {results_file}")
    
    return overall_stats

def main():
    """Main function with simple argument parsing."""
    parser = argparse.ArgumentParser(
        description="Serial Pickle Enrichment Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # Validate all pickle files in a directory
  python validate_enrichment_serial.py /scratch/us2193/LFP/enriched_pickles/
  
  # Validate a single pickle file
  python validate_enrichment_serial.py /scratch/us2193/LFP/enriched_pickles/file.pickle
        """
    )
    
    parser.add_argument(
        'input_path',
        help='Path to pickle file or directory containing pickle files'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("Serial Pickle Enrichment Validation Tool")
    logger.info("=" * 50)
    logger.info(f"Input path: {args.input_path}")
    logger.info("Processing mode: Serial (one file at a time)")
    
    try:
        validate_pickle_files_serial(args.input_path)
        logger.info("Serial validation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
