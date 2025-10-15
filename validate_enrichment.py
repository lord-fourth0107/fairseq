#!/usr/bin/env python3
"""
Pickle Enrichment Validation Script
Checks if pickle files are correctly enriched by verifying that all channels have distinct CCF coordinates.
Enrichment is considered successful if:
1. All labels have the expected enriched format (with CCF coordinates)
2. All channels within each file have distinct CCF coordinates
3. No channels share identical (ap, dv, lr) coordinates
"""

import pickle
import os
import sys
import argparse
import glob
from collections import defaultdict, Counter
from tqdm import tqdm
import logging
from datetime import datetime
import json

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=log_level,
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
    """Check if a label has the enriched format with CCF coordinates."""
    if not isinstance(label, str):
        return False
    parts = label.split('_')
    if len(parts) < 9:
        return False
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

def extract_channel_info(label: str):
    """Extract channel information from label."""
    if not isinstance(label, str) or '_' not in label:
        return None
    parts = label.split('_')
    if len(parts) < 4:
        return None
    try:
        session_id = parts[0]
        count = parts[1]
        probe_id = parts[2]
        channel_id = parts[3]
        return {
            'session_id': session_id,
            'count': count,
            'probe_id': probe_id,
            'channel_id': channel_id
        }
    except Exception:
        return None

def validate_pickle_file(pickle_path: str, sample_size: int = 10000):
    """Validate a single pickle file for correct enrichment."""
    logger = logging.getLogger(__name__)
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, (list, tuple)) or len(data) == 0:
            return {
                'file': pickle_path,
                'status': 'ERROR',
                'message': 'Empty or malformed data',
                'total_entries': 0,
                'enriched_entries': 0,
                'unique_channels': 0,
                'unique_ccf_coords': 0,
                'duplicate_coords': [],
                'issues': ['Empty or malformed data']
            }
        
        # Sample data if it's too large
        if len(data) > sample_size:
            import random
            data = random.sample(data, sample_size)
            logger.info(f"Sampled {sample_size} entries from {len(data)} total entries")
        
        total_entries = len(data)
        enriched_entries = 0
        channel_to_coords = {}
        coord_to_channels = defaultdict(list)
        issues = []
        
        for entry in data:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                signal, label = entry[0], entry[1]
                
                if is_label_enriched(label):
                    enriched_entries += 1
                    
                    # Extract channel info
                    channel_info = extract_channel_info(label)
                    if channel_info:
                        channel_id = channel_info['channel_id']
                        
                        # Extract CCF coordinates
                        ccf_coords = extract_ccf_coordinates(label)
                        if ccf_coords:
                            channel_to_coords[channel_id] = ccf_coords
                            coord_to_channels[ccf_coords].append(channel_id)
                        else:
                            issues.append(f"Channel {channel_id}: Could not extract CCF coordinates")
                    else:
                        issues.append(f"Could not extract channel info from label: {label}")
                else:
                    issues.append(f"Label not enriched: {label}")
        
        # Check for duplicate coordinates
        duplicate_coords = []
        for coords, channels in coord_to_channels.items():
            if len(channels) > 1:
                duplicate_coords.append({
                    'coordinates': coords,
                    'channels': channels,
                    'count': len(channels)
                })
        
        # Determine overall status
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
            'file': pickle_path,
            'status': status,
            'message': f"Enrichment {status.lower()}: {enriched_entries}/{total_entries} enriched, {len(channel_to_coords)} unique channels, {len(set(channel_to_coords.values()))} unique CCF coordinates",
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
            'file': pickle_path,
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

def validate_pickle_files(input_path, sample_size=10000, detailed_output=False):
    """Validate all pickle files in the given path."""
    logger = logging.getLogger(__name__)
    
    pickle_files = find_pickle_files(input_path)
    
    if not pickle_files:
        logger.error("No pickle files found to validate")
        return
    
    logger.info(f"Found {len(pickle_files)} pickle file(s) to validate")
    logger.info(f"Sample size per file: {sample_size}")
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
    
    # Process files
    for pickle_path in tqdm(pickle_files, desc="Validating", unit="file"):
        result = validate_pickle_file(pickle_path, sample_size)
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
        
        # Print result
        filename = os.path.basename(pickle_path)
        status_symbol = {
            'SUCCESS': '✓',
            'FAILED': '✗',
            'INCOMPLETE': '⚠',
            'ERROR': '!'
        }.get(result['status'], '?')
        
        print(f"{status_symbol} {filename:<50} {result['status']:<12} {result['message']}")
        
        # Print detailed issues if requested
        if detailed_output and result['issues']:
            for issue in result['issues']:
                print(f"    - {issue}")
    
    # Calculate final statistics
    overall_stats['end_time'] = datetime.now().isoformat()
    overall_stats['total_enrichment_ratio'] = (
        overall_stats['total_enriched_entries'] / overall_stats['total_entries'] 
        if overall_stats['total_entries'] > 0 else 0
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
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
    results_file = f"enrichment_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(overall_stats, f, indent=2)
    logger.info(f"Detailed results saved to: {results_file}")
    
    return overall_stats

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Validate Pickle File Enrichment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # Validate all pickle files in a directory
  python validate_enrichment.py /path/to/pickle/files/
  
  # Validate a single pickle file
  python validate_enrichment.py /path/to/single/file.pickle
  
  # Validate with detailed output
  python validate_enrichment.py /path/to/pickle/files/ --detailed
  
  # Validate with custom sample size
  python validate_enrichment.py /path/to/pickle/files/ --sample-size 5000
        """
    )
    
    parser.add_argument(
        'input_path',
        help='Path to pickle file or directory containing pickle files'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=10000,
        help='Maximum number of entries to sample per file (default: 10000)'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed issues for each file'
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
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Pickle Enrichment Validation Tool")
    logger.info("=" * 50)
    logger.info(f"Input path: {args.input_path}")
    logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Detailed output: {args.detailed}")
    
    try:
        validate_pickle_files(args.input_path, args.sample_size, args.detailed)
        logger.info("Validation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
