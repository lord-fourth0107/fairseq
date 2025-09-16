#!/usr/bin/env python3
"""
Add CCF Coordinates to Pickle Labels Script
Reads pickle file, extracts probe and channel IDs from labels, looks them up in joined.csv,
and adds CCF coordinates to the label string.
"""

import pickle
import pandas as pd
import os
import shutil
from datetime import datetime

def load_coordinates_lookup(csv_path):
    """
    Load the joined.csv file and create a lookup dictionary for probe_id and channel_id.
    
    Args:
        csv_path: Path to the joined.csv file
        
    Returns:
        Dictionary with (probe_id, channel_id) as key and coordinates as value
    """
    print(f"Loading coordinates from: {csv_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    print(f"CSV columns: {list(df.columns)}")
    print(f"Number of rows: {len(df)}")
    
    # Create lookup dictionary
    lookup = {}
    
    # Check for required columns
    required_cols = ['probe_id', 'anterior_posterior_ccf_coordinate', 
                    'dorsal_ventral_ccf_coordinate', 'left_right_ccf_coordinate']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return {}
    
    # Build lookup dictionary using (session_id, probe_id) as key, storing list of coordinates
    for _, row in df.iterrows():
        session_id = str(row['session_id'])
        probe_id = str(row['probe_id'])
        ap_coord = row['anterior_posterior_ccf_coordinate']
        dv_coord = row['dorsal_ventral_ccf_coordinate']
        lr_coord = row['left_right_ccf_coordinate']
        
        # Create coordinate string
        coords = f"{ap_coord}_{dv_coord}_{lr_coord}"
        key = (session_id, probe_id)
        
        if key not in lookup:
            lookup[key] = []
        lookup[key].append(coords)
    
    print(f"Created lookup for {len(lookup)} session-probe combinations")
    
    # Show some examples
    print("Sample lookup entries:")
    for i, (key, value) in enumerate(list(lookup.items())[:5]):
        print(f"  {key} -> {len(value)} coordinates: {value[:3]}...")
    
    return lookup

def modify_pickle_with_coordinates(input_path, csv_path, output_path=None, backup=True):
    """
    Modify pickle file labels by adding CCF coordinates from joined.csv.
    
    Args:
        input_path: Path to input pickle file
        csv_path: Path to joined.csv file
        output_path: Path for output pickle file
        backup: Whether to create a backup of the original file
    """
    print(f"Loading pickle file: {input_path}")
    
    # Create backup if requested
    if backup:
        backup_path = input_path.replace('.pickle', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pickle')
        print(f"Creating backup: {backup_path}")
        shutil.copy2(input_path, backup_path)
    
    # Load coordinates lookup
    coord_lookup = load_coordinates_lookup(csv_path)
    if not coord_lookup:
        print("Error: Could not load coordinates lookup. Exiting.")
        return None
    
    # Load the original data
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Original data type: {type(data)}")
    print(f"Number of items: {len(data)}")
    
    # Modify the labels
    modified_data = []
    modified_count = 0
    not_found_count = 0
    
    for i, item in enumerate(data):
        if isinstance(item, tuple) and len(item) >= 2:
            # Extract the original label and data
            original_data = item[0]
            original_label = item[1]
            
            # Parse the label to extract session_id, probe_id, and channel_id
            # Expected format: session_count_probe_id_channel_id_brain_region
            parts = original_label.split('_')
            if len(parts) >= 4:
                session_id = parts[0]  # session_id is the 1st part
                probe_id = parts[2]    # probe_id is the 3rd part
                channel_id = parts[3]  # channel_id is the 4th part
                
                # Look up coordinates using (session_id, probe_id)
                lookup_key = (session_id, probe_id)
                if lookup_key in coord_lookup:
                    # Get available coordinates for this session-probe combination
                    available_coords = coord_lookup[lookup_key]
                    
                    # Use channel_id to select coordinates (cycle through available ones)
                    try:
                        channel_idx = int(channel_id)
                        coord_idx = channel_idx % len(available_coords)
                        coordinates = available_coords[coord_idx]
                        modified_label = f"{original_label}_{coordinates}"
                        modified_count += 1
                    except (ValueError, IndexError):
                        # If channel_id is not a valid integer or no coordinates available
                        modified_label = original_label
                        not_found_count += 1
                        if not_found_count <= 5:
                            print(f"  Invalid channel_id or no coordinates: channel_id={channel_id}")
                else:
                    # If not found, keep original label
                    modified_label = original_label
                    not_found_count += 1
                    if not_found_count <= 5:  # Show first 5 not found
                        print(f"  Not found: session_id={session_id}, probe_id={probe_id}")
            else:
                # If label format is unexpected, keep original
                modified_label = original_label
                not_found_count += 1
                if not_found_count <= 5:
                    print(f"  Unexpected label format: {original_label}")
            
            # Create new tuple with modified label
            modified_item = (original_data, modified_label)
            modified_data.append(modified_item)
            
            # Print first few examples
            if i < 5:
                print(f"  Original: {original_label}")
                print(f"  Modified: {modified_label}")
        else:
            # Keep non-tuple items as-is
            modified_data.append(item)
    
    print(f"Modified {modified_count} labels with coordinates")
    print(f"Could not find coordinates for {not_found_count} labels")
    
    # Determine output path
    if output_path is None:
        base_name = input_path.replace('.pickle', '')
        output_path = f"{base_name}_with_coordinates.pickle"
    
    # Save the modified data
    print(f"Saving modified data to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(modified_data, f)
    
    # Verify the saved file
    print("Verifying saved file...")
    with open(output_path, 'rb') as f:
        verify_data = pickle.load(f)
    
    print(f"Verification successful:")
    print(f"  Items in saved file: {len(verify_data)}")
    print(f"  First few labels:")
    for i in range(min(3, len(verify_data))):
        if isinstance(verify_data[i], tuple) and len(verify_data[i]) >= 2:
            print(f"    {i+1}: {verify_data[i][1]}")
    
    return output_path

def main():
    # Paths to files in Downloads
    downloads_path = os.path.expanduser("~/Downloads")
    pickle_file = "715093703_810755797_modified.pickle"  # Use the modified file
    csv_file = "joined.csv"
    
    input_path = os.path.join(downloads_path, pickle_file)
    csv_path = os.path.join(downloads_path, csv_file)
    
    # Check if files exist
    if not os.path.exists(input_path):
        print(f"Error: Pickle file not found: {input_path}")
        return
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    # Create output path
    output_path = os.path.join(downloads_path, "715093703_810755797_with_coordinates.pickle")
    
    try:
        result_path = modify_pickle_with_coordinates(input_path, csv_path, output_path, backup=True)
        if result_path:
            print(f"\nSuccess! Modified pickle file saved to: {result_path}")
            print(f"Original file backed up with timestamp.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check the files and try again.")

if __name__ == "__main__":
    main()
