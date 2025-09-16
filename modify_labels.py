#!/usr/bin/env python3
"""
Modify Pickle Labels Script
Appends 'x_y_z' to each string label in the pickle file and saves the modified data.
"""

import pickle
import os
import shutil
from datetime import datetime

def modify_pickle_labels(input_path, output_path=None, backup=True):
    """
    Modify string labels in pickle file by appending 'x_y_z' to each label.
    
    Args:
        input_path: Path to input pickle file
        output_path: Path for output pickle file (default: same as input with _modified suffix)
        backup: Whether to create a backup of the original file
    """
    print(f"Loading pickle file: {input_path}")
    
    # Create backup if requested
    if backup:
        backup_path = input_path.replace('.pickle', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pickle')
        print(f"Creating backup: {backup_path}")
        shutil.copy2(input_path, backup_path)
    
    # Load the original data
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Original data type: {type(data)}")
    print(f"Number of items: {len(data)}")
    
    # Modify the labels
    modified_data = []
    modified_count = 0
    
    for i, item in enumerate(data):
        if isinstance(item, tuple) and len(item) >= 2:
            # Extract the original label and data
            original_data = item[0]
            original_label = item[1]
            
            # Append 'x_y_z' to the label
            modified_label = f"{original_label}_x_y_z"
            
            # Create new tuple with modified label
            modified_item = (original_data, modified_label)
            modified_data.append(modified_item)
            modified_count += 1
            
            # Print first few examples
            if i < 5:
                print(f"  Original: {original_label}")
                print(f"  Modified: {modified_label}")
        else:
            # Keep non-tuple items as-is
            modified_data.append(item)
    
    print(f"Modified {modified_count} labels")
    
    # Determine output path
    if output_path is None:
        base_name = input_path.replace('.pickle', '')
        output_path = f"{base_name}_modified.pickle"
    
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
    # Path to the pickle file in Downloads
    downloads_path = os.path.expanduser("~/Downloads")
    input_file = "715093703_810755797.pickle"
    input_path = os.path.join(downloads_path, input_file)
    
    # Check if file exists
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        print("Please make sure the file exists in your Downloads folder.")
        return
    
    # Create output path
    output_path = os.path.join(downloads_path, "715093703_810755797_modified.pickle")
    
    try:
        result_path = modify_pickle_labels(input_path, output_path, backup=True)
        print(f"\nSuccess! Modified pickle file saved to: {result_path}")
        print(f"Original file backed up with timestamp.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check the file and try again.")

if __name__ == "__main__":
    main()

