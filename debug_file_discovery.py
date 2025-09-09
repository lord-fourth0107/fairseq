#!/usr/bin/env python3
"""
Debug script to check what files are available in the data directory
"""

import os
import sys

def debug_file_discovery(data_path):
    """Debug what files are found in the data directory"""
    print(f"🔍 Debugging file discovery in: {data_path}")
    print("=" * 60)
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Path {data_path} does not exist")
        return
    
    if not os.path.isdir(data_path):
        print(f"❌ Error: Path {data_path} is not a directory")
        return
    
    # Find all pickle files
    all_pickle_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.pickle'):
                all_pickle_files.append(os.path.join(root, file))
    
    print(f"✅ Found {len(all_pickle_files)} pickle files")
    
    if len(all_pickle_files) > 0:
        print("\n📁 All pickle files found:")
        for i, file in enumerate(all_pickle_files):
            basename = os.path.basename(file)
            print(f"  {i+1:2d}. {basename}")
        
        print("\n🔍 Checking for session patterns:")
        allen_sessions = ['719161530', '768515987', '771160300', '798911424', '771990200']
        
        for session in allen_sessions:
            matching_files = [f for f in all_pickle_files if os.path.basename(f).startswith(session)]
            print(f"  Session {session}: {len(matching_files)} files")
            for file in matching_files:
                print(f"    - {os.path.basename(file)}")
    else:
        print("❌ No pickle files found in the directory")
        print("\n📂 Directory contents:")
        try:
            for item in os.listdir(data_path):
                item_path = os.path.join(data_path, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                else:
                    print(f"  📄 {item}")
        except Exception as e:
            print(f"  Error listing directory: {e}")

def main():
    # Default data path from the script
    default_path = "/scratch/mkp6112/LFP/region_decoding/script/Allen_w2v2/Allen"
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = default_path
    
    debug_file_discovery(data_path)

if __name__ == "__main__":
    main()
