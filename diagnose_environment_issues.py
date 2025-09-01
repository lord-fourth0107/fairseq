#!/usr/bin/env python3
"""
Diagnose specific environment issues that could cause SLURM job termination
"""

import os
import sys
import torch
import argparse

def diagnose_environment_issues():
    """Diagnose potential environment issues"""
    
    print("🔍 ENVIRONMENT ISSUE DIAGNOSIS")
    print("=" * 60)
    
    issues_found = []
    
    # 1. Check command line arguments
    print("1️⃣ COMMAND LINE ARGUMENTS:")
    try:
        # Simulate the argument parsing from your script
        parser = argparse.ArgumentParser(description='wav2vec2_2d single GPU')
        parser.add_argument('--data', type=str, default='Allen')
        parser.add_argument('--trial_length', type=int, default=60)
        parser.add_argument('--data_type', type=str, default='spectrogram')
        parser.add_argument('--sampling_rate', type=str, default='1250')
        parser.add_argument('--load_data', type=lambda x: x.lower() == 'true', default=True)
        parser.add_argument('--rand_init', type=lambda x: x.lower() == 'true', default=False)
        parser.add_argument('--ssl', type=lambda x: x.lower() == 'true', default=True)
        parser.add_argument('--session', type=str, default=None)
        parser.add_argument('--input_height', type=int, default=128)
        parser.add_argument('--input_width', type=int, default=128)
        parser.add_argument('--use_spatial_embedding', type=lambda x: x.lower() == 'true', default=True)
        parser.add_argument('--gpu_id', type=int, default=0)
        
        # Parse with empty args (simulating no arguments passed)
        args = parser.parse_args([])
        print("   ✅ Argument parsing works with defaults")
        
    except Exception as e:
        print(f"   ❌ Argument parsing failed: {e}")
        issues_found.append("Argument parsing issue")
    
    # 2. Check GPU ID issue
    print("\n2️⃣ GPU ID CONFIGURATION:")
    gpu_id = getattr(args, 'gpu_id', 0)
    print(f"   GPU ID from args: {gpu_id}")
    
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        print(f"   Available GPUs: {available_gpus}")
        
        if gpu_id >= available_gpus:
            print(f"   ❌ GPU ID {gpu_id} >= available GPUs {available_gpus}")
            issues_found.append(f"Invalid GPU ID: {gpu_id} >= {available_gpus}")
        else:
            print(f"   ✅ GPU ID {gpu_id} is valid")
            
        # Test GPU access
        try:
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(device)
            test_tensor = torch.randn(10, 10).to(device)
            print(f"   ✅ GPU {gpu_id} is accessible")
        except Exception as e:
            print(f"   ❌ GPU {gpu_id} access failed: {e}")
            issues_found.append(f"GPU {gpu_id} access failed")
    else:
        print("   ❌ CUDA not available")
        issues_found.append("CUDA not available")
    
    # 3. Check data path issues
    print("\n3️⃣ DATA PATH CONFIGURATION:")
    
    # Check the hardcoded path from your script
    data_loading_path = "/scratch/mkp6112/LFP/region_decoding/script/Allen_w2v2/Allen"
    print(f"   Your data path: {data_loading_path}")
    
    if os.path.exists(data_loading_path):
        print("   ✅ Your data path exists")
        # Count pickle files
        pickle_files = [f for f in os.listdir(data_loading_path) if f.endswith('.pickle')]
        print(f"   Found {len(pickle_files)} pickle files")
    else:
        print("   ❌ Your data path does not exist")
        issues_found.append("Your data path does not exist")
    
    # Check if there are any pickle files in subdirectories
    if os.path.exists(data_loading_path):
        total_pickles = 0
        for root, dirs, files in os.walk(data_loading_path):
            for file in files:
                if file.endswith('.pickle'):
                    total_pickles += 1
        print(f"   Total pickle files (including subdirs): {total_pickles}")
    
    # 4. Check output path permissions
    print("\n4️⃣ OUTPUT PATH PERMISSIONS:")
    output_path = "/vast/us2193/ssl_output/Allen/spectrogram/wav2vec2_2d/across_session"
    print(f"   Output path: {output_path}")
    
    try:
        os.makedirs(output_path, exist_ok=True)
        print("   ✅ Output path is writable")
    except Exception as e:
        print(f"   ❌ Output path creation failed: {e}")
        issues_found.append("Output path permission issue")
    
    # 5. Check import issues
    print("\n5️⃣ IMPORT DEPENDENCIES:")
    
    critical_imports = [
        'torch',
        'numpy',
        'pickle',
        'fairseq.models.wav2vec.wav2vec2_2d',
        'modified_session_dataset',
        'blind_localization.data.lazyloader_dataset'
    ]
    
    for module in critical_imports:
        try:
            if module == 'fairseq.models.wav2vec.wav2vec2_2d':
                from fairseq.models.wav2vec.wav2vec2_2d import Wav2Vec2_2DConfig, Wav2Vec2_2DModel
            elif module == 'modified_session_dataset':
                from modified_session_dataset import ModifiedSessionDataset
            elif module == 'blind_localization.data.lazyloader_dataset':
                from blind_localization.data.lazyloader_dataset import SessionDataset
            else:
                __import__(module)
            print(f"   ✅ {module}")
        except Exception as e:
            print(f"   ❌ {module}: {e}")
            issues_found.append(f"Import failed: {module}")
    
    # 6. Check virtual environment
    print("\n6️⃣ VIRTUAL ENVIRONMENT:")
    venv_path = "/vast/us2193/wav2vec2_env_py39"
    print(f"   Virtual environment path: {venv_path}")
    
    if os.path.exists(venv_path):
        print("   ✅ Virtual environment directory exists")
        
        # Check if we're in the virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("   ✅ Currently in a virtual environment")
        else:
            print("   ⚠️ Not currently in a virtual environment")
            issues_found.append("Not in virtual environment")
    else:
        print("   ❌ Virtual environment directory does not exist")
        issues_found.append("Virtual environment not found")
    
    # 7. Check file permissions
    print("\n7️⃣ FILE PERMISSIONS:")
    script_path = "/vast/us2193/fairseq/wav2vec2_2d_single_gpu.py"
    print(f"   Script path: {script_path}")
    
    if os.path.exists(script_path):
        if os.access(script_path, os.R_OK):
            print("   ✅ Script is readable")
        else:
            print("   ❌ Script is not readable")
            issues_found.append("Script not readable")
        
        if os.access(script_path, os.X_OK):
            print("   ✅ Script is executable")
        else:
            print("   ❌ Script is not executable")
            issues_found.append("Script not executable")
    else:
        print("   ❌ Script does not exist")
        issues_found.append("Script not found")
    
    # 8. Check SLURM environment
    print("\n8️⃣ SLURM ENVIRONMENT:")
    slurm_vars = ['SLURM_JOB_ID', 'SLURMD_NODENAME', 'SLURM_CPUS_PER_TASK', 'SLURM_MEM_PER_NODE']
    
    for var in slurm_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DIAGNOSIS SUMMARY:")
    
    if issues_found:
        print(f"❌ Found {len(issues_found)} issues:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
        
        print("\n🔧 RECOMMENDED FIXES:")
        
        if "Invalid GPU ID" in str(issues_found):
            print("   1. Fix GPU ID: Use --gpu_id 0 or remove the argument")
        
        if "Hardcoded data path" in str(issues_found):
            print("   2. Fix data path: Update script to use /scratch/us2193/neural_probe_data")
        
        if "Import failed" in str(issues_found):
            print("   3. Fix imports: Ensure all dependencies are installed in virtual environment")
        
        if "Virtual environment" in str(issues_found):
            print("   4. Activate virtual environment: source /vast/us2193/wav2vec2_env_py39/bin/activate")
        
        if "Script not" in str(issues_found):
            print("   5. Fix script permissions: chmod +x wav2vec2_2d_single_gpu.py")
        
    else:
        print("✅ No critical issues found!")
        print("   The job termination might be due to:")
        print("   - Infinite loops in the training code")
        print("   - Memory leaks despite sufficient allocation")
        print("   - SLURM time limits or other cluster policies")
        print("   - Network issues or file system problems")
    
    return issues_found

if __name__ == "__main__":
    issues = diagnose_environment_issues()
    sys.exit(0 if not issues else 1)
