#!/usr/bin/env python3
"""
Comprehensive resource analysis for Wav2Vec2 2D training
"""

def analyze_resources():
    """Analyze all resource requirements"""
    
    print("🔍 COMPREHENSIVE RESOURCE ANALYSIS")
    print("=" * 70)
    
    # 1. Model Memory (from previous calculation)
    print("📊 MODEL MEMORY REQUIREMENTS:")
    model_memory_fp32 = 1.69  # GB (from previous calculation)
    model_memory_fp16 = 0.85  # GB
    
    print(f"   Model + Training (FP32): {model_memory_fp32:.2f} GB")
    print(f"   Model + Training (FP16): {model_memory_fp16:.2f} GB")
    
    # 2. Data Loading Memory
    print("\n📁 DATA LOADING MEMORY:")
    
    # Configuration from your script
    subset_data = 0.1  # 10% of data
    batch_size = 16
    num_workers = 0  # Single process
    input_shape = (1, 3750, 93)  # (C, H, W)
    
    # Memory per sample
    bytes_per_sample = 1 * 3750 * 93 * 4  # 4 bytes per float32
    memory_per_sample_mb = bytes_per_sample / (1024 * 1024)
    
    # Batch memory
    batch_memory_mb = batch_size * memory_per_sample_mb
    
    # DataLoader memory (with pin_memory=True)
    dataloader_memory_mb = batch_memory_mb * 2  # 2x for prefetching
    
    print(f"   Memory per sample: {memory_per_sample_mb:.2f} MB")
    print(f"   Batch memory (size {batch_size}): {batch_memory_mb:.2f} MB")
    print(f"   DataLoader memory: {dataloader_memory_mb:.2f} MB")
    
    # 3. Dataset Size Estimation
    print("\n🗂️ DATASET SIZE ESTIMATION:")
    
    # Assuming you have multiple sessions
    estimated_sessions = 100  # Conservative estimate
    recordings_per_session = 1000  # Conservative estimate
    
    total_recordings = estimated_sessions * recordings_per_session
    subset_recordings = int(total_recordings * subset_data)
    
    total_dataset_memory_gb = (total_recordings * memory_per_sample_mb) / 1024
    subset_dataset_memory_gb = (subset_recordings * memory_per_sample_mb) / 1024
    
    print(f"   Estimated total recordings: {total_recordings:,}")
    print(f"   Subset recordings (10%): {subset_recordings:,}")
    print(f"   Total dataset memory: {total_dataset_memory_gb:.2f} GB")
    print(f"   Subset dataset memory: {subset_dataset_memory_gb:.2f} GB")
    
    # 4. System Memory Requirements
    print("\n💻 SYSTEM MEMORY REQUIREMENTS:")
    
    # Base system memory
    system_memory_gb = 2.0  # OS, Python, etc.
    
    # PyTorch overhead
    pytorch_overhead_gb = 1.0
    
    # CUDA memory
    cuda_overhead_gb = 0.5
    
    # Total system overhead
    total_overhead_gb = system_memory_gb + pytorch_overhead_gb + cuda_overhead_gb
    
    print(f"   System overhead: {total_overhead_gb:.2f} GB")
    print(f"   - OS + Python: {system_memory_gb:.2f} GB")
    print(f"   - PyTorch: {pytorch_overhead_gb:.2f} GB")
    print(f"   - CUDA: {cuda_overhead_gb:.2f} GB")
    
    # 5. Total Memory Requirements
    print("\n🎯 TOTAL MEMORY REQUIREMENTS:")
    
    # Training memory
    training_memory_fp32 = model_memory_fp32 + (dataloader_memory_mb / 1024) + total_overhead_gb
    training_memory_fp16 = model_memory_fp16 + (dataloader_memory_mb / 1024) + total_overhead_gb
    
    # Peak memory (during data loading + training)
    peak_memory_fp32 = training_memory_fp32 + subset_dataset_memory_gb
    peak_memory_fp16 = training_memory_fp16 + subset_dataset_memory_gb
    
    print(f"   Training memory (FP32): {training_memory_fp32:.2f} GB")
    print(f"   Training memory (FP16): {training_memory_fp16:.2f} GB")
    print(f"   Peak memory (FP32): {peak_memory_fp32:.2f} GB")
    print(f"   Peak memory (FP16): {peak_memory_fp16:.2f} GB")
    
    # 6. Resource Assessment
    print("\n✅ RESOURCE ASSESSMENT:")
    your_allocation = 50  # GB
    
    print(f"   Your allocation: {your_allocation} GB")
    print(f"   Required (FP32): {peak_memory_fp32:.2f} GB")
    print(f"   Required (FP16): {peak_memory_fp16:.2f} GB")
    
    # Safety margin
    safety_margin = 0.8  # Use 80% of allocated memory
    safe_allocation = your_allocation * safety_margin
    
    print(f"   Safe allocation (80%): {safe_allocation:.2f} GB")
    
    if peak_memory_fp32 <= safe_allocation:
        print("   ✅ FP32 training should work safely")
    else:
        print("   ⚠️ FP32 training may exceed safe limits")
    
    if peak_memory_fp16 <= safe_allocation:
        print("   ✅ FP16 training should work safely")
    else:
        print("   ⚠️ FP16 training may exceed safe limits")
    
    # 7. CPU Requirements
    print("\n🖥️ CPU REQUIREMENTS:")
    your_cpus = 32
    
    # CPU usage breakdown
    data_loading_cpus = 1  # num_workers = 0
    training_cpus = 1      # Single GPU training
    system_cpus = 2        # OS and other processes
    
    total_cpu_usage = data_loading_cpus + training_cpus + system_cpus
    
    print(f"   Your allocation: {your_cpus} CPUs")
    print(f"   Required: {total_cpu_usage} CPUs")
    print(f"   - Data loading: {data_loading_cpus} CPU")
    print(f"   - Training: {training_cpus} CPU")
    print(f"   - System: {system_cpus} CPUs")
    
    if total_cpu_usage <= your_cpus:
        print("   ✅ CPU allocation is sufficient")
    else:
        print("   ⚠️ CPU allocation may be insufficient")
    
    # 8. Recommendations
    print("\n💡 OPTIMIZATION RECOMMENDATIONS:")
    
    if peak_memory_fp32 > safe_allocation:
        print("   🔧 Enable mixed precision training (FP16)")
        print("   🔧 Reduce batch size from 16 to 8 or 4")
        print("   🔧 Reduce subset_data from 0.1 to 0.05")
        print("   🔧 Use gradient checkpointing")
    
    print("   🔧 Monitor memory usage with: nvidia-smi")
    print("   🔧 Use torch.cuda.empty_cache() periodically")
    print("   🔧 Consider using num_workers=1 for faster data loading")
    
    # 9. Final Assessment
    print("\n🎯 FINAL ASSESSMENT:")
    
    if peak_memory_fp16 <= safe_allocation:
        print("   ✅ RESOURCES ARE SUFFICIENT for FP16 training")
        print("   ✅ Your 50GB RAM + 32 CPUs should work well")
        print("   ✅ Consider enabling mixed precision for efficiency")
    elif peak_memory_fp32 <= safe_allocation:
        print("   ✅ RESOURCES ARE SUFFICIENT for FP32 training")
        print("   ✅ Your 50GB RAM + 32 CPUs should work well")
        print("   ⚠️ Monitor memory usage closely")
    else:
        print("   ⚠️ RESOURCES MAY BE INSUFFICIENT")
        print("   🔧 Reduce batch size or subset_data")
        print("   🔧 Enable mixed precision training")
    
    return peak_memory_fp32, peak_memory_fp16

if __name__ == "__main__":
    analyze_resources()
