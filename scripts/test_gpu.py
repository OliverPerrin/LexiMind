# test_gpu.py
import torch

print("=" * 50)
print("GPU Information")
print("=" * 50)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ Memory: {gpu_memory:.2f} GB")
    
    # Test tensor creation
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = x @ y
    
    print(f"✅ CUDA operations working!")
    print(f"✅ Current memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"✅ Max memory allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
else:
    print("❌ CUDA not available!")
    print("Using CPU - training will be slow!")

print("=" * 50)