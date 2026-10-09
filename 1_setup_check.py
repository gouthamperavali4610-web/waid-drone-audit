"""
1_setup_check.py
-----------------
Run this FIRST. It confirms your RTX 4070 is actually being used.

Why this matters: if PyTorch was installed without CUDA support, everything
still "works" -- it just runs on your CPU instead of your GPU, which turns a
30-minute training run into a 20-hour one. This script catches that.
"""

print("=" * 60)
print("SETUP CHECK")
print("=" * 60)

# --- Check PyTorch and the GPU ------------------------------------------
try:
    import torch
except ImportError:
    print("\n[X] PyTorch is not installed.")
    print("    Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    raise SystemExit(1)

print(f"\nPyTorch version : {torch.__version__}")

# torch.cuda.is_available() is the single most important line here.
# True  = your GPU will be used. Good.
# False = you're on CPU. Fix it before continuing.
cuda_ok = torch.cuda.is_available()
print(f"CUDA available  : {cuda_ok}")

if cuda_ok:
    # torch.cuda.get_device_name(0) prints the name of your first GPU.
    print(f"GPU detected    : {torch.cuda.get_device_name(0)}")
    # Total video memory, in gigabytes. Your 4070 laptop chip likely shows ~8 GB.
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM            : {vram_gb:.1f} GB")
    print("\n[OK] Your GPU is ready. Training will take ~20-40 minutes.")
else:
    print("\n[X] PROBLEM: PyTorch cannot see your GPU.")
    print("    You almost certainly installed the CPU-only build.")
    print("    Fix it with these two commands:")
    print("      pip uninstall torch torchvision -y")
    print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    print("    Then run this script again.")

# --- Check the other packages -------------------------------------------
print("\n" + "-" * 60)
print("Other packages:")

# We check each one separately so you know exactly which is missing.
for package_name, import_name in [
    ("ultralytics", "ultralytics"),   # the YOLO library
    ("opencv-python", "cv2"),         # image editing (for the degradation experiments)
    ("matplotlib", "matplotlib"),     # plots
    ("pandas", "pandas"),             # tables / CSV handling
    ("pyyaml", "yaml"),               # reads the data.yaml config file
]:
    try:
        __import__(import_name)
        print(f"  [OK] {package_name}")
    except ImportError:
        print(f"  [X]  {package_name}  -->  pip install {package_name}")

print("\n" + "=" * 60)
