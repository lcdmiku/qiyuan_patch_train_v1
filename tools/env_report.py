"""Report installed packages and CUDA availability for the working conda environment."""
import importlib
import sys

print("python:", sys.executable)
print("version:", sys.version.split()[0])

for name in ["numpy", "torch", "torchvision", "cv2", "PIL", "scipy", "skimage",
             "kornia", "timm", "matplotlib", "tqdm", "pytest"]:
    try:
        module = importlib.import_module(name)
        print(f"  {name:14s} {getattr(module, '__version__', '?')}")
    except Exception as exc:
        print(f"  {name:14s} MISSING ({type(exc).__name__})")

print("-" * 50)
try:
    import torch

    print("torch cuda available:", torch.cuda.is_available())
    print("torch cuda version  :", torch.version.cuda)
    if torch.cuda.is_available():
        print("device              :", torch.cuda.get_device_name(0))
        print("capability          :", torch.cuda.get_device_capability(0))
        free, total = torch.cuda.mem_get_info()
        print(f"vram                : {free/2**30:.1f} GiB free / {total/2**30:.1f} GiB total")
    print("cudnn version       :", torch.backends.cudnn.version())
except Exception as exc:
    print("torch unavailable:", exc)
