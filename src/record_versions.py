"""Print the software versions of the current environment (run it in the same Colab runtime type used for the experiments)."""
import sys, platform, importlib
print("python", sys.version.split()[0], platform.platform())
for m in ("torch", "transformers", "accelerate", "bitsandbytes", "networkx", "numpy", "tokenizers"):
    try: print(m, importlib.import_module(m).__version__)
    except Exception as e: print(m, "not installed")
try:
    import torch; print("cuda", torch.version.cuda, "| gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
except Exception: pass
