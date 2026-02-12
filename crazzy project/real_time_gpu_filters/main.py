#!/usr/bin/env python3
"""
Real-Time GPU-Accelerated Image Filter Application
Optimized for RTX 5070Ti with 12GB VRAM

Author: AI Assistant
Version: 1.0
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main() -> None:
    try:
        from src.gui_application import main as gui_main
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please install required dependencies:")
        print("pip install -r requirements.txt")
        sys.exit(1)

    print("="*60)
    print("Real-Time GPU Image Filter Application")
    print("Optimized for RTX 5070Ti (12GB VRAM)")
    print("="*60)
    print()

    # Check CUDA availability (best-effort)
    try:
        import pycuda.driver as cuda  # type: ignore
        import pycuda.autoinit  # type: ignore

        device = cuda.Device(0)
        attrs = device.get_attributes()

        print(f"GPU Device: {device.name()}")
        print(f"Compute Capability: {device.compute_capability()}")
        print(f"Total Memory: {device.total_memory() / 1024**3:.1f} GB")
        print(f"Max Threads per Block: {attrs[cuda.device_attribute.MAX_THREADS_PER_BLOCK]}")
        print()

    except Exception as e:
        print(f"Warning: CUDA not available - {e}")
        print("Falling back to CuPy/CPU if available...")
        print()

    print("Starting application...")
    gui_main()


if __name__ == "__main__":
    main()
