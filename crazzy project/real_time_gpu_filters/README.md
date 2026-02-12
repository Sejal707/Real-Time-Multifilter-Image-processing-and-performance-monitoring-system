# Real-Time GPU Image Filters

A real-time, GPU-accelerated image filtering app with a modern Qt GUI. It supports multiple GPU backends (CuPy preferred, PyCUDA optional) and a CPU (OpenCV) fallback, so it runs even without CUDA drivers.
CuPy- Numpy compatible GPU array lib, easy to use
PyCUDA- hard to use, write kernels ourselves, control GPU memory and kernels

## Features
- Real-time camera or video file processing
- Multiple filters with adjustable parameters:
  - Gaussian Blur (kernel size, sigma)  
  - Sobel Edge Detection
  - Bilateral Filter (radius, sigma_color, sigma_space)
  - Color Temperature (warm/cool)
  - Emboss
  - Sharpen
  - Grayscale
  - Brightness/Contrast (alpha/beta)
- Presets: Beauty, Artistic, Edge_Enhancement, Cinematic, SharpenAndTone [combinations of multiple filters with tuned parameters that give specific visual effects]
- Performance overlay and metrics (FPS, processing time(how long each filter takes), GPU memory via NVML if available)
- Threaded capture/processing for low latency 
      - To achieve real-time video (30–60 FPS), capturing and processing must happen simultaneously, not sequentially
      - Thread 1: Capture → Queue (fast) [thread continuously reads frames from the camera and pushes them into a queue]
        Thread 2: Process → Display (parallel) [thread independently pulls frames from the queue, applies filters & updates the display frame]
      - Low latency: Camera input never waits for filter processing

## Gaussian Blur
smooths (blurs) image by (weighted) averaging neighbouring pixels with Gaussian weighting [pixelvalue]
- kernel size: area of neighbourhood
- sigma: control how much blur

## Sobel Edge Detection
Detects edges by highlighting areas of high intensity change
- uses 2 3x3 convolution kernels to detect horizontal and vertical gradients

## Bilateral Filter
Smooths noise while preserving edges
- wont blur across strong edges
- pixel weight by dist and color difference
- sigma_color: how much color difference matters
- sigma_space: how far neighboring pixels affect each other

## Emboss
gives 3D look by emphasizing edges with light and shadow
- Uses a kernel that subtracts one side and brightens the other, shifting light direction

## Sharpen
Increases contrast at edges — makes image look crisper

## Grayscale
Removes color, keeping only brightness

## Brightness/Contrast
- beta: Adjusts how bright or dark the image is
- alpha: how strong the contrast is

## Beauty
soft smooth skin enhancing look
- Smooths skin using Gaussian or bilateral blur (to reduce harsh textures).
- warmth and soft light using color temperature adjustment.
- increases brightness and reduces contrast for a soft glow.

## Artistic
- bilateral filter to keep edges sharp but smooth colors → “oil painting” feel.
- emboss or edge enhancement to add texture.
- Adjust contrast to make colors pop.

## Edge_Enhancement
- Sobel edge detection or sharpen filter.
- contrast boost to make edges stand out.
- grayscale for pure edge maps.

## Cinematic
- color temperature (slightly warm midtones).
- contrast and dark shadows for drama.
- Optional Gaussian blur at small radius to soften background slightly

## SharpenAndTone
Enhances clarity while adjusting tone and brightness
- sharpen for detail enhancement
- contrast and brightness for crispness
- Optionally reduce color temperature to maintain neutral tone

## Architecture
- `src/filter_manager.py`
  - Chooses backend automatically: CuPy -> PyCUDA -> CPU
  - Manages active filter pipeline and presets
  - Tracks performance statistics
- `src/gpu_filters_cupy.py`
  - CuPy-based implementations for filters using GPU arrays and cp.ndimage
- `src/gpu_filters.py`
  - PyCUDA backend with custom CUDA kernels (optional)
- `src/video_processor.py`
  - Multi-threaded capture and filter application pipeline
  - Queues to decouple capture from processing
- `src/gui_application.py`
  - PyQt5 GUI, controls for filters, parameters, and performance view
- `kernels/cuda_kernels.cu`
  - CUDA kernels used by the PyCUDA backend
- `main.py`
  - Entry point, environment checks, GUI start

## cuda_kernels.cu
- defines GPU kernels for filters
- C++ kernals launched by PyCUDA
- Gaussian blur, Sobel edge, Emboss, Bilateral filter, Color temperature

## Requirements
- Python 3.8+
- Core:
  - numpy, opencv-python, PyQt5, pillow, matplotlib
- GPU (choose appropriate):
  - CuPy: install a wheel matching your CUDA, e.g. cupy-cuda12x
  - Optional: pycuda (if you want the PyCUDA backend)
- Metrics (optional): pynvml, psutil

Install base dependencies:
```bash
pip install -r requirements.txt
```

CuPy note: pick the package matching your CUDA version, e.g. for CUDA 12.x:
```bash
pip install cupy-cuda12x
```
For other CUDA versions see CuPy docs.

Optional PyCUDA:
```bash
pip install pycuda
```

## Run
- From the project root:
```bash
python real_time_gpu_filters/main.py
```
- Or via console script after installing the package (editable install recommended):
```bash
pip install -e .
gpu-filters
```

## Using the App
- Click “Start Camera” to process live video, or “Load Video” to select a file.
- Add filters from the Filters tab:
  - Gaussian Blur: set kernel size and sigma, click Apply
  - Color Temperature: move slider, click Apply
  - Bilateral Filter: set radius, click Apply
  - Edge/Emboss/Sharpen/Grayscale: toggle buttons to add
  - Brightness/Contrast: adjust sliders, click Apply
- Clear All Filters removes the current pipeline.
- Performance tab shows FPS, GPU usage/memory (if NVML available), and per-filter times.

## Notes and Tips
- Backend priority: CuPy -> PyCUDA -> CPU. If GPU libs are missing, the app continues on CPU.
- Camera settings target 1080p@60 FPS; adjust in `video_processor.py` if needed.
- For best performance, use a recent NVIDIA GPU with matching CUDA/CuPy build.

## Development
- Lint/type-check (example):
```bash
python -m pip install ruff mypy
ruff check .
```
- Package install for development:
```bash
pip install -e .
```

## License
MIT
