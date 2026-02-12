import numpy as np
import cupy as cp
from cupyx.scipy import ndimage as cnd
import cv2
from typing import Dict, Tuple, Optional
import time

class CuPyFilterManager:
    def __init__(self):
        # Initialize CuPy
        self.device = cp.cuda.Device(0)
        self.device.use()
        
        # Memory pool for efficient memory management
        self.mempool = cp.get_default_memory_pool()
        self.pinned_mempool = cp.get_default_pinned_memory_pool()
        
        self.current_frame_size = None
        self.gpu_memory = {}
        
    def allocate_gpu_memory(self, frame_shape: Tuple[int, int, int]):
        """Allocate GPU memory for frame processing"""
        if self.current_frame_size == frame_shape:
            return
            
        height, width, channels = frame_shape
        
        # Allocate GPU arrays
        self.gpu_memory = {
            'input': cp.zeros((height, width, channels), dtype=cp.uint8),
            'output': cp.zeros((height, width, channels), dtype=cp.uint8),
            'temp': cp.zeros((height, width, channels), dtype=cp.uint8),
            'float_temp': cp.zeros((height, width, channels), dtype=cp.float32)
        }
        
        self.current_frame_size = frame_shape
        
    def apply_gaussian_blur(self, frame: np.ndarray, kernel_size: int = 15, sigma: float = 2.0) -> np.ndarray:
        """Apply Gaussian blur using CuPy"""
        self.allocate_gpu_memory(frame.shape)
        
        # Copy to GPU
        gpu_frame = cp.asarray(frame)
        
        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel_cupy(kernel_size, sigma)
        
        # Apply convolution to each channel
        result = cp.zeros_like(gpu_frame)
        for c in range(frame.shape[2]):
            result[:, :, c] = cnd.convolve(
                gpu_frame[:, :, c].astype(cp.float32), 
                kernel, 
                mode='reflect'
            ).astype(cp.uint8)
        
        return cp.asnumpy(result)
        
    def apply_sobel_edge(self, frame: np.ndarray) -> np.ndarray:
        """Apply Sobel edge detection using CuPy"""
        self.allocate_gpu_memory(frame.shape)
        
        gpu_frame = cp.asarray(frame)
        
        # Sobel kernels
        sobel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=cp.float32)
        sobel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=cp.float32)
        
        result = cp.zeros_like(gpu_frame)
        
        for c in range(frame.shape[2]):
            # Convert to float for processing
            channel = gpu_frame[:, :, c].astype(cp.float32)
            
            # Apply Sobel filters
            grad_x = cnd.convolve(channel, sobel_x, mode='reflect')
            grad_y = cnd.convolve(channel, sobel_y, mode='reflect')
            
            # Calculate magnitude
            magnitude = cp.sqrt(grad_x**2 + grad_y**2)
            result[:, :, c] = cp.clip(magnitude, 0, 255).astype(cp.uint8)
        
        return cp.asnumpy(result)
        
    def apply_color_temperature(self, frame: np.ndarray, temperature: float = 0.0) -> np.ndarray:
        """Apply color temperature adjustment using CuPy"""
        gpu_frame = cp.asarray(frame).astype(cp.float32)
        
        if temperature > 0:
            # Warm (increase red, decrease blue)
            gpu_frame[:, :, 2] *= (1.0 + temperature * 0.01)  # Red channel
            gpu_frame[:, :, 0] *= (1.0 - temperature * 0.01)  # Blue channel
        else:
            # Cool (decrease red, increase blue)
            gpu_frame[:, :, 2] *= (1.0 + temperature * 0.01)  # Red channel
            gpu_frame[:, :, 0] *= (1.0 - temperature * 0.01)  # Blue channel
        
        gpu_frame = cp.clip(gpu_frame, 0, 255)
        return cp.asnumpy(gpu_frame.astype(cp.uint8))
        
    def apply_bilateral_filter(self, frame: np.ndarray, radius: int = 5, 
                              sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
        """Apply bilateral filter using CuPy (approximation).
        Accepts `radius` to align with UI/manager API. `radius`~kernel influence size.
        """
        gpu_frame = cp.asarray(frame)
        
        # Approximate bilateral with a blend of Gaussian blurs of different sigmas
        # Scale kernel sizes heuristically from radius
        k1 = max(3, int(radius) * 2 + 1)
        k2 = max(3, int(radius) * 4 + 1)
        
        gaussian1 = self._gaussian_blur_cupy(gpu_frame, k1, sigma_space/50.0)
        gaussian2 = self._gaussian_blur_cupy(gpu_frame, k2, sigma_space/25.0)
        
        # Weight by color sigma to loosely reflect edge preservation strength
        w1 = cp.float32(min(1.0, max(0.0, sigma_color / 100.0)))
        w2 = cp.float32(1.0) - w1
        result = (gaussian1 * w1 + gaussian2 * w2).astype(cp.uint8)
        
        return cp.asnumpy(result)
        
    def apply_emboss(self, frame: np.ndarray) -> np.ndarray:
        """Apply emboss effect using CuPy"""
        gpu_frame = cp.asarray(frame)
        
        # Emboss kernel
        emboss_kernel = cp.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=cp.float32)
        
        result = cp.zeros_like(gpu_frame)
        
        for c in range(frame.shape[2]):
            channel = gpu_frame[:, :, c].astype(cp.float32)
            embossed = cp.ndimage.convolve(channel, emboss_kernel, mode='reflect')
            embossed += 128  # Add gray level
            result[:, :, c] = cp.clip(embossed, 0, 255).astype(cp.uint8)
        
        return cp.asnumpy(result)

    def apply_sharpen(self, frame: np.ndarray) -> np.ndarray:
        """Apply sharpen filter using a 3x3 kernel."""
        gpu_frame = cp.asarray(frame)
        kernel = cp.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=cp.float32)
        result = cp.zeros_like(gpu_frame)
        for c in range(frame.shape[2]):
            result[:, :, c] = cnd.convolve(
                gpu_frame[:, :, c].astype(cp.float32), kernel, mode='reflect'
            ).astype(cp.uint8)
        return cp.asnumpy(result)

    def apply_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale while keeping 3 channels."""
        gpu_frame = cp.asarray(frame).astype(cp.float32)
        # BGR weights
        gray = (gpu_frame[:, :, 0] * 0.114 + gpu_frame[:, :, 1] * 0.587 + gpu_frame[:, :, 2] * 0.299)
        gray_u8 = cp.clip(gray, 0, 255).astype(cp.uint8)
        result = cp.stack([gray_u8, gray_u8, gray_u8], axis=2)
        return cp.asnumpy(result)

    def apply_brightness_contrast(self, frame: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
        """Adjust contrast (alpha) and brightness (beta). alpha in [0.5, 2.0], beta in [-100, 100]."""
        gpu_frame = cp.asarray(frame).astype(cp.float32)
        adjusted = cp.clip(gpu_frame * float(alpha) + float(beta), 0, 255)
        return cp.asnumpy(adjusted.astype(cp.uint8))
        
    def _create_gaussian_kernel_cupy(self, size: int, sigma: float) -> cp.ndarray:
        """Create Gaussian kernel using CuPy"""
        kernel = cp.fromfunction(
            lambda x, y: (1/(2*cp.pi*sigma**2)) * 
                        cp.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
            (size, size)
        )
        return (kernel / kernel.sum()).astype(cp.float32)
        
    def _gaussian_blur_cupy(self, gpu_frame: cp.ndarray, kernel_size: int, sigma: float) -> cp.ndarray:
        """Internal Gaussian blur helper"""
        kernel = self._create_gaussian_kernel_cupy(kernel_size, sigma)
        result = cp.zeros_like(gpu_frame)
        
        for c in range(gpu_frame.shape[2]):
            result[:, :, c] = cnd.convolve(
                gpu_frame[:, :, c].astype(cp.float32), 
                kernel, 
                mode='reflect'
            )
        
        return result
        
    def get_available_filters(self) -> Dict[str, callable]:
        """Return available filter functions"""
        return {
            'gaussian_blur': self.apply_gaussian_blur,
            'sobel_edge': self.apply_sobel_edge,
            'color_temperature': self.apply_color_temperature,
            'bilateral_filter': self.apply_bilateral_filter,
            'emboss': self.apply_emboss,
            'sharpen': self.apply_sharpen,
            'grayscale': self.apply_grayscale,
            'brightness_contrast': self.apply_brightness_contrast
        }
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get GPU memory usage info"""
        return {
            'used_gb': self.mempool.used_bytes() / 1024**3,
            'total_gb': self.mempool.total_bytes() / 1024**3
        }
        
    def cleanup(self):
        """Cleanup GPU memory"""
        self.mempool.free_all_blocks()
        self.pinned_mempool.free_all_blocks()
