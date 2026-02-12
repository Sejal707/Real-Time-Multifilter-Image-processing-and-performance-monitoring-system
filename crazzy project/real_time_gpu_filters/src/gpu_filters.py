import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import cv2
from typing import Dict, Tuple, Optional
import os

class GPUFilterManager:
    def __init__(self):
        self.context = cuda.Device(0).make_context()
        self.stream = cuda.Stream()
        self.filters = {}
        self.gpu_memory = {}
        self.current_frame_size = None
        self._load_cuda_kernels()
        
    def _load_cuda_kernels(self):
        """Load and compile CUDA kernels"""
        kernel_path = os.path.join(os.path.dirname(__file__), '..', 'kernels', 'cuda_kernels.cu')
        
        if not os.path.exists(kernel_path):
            # Fallback: inline CUDA code
            cuda_code = """
            #include <cuda_runtime.h>
            #include <device_launch_parameters.h>

            extern "C" {
            
            __global__ void gaussian_blur_kernel(unsigned char* input, unsigned char* output, 
                                               int width, int height, int channels, float* kernel, int kernel_size) {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                
                if (x >= width || y >= height) return;
                
                int half_kernel = kernel_size / 2;
                
                for (int c = 0; c < channels; c++) {
                    float sum = 0.0f;
                    float weight_sum = 0.0f;
                    
                    for (int ky = -half_kernel; ky <= half_kernel; ky++) {
                        for (int kx = -half_kernel; kx <= half_kernel; kx++) {
                            int nx = x + kx;
                            int ny = y + ky;
                            
                            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                                int kernel_idx = (ky + half_kernel) * kernel_size + (kx + half_kernel);
                                int pixel_idx = (ny * width + nx) * channels + c;
                                
                                float weight = kernel[kernel_idx];
                                sum += input[pixel_idx] * weight;
                                weight_sum += weight;
                            }
                        }
                    }
                    
                    int output_idx = (y * width + x) * channels + c;
                    output[output_idx] = (unsigned char)(sum / weight_sum);
                }
            }
            
            __global__ void sobel_edge_kernel(unsigned char* input, unsigned char* output, 
                                            int width, int height, int channels) {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                
                if (x >= width - 1 || y >= height - 1 || x < 1 || y < 1) return;
                
                int sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
                int sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
                
                for (int c = 0; c < channels; c++) {
                    float gx = 0.0f, gy = 0.0f;
                    
                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            int nx = x + kx;
                            int ny = y + ky;
                            int pixel_idx = (ny * width + nx) * channels + c;
                            int kernel_idx = (ky + 1) * 3 + (kx + 1);
                            
                            gx += input[pixel_idx] * sobel_x[kernel_idx];
                            gy += input[pixel_idx] * sobel_y[kernel_idx];
                        }
                    }
                    
                    float magnitude = sqrtf(gx * gx + gy * gy);
                    int output_idx = (y * width + x) * channels + c;
                    output[output_idx] = (unsigned char)fminf(255.0f, magnitude);
                }
            }
            
            __global__ void color_temperature_kernel(unsigned char* input, unsigned char* output, 
                                                   int width, int height, float temperature) {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                
                if (x >= width || y >= height) return;
                
                int idx = (y * width + x) * 3;
                
                float r = input[idx + 2];
                float g = input[idx + 1];
                float b = input[idx];
                
                if (temperature > 0) {
                    r = fminf(255.0f, r * (1.0f + temperature * 0.01f));
                    b = fmaxf(0.0f, b * (1.0f - temperature * 0.01f));
                } else {
                    r = fmaxf(0.0f, r * (1.0f + temperature * 0.01f));
                    b = fminf(255.0f, b * (1.0f - temperature * 0.01f));
                }
                
                output[idx] = (unsigned char)b;
                output[idx + 1] = (unsigned char)g;
                output[idx + 2] = (unsigned char)r;
            }
            
            __global__ void bilateral_filter_kernel(unsigned char* input, unsigned char* output,
                                                  int width, int height, int channels,
                                                  int radius, float sigma_color, float sigma_space) {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                
                if (x >= width || y >= height) return;
                
                for (int c = 0; c < channels; c++) {
                    float sum = 0.0f;
                    float weight_sum = 0.0f;
                    int center_idx = (y * width + x) * channels + c;
                    float center_value = input[center_idx];
                    
                    for (int dy = -radius; dy <= radius; dy++) {
                        for (int dx = -radius; dx <= radius; dx++) {
                            int nx = x + dx;
                            int ny = y + dy;
                            
                            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                                int neighbor_idx = (ny * width + nx) * channels + c;
                                float neighbor_value = input[neighbor_idx];
                                
                                float spatial_dist = sqrtf(dx * dx + dy * dy);
                                float spatial_weight = expf(-(spatial_dist * spatial_dist) / (2.0f * sigma_space * sigma_space));
                                
                                float color_dist = fabsf(center_value - neighbor_value);
                                float color_weight = expf(-(color_dist * color_dist) / (2.0f * sigma_color * sigma_color));
                                
                                float weight = spatial_weight * color_weight;
                                sum += neighbor_value * weight;
                                weight_sum += weight;
                            }
                        }
                    }
                    
                    output[center_idx] = (unsigned char)(sum / weight_sum);
                }
            }
            
            __global__ void emboss_kernel(unsigned char* input, unsigned char* output, 
                                        int width, int height, int channels) {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                
                if (x >= width - 1 || y >= height - 1 || x < 1 || y < 1) return;
                
                int emboss_kernel[9] = {-2, -1, 0, -1, 1, 1, 0, 1, 2};
                
                for (int c = 0; c < channels; c++) {
                    float sum = 0.0f;
                    
                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            int nx = x + kx;
                            int ny = y + ky;
                            int pixel_idx = (ny * width + nx) * channels + c;
                            int kernel_idx = (ky + 1) * 3 + (kx + 1);
                            
                            sum += input[pixel_idx] * emboss_kernel[kernel_idx];
                        }
                    }
                    
                    int result = (int)(sum + 128.0f);
                    int output_idx = (y * width + x) * channels + c;
                    output[output_idx] = (unsigned char)fmaxf(0.0f, fminf(255.0f, result));
                }
            }
            
            }
            """
        else:
            with open(kernel_path, 'r') as f:
                cuda_code = f.read()
        
        # Compile CUDA module
        self.mod = SourceModule(cuda_code)
        
        # Get kernel functions
        self.gaussian_blur = self.mod.get_function("gaussian_blur_kernel")
        self.sobel_edge = self.mod.get_function("sobel_edge_kernel")
        self.color_temperature = self.mod.get_function("color_temperature_kernel")
        self.bilateral_filter = self.mod.get_function("bilateral_filter_kernel")
        self.emboss = self.mod.get_function("emboss_kernel")
        
    def allocate_gpu_memory(self, frame_shape: Tuple[int, int, int]):
        """Allocate GPU memory for frame processing"""
        if self.current_frame_size == frame_shape:
            return
            
        # Free existing memory
        if self.gpu_memory:
            for key in self.gpu_memory:
                if self.gpu_memory[key]:
                    self.gpu_memory[key].free()
        
        height, width, channels = frame_shape
        frame_size = height * width * channels
        
        # Allocate GPU memory
        self.gpu_memory = {
            'input': cuda.mem_alloc(frame_size),
            'output': cuda.mem_alloc(frame_size),
            'temp': cuda.mem_alloc(frame_size)
        }
        
        self.current_frame_size = frame_shape
        
    def create_gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """Create Gaussian kernel for blurring"""
        kernel = np.fromfunction(
            lambda x, y: (1/(2*np.pi*sigma**2)) * 
                        np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
            (size, size)
        )
        return (kernel / kernel.sum()).astype(np.float32)
        
    def apply_gaussian_blur(self, frame: np.ndarray, kernel_size: int = 15, sigma: float = 2.0) -> np.ndarray:
        """Apply Gaussian blur filter"""
        height, width, channels = frame.shape
        self.allocate_gpu_memory(frame.shape)
        
        # Create Gaussian kernel
        kernel = self.create_gaussian_kernel(kernel_size, sigma)
        kernel_gpu = cuda.mem_alloc(kernel.nbytes)
        cuda.memcpy_htod(kernel_gpu, kernel)
        
        # Copy input to GPU
        cuda.memcpy_htod_async(self.gpu_memory['input'], frame, self.stream)
        
        # Configure kernel launch parameters
        block_size = (16, 16, 1)
        grid_size = ((width + block_size[0] - 1) // block_size[0],
                    (height + block_size[1] - 1) // block_size[1], 1)
        
        # Launch kernel
        self.gaussian_blur(
            self.gpu_memory['input'], self.gpu_memory['output'],
            np.int32(width), np.int32(height), np.int32(channels),
            kernel_gpu, np.int32(kernel_size),
            block=block_size, grid=grid_size, stream=self.stream
        )
        
        # Copy result back
        result = np.empty_like(frame)
        cuda.memcpy_dtoh_async(result, self.gpu_memory['output'], self.stream)
        self.stream.synchronize()
        
        kernel_gpu.free()
        return result
        
    def apply_sobel_edge(self, frame: np.ndarray) -> np.ndarray:
        """Apply Sobel edge detection"""
        height, width, channels = frame.shape
        self.allocate_gpu_memory(frame.shape)
        
        cuda.memcpy_htod_async(self.gpu_memory['input'], frame, self.stream)
        
        block_size = (16, 16, 1)
        grid_size = ((width + block_size[0] - 1) // block_size[0],
                    (height + block_size[1] - 1) // block_size[1], 1)
        
        self.sobel_edge(
            self.gpu_memory['input'], self.gpu_memory['output'],
            np.int32(width), np.int32(height), np.int32(channels),
            block=block_size, grid=grid_size, stream=self.stream
        )
        
        result = np.empty_like(frame)
        cuda.memcpy_dtoh_async(result, self.gpu_memory['output'], self.stream)
        self.stream.synchronize()
        
        return result
        
    def apply_color_temperature(self, frame: np.ndarray, temperature: float = 0.0) -> np.ndarray:
        """Apply color temperature adjustment"""
        height, width, channels = frame.shape
        self.allocate_gpu_memory(frame.shape)
        
        cuda.memcpy_htod_async(self.gpu_memory['input'], frame, self.stream)
        
        block_size = (16, 16, 1)
        grid_size = ((width + block_size[0] - 1) // block_size[0],
                    (height + block_size[1] - 1) // block_size[1], 1)
        
        self.color_temperature(
            self.gpu_memory['input'], self.gpu_memory['output'],
            np.int32(width), np.int32(height), np.float32(temperature),
            block=block_size, grid=grid_size, stream=self.stream
        )
        
        result = np.empty_like(frame)
        cuda.memcpy_dtoh_async(result, self.gpu_memory['output'], self.stream)
        self.stream.synchronize()
        
        return result
        
    def apply_bilateral_filter(self, frame: np.ndarray, radius: int = 5, 
                              sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
        """Apply bilateral filter for noise reduction while preserving edges"""
        height, width, channels = frame.shape
        self.allocate_gpu_memory(frame.shape)
        
        cuda.memcpy_htod_async(self.gpu_memory['input'], frame, self.stream)
        
        block_size = (16, 16, 1)
        grid_size = ((width + block_size[0] - 1) // block_size[0],
                    (height + block_size[1] - 1) // block_size[1], 1)
        
        self.bilateral_filter(
            self.gpu_memory['input'], self.gpu_memory['output'],
            np.int32(width), np.int32(height), np.int32(channels),
            np.int32(radius), np.float32(sigma_color), np.float32(sigma_space),
            block=block_size, grid=grid_size, stream=self.stream
        )
        
        result = np.empty_like(frame)
        cuda.memcpy_dtoh_async(result, self.gpu_memory['output'], self.stream)
        self.stream.synchronize()
        
        return result
        
    def apply_emboss(self, frame: np.ndarray) -> np.ndarray:
        """Apply emboss effect"""
        height, width, channels = frame.shape
        self.allocate_gpu_memory(frame.shape)
        
        cuda.memcpy_htod_async(self.gpu_memory['input'], frame, self.stream)
        
        block_size = (16, 16, 1)
        grid_size = ((width + block_size[0] - 1) // block_size[0],
                    (height + block_size[1] - 1) // block_size[1], 1)
        
        self.emboss(
            self.gpu_memory['input'], self.gpu_memory['output'],
            np.int32(width), np.int32(height), np.int32(channels),
            block=block_size, grid=grid_size, stream=self.stream
        )
        
        result = np.empty_like(frame)
        cuda.memcpy_dtoh_async(result, self.gpu_memory['output'], self.stream)
        self.stream.synchronize()
        
        return result
        
    def get_available_filters(self) -> Dict[str, callable]:
        """Return available filter functions"""
        return {
            'gaussian_blur': self.apply_gaussian_blur,
            'sobel_edge': self.apply_sobel_edge,
            'color_temperature': self.apply_color_temperature,
            'bilateral_filter': self.apply_bilateral_filter,
            'emboss': self.apply_emboss
        }

    def get_memory_info(self) -> Dict[str, float]:
        try:
            free, total = cuda.mem_get_info()
            used = total - free
            return {
                'used_gb': used / 1024**3,
                'total_gb': total / 1024**3,
            }
        except Exception:
            return {}
        
    def __del__(self):
        """Cleanup GPU resources"""
        if hasattr(self, 'gpu_memory'):
            for key in self.gpu_memory:
                if self.gpu_memory[key]:
                    self.gpu_memory[key].free()
        
        if hasattr(self, 'context'):
            self.context.pop()
